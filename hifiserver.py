import logging
import os
import re
from pathlib import Path  # path manipulation
import dataclasses
import sys
import tempfile
import traceback
import yaml

import numpy as np  # Numpy <3
import torch
import soundfile as sf  # WAV read + write
import scipy.interpolate as interp  # Interpolator for feats
import resampy  # Resampler (as in sampling rate stuff)
from http.server import BaseHTTPRequestHandler, HTTPServer
from concurrent.futures import ThreadPoolExecutor
from filelock import FileLock, Timeout

from util.load_config_from_yaml import load_config_from_yaml
from util.wav2mel import PitchAdjustableMelSpectrogram
from hnsep.nets import CascadedNet

logging.basicConfig(format='%(message)s', level=logging.INFO)

version = '0.0.6-hifisampler'
help_string = '''usage: hifisampler in_file out_file pitch velocity [flags] [offset] [length] [consonant] [cutoff] [volume] [modulation] [tempo] [pitch_string]

Resamples using the PC-NSF-HIFIGAN Vocoder.

arguments:
\tin_file\t\tPath to input file.
\tout_file\tPath to output file.
\tpitch\t\tThe pitch to render on.
\tvelocity\tThe consonant velocity of the render.

optional arguments:
\tflags\t\tThe flags of the render. But now, it's not implemented yet.
\toffset\t\tThe offset from the start of the render area of the sample. (default: 0)
\tlength\t\tThe length of the stretched area in milliseconds. (default: 1000)
\tconsonant\tThe unstretched area of the render in milliseconds. (default: 0)
\tcutoff\t\tThe cutoff from the end or from the offset for the render area of the sample. (default: 0)
\tvolume\t\tThe volume of the render in percentage. (default: 100)
\tmodulation\tThe pitch modulation of the render in percentage. (default: 0)
\ttempo\t\tThe tempo of the render. Needs to have a ! at the start. (default: !100)
\tpitch_string\tThe UTAU pitchbend parameter written in Base64 with RLE encoding. (default: AA)'''

notes = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6,
         'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}  # Note names lol
note_re = re.compile(r'([A-G]#?)(-?\d+)')  # Note Regex for conversion
cache_ext = '.hifi.npz'  # cache file extension

# Flags
flags = ['fe', 'fl', 'fo', 'fv', 'fp', 've', 'vo', 'g', 't',
         'A', 'B', 'G', 'P', 'S', 'p', 'R', 'D', 'C', 'Z', 'Hv', 'Hb', 'Ht', 'He']
flag_re = '|'.join(flags)
flag_re = f'({flag_re})([+-]?\\d+)?'
flag_re = re.compile(flag_re)

server_ready = False


@load_config_from_yaml(script_path=Path(__file__))
@dataclasses.dataclass
class Config:
    sample_rate: int = 44100  # UTAU only really likes 44.1khz
    win_size: int = 2048     # 必须和vocoder训练时一致
    hop_size: int = 512      # 必须和vocoder训练时一致
    origin_hop_size: int = 128  # 插值前的hopsize,可以适当调小改善长音的电音
    n_mels: int = 128        # 必须和vocoder训练时一致
    n_fft: int = 2048        # 必须和vocoder训练时一致
    mel_fmin: float = 40     # 必须和vocoder训练时一致
    mel_fmax: float = 16000  # 必须和vocoder训练时一致
    fill: int = 6
    vocoder_path: str = r"\path\to\your\vocoder\pc_nsf_hifigan\model.ckpt"
    model_type: str = 'ckpt'  # or 'onnx'
    hnsep_model_path: str = r"\path\to\your\hnsep\model.pt"
    wave_norm: bool = False
    trim_silence: bool = True  # 是否在响度标准化前截取无声部分
    silence_threshold: float = -52.0
    loop_mode: bool = False
    peak_limit: float = 1.0
    max_workers: int = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DotDict(dict):
    def __getattr__(*args):
        val = dict.get(*args)
        return DotDict(val) if type(val) is dict else val

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def dynamic_range_compression_torch(x, C=1, clip_val=1e-9):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def loudness_norm(
    audio: np.ndarray, rate: int, peak=-1.0, loudness=-23.0, block_size=0.400, strength=100
) -> np.ndarray:
    """
    Perform loudness normalization (ITU-R BS.1770-4) on audio files.

    Args:
        audio: audio data
        rate: sample rate
        peak: peak normalize audio to N dB. Defaults to -1.0.
        loudness: loudness normalize audio to N dB LUFS. Defaults to -23.0.
        block_size: block size for loudness measurement. Defaults to 0.400. (400 ms)
        strength: strength of the normalization. Defaults to 100.

    Returns:
        loudness normalized audio
    """

    original_length = len(audio)
    original_audio = audio.copy()  # 保存原始音频用于后续处理
    
    if Config.trim_silence:
        def get_rms_db(audio_segment):
            if len(audio_segment) == 0:
                return -np.inf
            rms = np.sqrt(np.mean(np.square(audio_segment)))
            if rms < 1e-10:  # 避免log(0)错误
                return -np.inf
            return 20 * np.log10(rms)
        
        frame_length = int(rate * 0.02)  # 20ms窗口
        hop_length = int(rate * 0.01)    # 10ms步长
        
        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i+frame_length]
            rms_db = get_rms_db(frame)
            rms_values.append(rms_db)
        
        # 使用阈值检测有声帧
        voiced_frames = [i for i, rms in enumerate(rms_values) if rms > Config.silence_threshold]
        
        if voiced_frames:
            first_voiced = voiced_frames[0]
            last_voiced = voiced_frames[-1]
            
            # 添加一些余量，避免突然截断
            padding_frames = int(rate * 0.1) // hop_length  # 添加100ms的余量
            
            # 确保不超出边界
            start_sample = max(0, first_voiced * hop_length)
            end_sample = min(len(audio), (last_voiced + 1 + padding_frames) * hop_length + frame_length)
            
            trimmed_audio = audio[start_sample:end_sample]
            logging.info(f'Trimmed silence: {len(audio)} -> {len(trimmed_audio)} samples')
            
            # 使用截取后的音频进行响度标准化
            audio = trimmed_audio
    
    # 如果音频长度小于最小块大小，进行填充
    if len(audio) < int(rate * block_size):
        padding_length = int(rate * block_size) - len(audio)
        audio = np.pad(audio, (0, padding_length), mode='reflect')

    # Measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    # Apply strength to calculate the target loudness
    final_loudness = _loudness + (loudness - _loudness) * strength / 100

    # Loudness normalize audio to [loudness] LUFS
    audio = pyln.normalize.loudness(audio, _loudness, final_loudness)

    # 如果启用了无声截取功能，需要恢复原始长度
    if Config.trim_silence:
        # 创建一个全零数组作为输出
        output = np.zeros(original_length)
        
        # 将标准化后的音频放回原位置，并添加平滑过渡
        if voiced_frames:  # 确保有声音帧存在
            start_sample = max(0, first_voiced * hop_length)
            
            # 计算需要放回的音频长度
            available_length = min(len(audio), original_length - start_sample)
            
            # 创建一个逐渐衰减的窗函数，用于音频的尾部淡出
            fade_length = min(int(rate * 0.2), available_length // 4)  # 最多200ms或音频长度的1/4
            fade_out = np.ones(available_length)
            
            if fade_length > 0:
                # 在末尾应用淡出效果
                fade_out[-fade_length:] = np.linspace(1.0, 0.0, fade_length)
            
            # 应用淡出效果并放回原位置
            output[start_sample:start_sample+available_length] = audio[:available_length] * fade_out
              # 如果原始音频有后续部分，应用交叉淡入淡出
            if start_sample + available_length < original_length:
                remain_length = original_length - (start_sample + available_length)
                crossfade_length = min(fade_length, remain_length)
                
                if crossfade_length > 0:
                    crossfade_start = start_sample + available_length
                    # 从原始音频获取剩余部分
                    remain_audio = original_audio[crossfade_start:original_length]
                    
                    # 应用淡入效果到原始音频的剩余部分
                    fade_in = np.ones(remain_length)
                    fade_in[:crossfade_length] = np.linspace(0.0, 1.0, crossfade_length)
                    
                    # 填充剩余部分
                    output[crossfade_start:original_length] = remain_audio * fade_in
        else:  # 如果没有声音帧，直接返回原始音频
            output = audio[:original_length]
        
        audio = output

    # 如果原始音频短于block_size，裁剪回原始长度
    if original_length < int(rate * block_size):
        audio = audio[:original_length]

    return audio


def load_sep_model(model_path, device='cpu'):
    model_dir = os.path.dirname(os.path.abspath(model_path))
    config_file = os.path.join(model_dir, 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    model = CascadedNet(
        args.n_fft,
        args.hop_length,
        args.n_out,
        args.n_out_lstm,
        True,
        is_mono=args.is_mono,
        fixed_length=True if args.fixed_length is None else args.fixed_length)
    model.to(device)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model, args


def pre_emphasis_base_tension(wave, b):
    """
    Args:
        wave: [1, 1, t]
    """
    original_length = wave.size(-1)
    pad_length = (Config.hop_size - (original_length %
                  Config.hop_size)) % Config.hop_size
    wave = torch.nn.functional.pad(
        wave, (0, pad_length), mode='constant', value=0)
    wave = wave.squeeze(1)

    spec = torch.stft(
        wave,
        Config.n_fft,
        hop_length=Config.hop_size,
        win_length=Config.win_size,
        window=torch.hann_window(Config.win_size).to(wave.device),
        return_complex=True
    )
    spec_amp = torch.abs(spec)
    spec_phase = torch.atan2(spec.imag, spec.real)

    spec_amp_db = torch.log(torch.clamp(spec_amp, min=1e-9))

    fft_bin = Config.n_fft // 2 + 1
    x0 = fft_bin / ((Config.sample_rate / 2) / 1500)
    freq_filter = (-b / x0) * torch.arange(0, fft_bin, device=wave.device) + b
    spec_amp_db = spec_amp_db + \
        torch.clamp(freq_filter, min=-2, max=2).unsqueeze(0).unsqueeze(2)

    spec_amp = torch.exp(spec_amp_db)

    filtered_wave = torch.istft(
        torch.complex(spec_amp * torch.cos(spec_phase),
                      spec_amp * torch.sin(spec_phase)),
        n_fft=Config.n_fft,
        hop_length=Config.hop_size,
        win_length=Config.win_size,
        window=torch.hann_window(Config.win_size).to(wave.device)
    )

    original_max = torch.max(torch.abs(wave))
    filtered_max = torch.max(torch.abs(filtered_wave))
    filtered_wave = filtered_wave * \
        (original_max / filtered_max) * (np.clip(b/(-15), 0, 0.33) + 1)
    filtered_wave = filtered_wave.unsqueeze(1)
    filtered_wave = filtered_wave[:, :, :original_length]

    return filtered_wave

# Pitch string interpreter


def to_uint6(b64):
    """Convert one Base64 character to an unsigned integer.

    Parameters
    ----------
    b64 : str
        The Base64 character.

    Returns
    -------
    int
        The equivalent of the Base64 character as an integer.
    """
    c = ord(b64)  # Convert based on ASCII mapping
    if c >= 97:
        return c - 71
    elif c >= 65:
        return c - 65
    elif c >= 48:
        return c + 4
    elif c == 43:
        return 62
    elif c == 47:
        return 63
    else:
        raise Exception


def to_int12(b64):
    """Converts two Base64 characters to a signed 12-bit integer.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    int
        The equivalent of the Base64 characters as a signed 12-bit integer (-2047 to 2048)
    """
    uint12 = to_uint6(b64[0]) << 6 | to_uint6(
        b64[1])  # Combined uint6 to uint12
    if uint12 >> 11 & 1 == 1:  # Check most significant bit to simulate two's complement
        return uint12 - 4096
    else:
        return uint12


def to_int12_stream(b64):
    """Converts a Base64 string to a list of integers.

    Parameters
    ----------
    b64 : str
        The Base64 string.

    Returns
    -------
    list
        The equivalent of the Base64 string if split every 12-bits and interpreted as a signed 12-bit integer.
    """
    res = []
    for i in range(0, len(b64), 2):
        res.append(to_int12(b64[i:i+2]))
    return res


def pitch_string_to_cents(x):
    """Converts UTAU's pitchbend argument to an ndarray representing the pitch offset in cents.

    Parameters
    ----------
    x : str
        The pitchbend argument.

    Returns
    -------
    ndarray
        The pitchbend argument as pitch offset in cents.
    """
    pitch = x.split('#')  # Split RLE Encoding
    res = []
    for i in range(0, len(pitch), 2):
        # Go through each pair
        p = pitch[i:i+2]
        if len(p) == 2:
            # Decode pitch string and extend RLE
            pitch_str, rle = p
            res.extend(to_int12_stream(pitch_str))
            res.extend([res[-1]] * int(rle))
        else:
            # Decode last pitch string without RLE if it exists
            res.extend(to_int12_stream(p[0]))
    res = np.array(res, dtype=np.int32)
    if np.all(res == res[0]):
        return np.zeros(res.shape)
    else:
        return np.concatenate([res, np.zeros(1)])

# Pitch conversion


def note_to_midi(x):
    """Note name to MIDI note number."""
    note, octave = note_re.match(x).group(1, 2)
    octave = int(octave) + 1
    return octave * 12 + notes[note]


def midi_to_hz(x):
    """MIDI note number to Hertz using equal temperament. A4 = 440 Hz."""
    return 440 * np.exp2((x - 69) / 12)

# WAV read/write


def read_wav(loc):
    """Read audio files supported by soundfile and resample to 44.1kHz if needed. Mixes down to mono if needed.

    Parameters
    ----------
    loc : str or file
        Input audio file.

    Returns
    -------
    ndarray
        Data read from WAV file remapped to [-1, 1] and in 44.1kHz
    """
    if type(loc) == str:  # make sure input is Path
        loc = Path(loc)

    exists = loc.exists()
    if not exists:  # check for alternative files
        for ext in sf.available_formats().keys():
            loc = loc.with_suffix('.' + ext.lower())
            exists = loc.exists()
            if exists:
                break

    if not exists:
        raise FileNotFoundError("No supported audio file was found.")

    x, fs = sf.read(str(loc))
    if len(x.shape) == 2:
        # Average all channels... Probably not too good for formats bigger than stereo
        x = np.mean(x, axis=1)

    if fs != Config.sample_rate:
        x = resampy.resample(x, fs, Config.sample_rate)

    return x


def save_wav(loc, x):
    """Save data into a WAV file.

    Parameters
    ----------
    loc : str or file
        Output WAV file.

    x : ndarray
        Audio data in 44.1kHz within [-1, 1].

    Returns
    -------
    None
    """
    try:
        sf.write(str(loc), x, Config.sample_rate, 'PCM_16')
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")


class Resampler:
    """
    A class for the UTAU resampling process.

    Attributes
    ----------
    in_file : str
        Path to input file.

    out_file : str
        Path to output file.

    pitch : str
        The pitch of the note.

    velocity : str or float
        The consonant velocity of the note.

    flags : str
        The flags of the note.

    offset : str or float
        The offset from the start for the render area of the sample.

    length : str or int
        The length of the stretched area in milliseconds.

    consonant : str or float
        The unstretched area of the render.

    cutoff : str or float
        The cutoff from the end or from the offset for the render area of the sample.

    volume : str or float
        The volume of the note in percentage.

    modulation : str or float
        The modulation of the note in percentage.

    tempo : str
        The tempo of the note.

    pitch_string : str
        The UTAU pitchbend parameter.

    Methods
    -------
    render(self):
        The rendering workflow. Immediately starts when class is initialized.

    get_features(self):
        Gets the MEL features either from a cached file or generating it if it doesn't exist.

    generate_features(self, features_path):
        Generates MEL features and saves it for later.

    resample(self, features):
        Renders a WAV file using the passed MEL features.
    """

    def __init__(self, in_file, out_file, pitch, velocity, flags='', offset=0, length=1000, consonant=0, cutoff=0, volume=100, modulation=0, tempo='!100', pitch_string='AA'):
        """Initializes the renderer and immediately starts it.

        Parameters
        ---------
        in_file : str
            Path to input file.

        out_file : str
            Path to output file.

        pitch : str
            The pitch of the note.

        velocity : str or float
            The consonant velocity of the note.

        flags : str
            The flags of the note.

        offset : str or float
            The offset from the start for the render area of the sample.

        length : str or int
            The length of the stretched area in milliseconds.

        consonant : str or float
            The unstretched area of the render.

        cutoff : str or float
            The cutoff from the end or from the offset for the render area of the sample.

        volume : str or float
            The volume of the note in percentage.

        modulation : str or float
            The modulation of the note in percentage.

        tempo : str
            The tempo of the note.

        pitch_string : str
            The UTAU pitchbend parameter.
        """
        self.in_file = Path(in_file)
        self.out_file = out_file
        self.pitch = note_to_midi(pitch)
        self.velocity = float(velocity)
        self.flags = {k: int(v) if v else None for k,
                      v in flag_re.findall(flags.replace('/', ''))}
        self.offset = float(offset)
        self.length = int(length)
        self.consonant = float(consonant)
        self.cutoff = float(cutoff)
        self.volume = float(volume)
        self.modulation = float(modulation)
        self.tempo = float(tempo[1:])
        self.pitchbend = pitch_string_to_cents(pitch_string)

        self.render()

    def render(self):
        """The rendering workflow. Immediately starts when class is initialized.

        Parameters
        ----------
        None
        """
        features = self.get_features()
        self.resample(features)

    def get_features(self):
        """Gets the MEL features either from a cached file or generating it if it doesn't exist.

        Parameters
        ----------
        None

        Returns
        -------
        features : dict
            A dictionary of the MEL.
        """
        features_path = self.in_file.with_suffix(cache_ext)

        self.flags['Hb'] = self.flags.get('Hb', 100)
        self.flags['Hv'] = self.flags.get('Hv', 100)
        self.flags['Ht'] = self.flags.get('Ht', 0)
        self.flags['g'] = self.flags.get('g', 0)

        flag_suffix = '_'.join(f"{k}{v if v is not None else ''}" for k, v in sorted(
            self.flags.items()) if k in ['Hb', 'Hv', 'Ht', 'g'])
        if flag_suffix:
            features_path = features_path.with_name(
                f'{self.in_file.stem}_{flag_suffix}{cache_ext}')
        else:
            features_path = features_path.with_name(
                f'{self.in_file.stem}{cache_ext}')

        lock_path = str(features_path) + ".lock"

        lock = FileLock(lock_path, timeout=60)  # 设置60秒超时
        features = None  # 初始化 features 变量

        try:
            with lock:
                force_generate = 'G' in self.flags.keys()

                if force_generate:
                    logging.info('G flag exists. Forcing feature generation.')
                    features = self.generate_features(features_path)
                elif features_path.exists():
                    try:
                        features = np.load(str(features_path))
                        logging.info('Cache loaded successfully.')
                    except (EOFError, OSError, ValueError) as e:
                        logging.warning(
                            f'Failed to load cache {features_path} ({type(e).__name__}: {e}). Regenerating...')
                        try:
                            os.remove(features_path)
                        except OSError as rm_err:
                            logging.error(
                                f"Could not remove corrupted cache file {features_path}: {rm_err}")
                        # 在锁内重新生成
                        features = self.generate_features(features_path)
                else:
                    logging.info(
                        f'{features_path} not found. Generating features.')
                    features = self.generate_features(features_path)

                logging.info(f'File lock released for {lock_path}')
            # 自动释放锁

        except Timeout:
            logging.error(
                f"Could not acquire lock for {lock_path} within 60 seconds!")
            raise RuntimeError(
                f"Failed to acquire cache lock for {features_path}")

        # 确保 features 被成功赋值
        if features is None:
            logging.error(
                f"Logic error: Features could not be loaded or generated for {features_path}")
            raise RuntimeError(f"Could not get features for {features_path}")

        return features

    def generate_features(self, features_path):
        """Generates PC-NSF-hifigan features and saves it for later.

        Parameters
        ----------
        features_path : str or file
            The path for caching the features.

        Returns
        -------
        features : dict
            A dictionary of the MEL.
        """
        wave = read_wav(self.in_file)
        wave = torch.from_numpy(wave).to(
            dtype=torch.float32, device=Config.device).unsqueeze(0).unsqueeze(0)
        print(wave.shape)

        breath = self.flags.get("Hb", 100)
        voicing = self.flags.get("Hv", 100)
        tension = self.flags.get("Ht", 0)
        print(f'breath: {breath}, voicing: {voicing}, tension: {tension}')
        if breath != 100 or voicing != 100 or tension != 0:
            logging.info(
                'Hb or Hv or Ht flag exists. Split audio into breath, voicing')

            hnsep_cache_path = self.in_file.with_name(
                f'{self.in_file.stem}_hnsep')
            lock_path = str(hnsep_cache_path) + ".lock"
            lock_hnsep = FileLock(lock_path, timeout=60)  # 设置60秒超时
            
            seg_output = None  # 初始化 features 变量
            re_generate_hnsep = True
            try:
                with lock_hnsep:
                    force_generate = 'G' in self.flags.keys()

                    if force_generate:
                        logging.info('G flag exists. Forcing hnsep feature generation.')
                        with torch.no_grad():
                            seg_output = hnsep_model.predict_fromaudio(wave)  # 预测谐波
                    elif hnsep_cache_path.exists():
                        try:
                            seg_output = torch.load(str(hnsep_cache_path), map_location=Config.device)
                            logging.info('Cache loaded seg_output successfully.')
                            re_generate_hnsep = False
                        except (EOFError, OSError, ValueError) as e:
                            logging.warning(
                                f'Failed to load cache {hnsep_cache_path} ({type(e).__name__}: {e}). Regenerating...')
                            try:
                                os.remove(hnsep_cache_path)
                            except OSError as rm_err:
                                logging.error(
                                    f"Could not remove corrupted cache file {hnsep_cache_path}: {rm_err}")
                            # 在锁内重新生成
                            with torch.no_grad():
                                seg_output = hnsep_model.predict_fromaudio(wave)
                    else:
                        logging.info(
                            f'{hnsep_cache_path} not found. Generating features.')
                        with torch.no_grad():
                            seg_output = hnsep_model.predict_fromaudio(wave)
                    
                    if re_generate_hnsep:
                        
                        # 原子写入
                        temp_suffix = ".hnsep_tmp"
                        temp_path = hnsep_cache_path.with_suffix(
                            hnsep_cache_path.suffix + temp_suffix)
                        print("temp_path:",temp_path)

                        try:
                            #np.savez_compressed(str(temp_path), **features)
                            torch.save(seg_output, str(temp_path))
                            os.replace(str(temp_path), str(hnsep_cache_path))
                            logging.info(f'Hnsep features saved successfully to {hnsep_cache_path}')
                        except Exception as e:
                            logging.error(
                                f'Error during saving/renaming cache file {hnsep_cache_path}: {e}', exc_info=True)

                            if temp_path.exists():
                                try:
                                    os.remove(str(temp_path))
                                    logging.info(
                                        f'Removed temporary file {temp_path} after error.')
                                except OSError as rm_err:
                                    logging.error(
                                        f"Could not remove temporary file {temp_path} after error: {rm_err}")
                            raise
                        
                    logging.info(f'File lock released for {lock_path}')
                # 自动释放锁

            except Timeout:
                logging.error(
                    f"Could not acquire lock for {lock_path} within 60 seconds!")
                raise RuntimeError(
                    f"Failed to acquire cache lock for {seg_output}")

            # 确保 seg_output 被成功赋值
            if seg_output is None:
                logging.error(
                    f"Logic error: Features could not be loaded or generated for {hnsep_cache_path}")
                raise RuntimeError(f"Could not get features for {hnsep_cache_path}")

            breath = np.clip(breath, 0, 500)
            voicing = np.clip(voicing, 0, 150)
            if tension != 0:
                tension = np.clip(tension, -100, 100)
                wave = (breath/100)*(wave - seg_output) + \
                    pre_emphasis_base_tension(
                        (voicing/100)*seg_output, -tension/50)
            else:
                wave = (breath/100)*(wave - seg_output) + \
                    (voicing/100)*seg_output
        wave = wave.squeeze(0).squeeze(0).cpu().numpy()
        wave = torch.from_numpy(wave).to(
            dtype=torch.float32, device=Config.device).unsqueeze(0)  # 默认不缩放
        wave_max = torch.max(torch.abs(wave))
        if wave_max >= 0.5:
            logging.info('The audio volume is too high. Scaling down to 0.5')
            # 先缩小到最大0.5
            scale = 0.5 / wave_max
            wave = wave * scale
            scale = scale.item()
        else:
            logging.info('The audio volume is already low enough')
            scale = 1.0

        gender = self.flags.get("g", 0)
        gender = np.clip(gender, -600, 600)
        logging.info(f'gender: {gender}')

        mel_origin = melAnalysis(
            wave,
            gender/100, 1).squeeze()
        logging.info(f'mel_origin: {mel_origin.shape}')
        mel_origin = dynamic_range_compression_torch(mel_origin).cpu().numpy()
        logging.info('Saving features.')

        features = {'mel_origin': mel_origin, 'scale': scale}

        # 原子写入
        temp_suffix = ".tmp"
        temp_path = features_path.with_suffix(
            features_path.suffix + temp_suffix)

        try:
            np.savez_compressed(str(temp_path), **features)
            os.replace(str(temp_path) + '.npz', str(features_path))
            logging.info(f'Features saved successfully to {features_path}')
        except Exception as e:
            logging.error(
                f'Error during saving/renaming cache file {features_path}: {e}', exc_info=True)

            if temp_path.exists():
                try:
                    os.remove(str(temp_path))
                    logging.info(
                        f'Removed temporary file {temp_path} after error.')
                except OSError as rm_err:
                    logging.error(
                        f"Could not remove temporary file {temp_path} after error: {rm_err}")
            raise

        return features

    def resample(self, features):
        """
        Renders a WAV file using the passed MEL features.

        Parameters
        ----------
        features : dict
            A dictionary of the mel.

        Returns
        -------
        None
        """
        if self.out_file == 'nul':
            logging.info('Null output file. Skipping...')
            return

        mod = self.modulation / 100
        logging.info(f"mod: {mod}")

        self.out_file = Path(self.out_file)
        wave = read_wav(Path(self.in_file))
        logging.info(f'wave: {wave.shape}')

        scale = features['scale']
        logging.info(f'scale: {scale}')

        mel_origin = features['mel_origin']
        logging.info(f'mel_origin: {mel_origin.shape}')

        thop_origin = Config.origin_hop_size / Config.sample_rate
        thop = Config.hop_size / Config.sample_rate
        logging.info(f'thop_origin: {thop_origin}')
        logging.info(f'thop: {thop}')

        t_area_origin = np.arange(
            mel_origin.shape[1]) * thop_origin + thop_origin / 2
        total_time = t_area_origin[-1] + thop_origin/2
        logging.info(f"t_area_mel_origin: {t_area_origin.shape}")
        logging.info(f"total_time: {total_time}")

        vel = np.exp2(1 - self.velocity / 100)
        offset = self.offset / 1000  # start time
        cutoff = self.cutoff / 1000  # end time
        start = offset
        logging.info(f'vel:{vel}')
        logging.info(f'offset:{offset}')
        logging.info(f'cutoff:{cutoff}')

        logging.info('Calculating timing.')
        if self.cutoff < 0:  # deal with relative end time
            end = start - cutoff  # ???
        else:
            end = total_time - cutoff
        con = start + self.consonant / 1000
        logging.info(f'start:{start}')
        logging.info(f'end:{end}')
        logging.info(f'con:{con}')

        logging.info('Preparing interpolators.')

        length_req = self.length / 1000
        stretch_length = end - con
        logging.info(f'length_req: {length_req}')
        logging.info(f'stretch_length: {stretch_length}')

        if Config.loop_mode or "He" in self.flags.keys():
            # 添加循环拼接模式
            logging.info('Looping.')
            logging.info(
                f'con_mel_frame: {int((con + thop_origin/2)//thop_origin)}')
            mel_loop = mel_origin[:, int(
                (con + thop_origin/2)//thop_origin):int((end + thop_origin/2)//thop_origin)]
            logging.info(f'mel_loop: {mel_loop.shape}')
            pad_loop_size = length_req//thop_origin + 1
            logging.info(f'pad_loop_size: {pad_loop_size}')
            padded_mel = np.pad(mel_loop, pad_width=(
                (0, 0), (0, int(pad_loop_size))), mode='reflect')  # 多pad一点
            logging.info(f'padded_mel: {padded_mel.shape}')
            mel_origin = np.concatenate(
                (mel_origin[:, :int((con + thop_origin/2)//thop_origin)], padded_mel), axis=1)
            logging.info(f'mel_origin: {mel_origin.shape}')
            stretch_length = pad_loop_size*thop_origin
            t_area_origin = np.arange(
                mel_origin.shape[1]) * thop_origin + thop_origin / 2
            total_time = t_area_origin[-1] + thop_origin/2
            logging.info(f'new_total_time: {total_time}')

        # Make interpolators to render new areas
        mel_interp = interp.interp1d(t_area_origin, mel_origin, axis=1)

        if stretch_length < length_req:
            logging.info('stretch_length < length_req')
            scaling_ratio = length_req / stretch_length
        else:
            logging.info('stretch_length >= length_req, no stretching needed.')
            scaling_ratio = 1

        def stretch(t, con, scaling_ratio):
            return np.where(t < vel*con, t/vel, con + (t - vel*con) / scaling_ratio)

        stretched_n_frames = (con*vel + (total_time - con)
                              * scaling_ratio) // thop + 1
        stretched_t_mel = np.arange(stretched_n_frames) * thop + thop / 2
        logging.info(f'stretched_n_frames: {stretched_n_frames}')
        logging.info(f'stretched_t_mel: {stretched_t_mel.shape}')

        # 在start左边的mel帧数
        start_left_mel_frames = (start*vel + thop/2)//thop
        if start_left_mel_frames > Config.fill:
            cut_left_mel_frames = start_left_mel_frames - Config.fill
        else:
            cut_left_mel_frames = 0
        logging.info(f'start_left_mel_frames: {start_left_mel_frames}')
        logging.info(f'cut_left_mel_frames: {cut_left_mel_frames}')

        # 在length_req+con右边的mel帧数
        end_right_mel_frames = stretched_n_frames - \
            (length_req+con*vel + thop/2)//thop
        if end_right_mel_frames > Config.fill:
            cut_right_mel_frames = end_right_mel_frames - Config.fill
        else:
            cut_right_mel_frames = 0
        logging.info(f'end_right_mel_frames: {end_right_mel_frames}')
        logging.info(f'cut_right_mel_frames: {cut_right_mel_frames}')

        logging.info(f'length_req: {length_req}')
        logging.info(f'stretch_length: {stretch_length}')
        logging.info(
            f'(length_req+con*vel + thop/2)//thop: {(length_req+con*vel + thop/2)//thop}')

        stretched_t_mel = stretched_t_mel[int(cut_left_mel_frames):int(
            stretched_n_frames-cut_right_mel_frames)]
        logging.info(f'stretched_t_mel: {stretched_t_mel.shape}')

        stretch_t_mel = np.clip(
            stretch(stretched_t_mel, con, scaling_ratio), 0, t_area_origin[-1])
        logging.info(f'stretch_t_mel: {stretch_t_mel.shape}')

        new_start = start*vel - cut_left_mel_frames * thop
        new_end = (length_req+con*vel) - cut_left_mel_frames * thop
        logging.info(f'new_start: {new_start}')
        logging.info(f'new_end: {new_end}')
        logging.info(f'stretched_t_mel[0]: {stretched_t_mel[0]}')
        logging.info(f'stretched_t_mel[-1]: {stretched_t_mel[-1]}')

        mel_render = mel_interp(stretch_t_mel)
        logging.info(f'mel_render: {mel_render.shape}')

        t = np.arange(mel_render.shape[1]) * thop
        logging.info(f't: {t.shape}')
        logging.info('Calculating pitch.')
        # Calculate pitch in MIDI note number terms
        pitch = self.pitchbend / 100 + self.pitch
        if "t" in self.flags.keys() and self.flags["t"]:
            pitch = pitch + self.flags["t"] / 100
        t_pitch = 60 * np.arange(len(pitch)) / (self.tempo * 96) + new_start
        pitch_interp = interp.Akima1DInterpolator(t_pitch, pitch)
        pitch_render = pitch_interp(np.clip(t, new_start, t_pitch[-1]))
        f0_render = midi_to_hz(pitch_render)
        logging.info(f'f0_render: {f0_render.shape}')

        logging.info('Cutting mel and f0.')

        if Config.model_type == "ckpt":

            mel_render = torch.from_numpy(
                mel_render).unsqueeze(0).to(dtype=torch.float32)
            f0_render = torch.from_numpy(f0_render).unsqueeze(
                0).to(dtype=torch.float32)
            logging.info(f'mel_render: {mel_render.shape}')
            logging.info(f'f0_render: {f0_render.shape}')

            logging.info('Rendering audio.')

            wav_con = vocoder.spec2wav_torch(mel_render.to(
                Config.device), f0=f0_render.to(Config.device))
            render = wav_con[int(new_start * Config.sample_rate)                             :int(new_end * Config.sample_rate)].to('cpu').numpy()
            logging.info(f'cut_l:{int(new_start * Config.sample_rate)}')
            logging.info(
                f'cut_r:{len(wav_con)-int(new_end * Config.sample_rate)}')
            logging.info(
                f'mel_l:{(int(new_start * Config.sample_rate)+256)//Config.hop_size}')
            logging.info(
                f'mel_r:{(len(wav_con)-int(new_end * Config.sample_rate)+256)//Config.hop_size}')

            logging.info(f'wav_con: {wav_con.shape}')
            logging.info(f'render: {render.shape}')
        elif Config.model_type == "onnx":
            logging.info('Rendering audio.')
            f0 = f0_render.astype(np.float32)
            mel = mel_render.astype(np.float32)
            # 给mel和f0添加batched维度
            mel = np.expand_dims(mel, axis=0).transpose(0, 2, 1)
            f0 = np.expand_dims(f0, axis=0)
            input_data = {'mel': mel, 'f0': f0, }
            output = ort_session.run(['waveform'], input_data)[0]
            wav_con = output[0]

            render = wav_con[int(new_start * Config.sample_rate)                             :int(new_end * Config.sample_rate)]
            logging.info(f'cut_l:{int(new_start * Config.sample_rate)}')
            logging.info(
                f'cut_r:{len(wav_con)-int(new_end * Config.sample_rate)}')
            logging.info(
                f'mel_l:{(int(new_start * Config.sample_rate)+256)//Config.hop_size}')
            logging.info(
                f'mel_r:{(len(wav_con)-int(new_end * Config.sample_rate)+256)//Config.hop_size}')

            logging.info(f'wav_con: {wav_con.shape}')
            logging.info(f'render: {render.shape}')
        else:
            raise ValueError(f"Unsupported model type: {Config.model_type}")

        # 添加幅度调制
        A_flag = self.flags.get('A', 0)
        if A_flag != 0:
            logging.info(f'Applying Amplitude Modulation A={A_flag}')
            A_clamped = np.clip(A_flag, -100, 100)

            if len(pitch_render) > 1 and len(t) > 1:
                pitch_derivative = np.gradient(pitch_render, t)
                gain_at_mel_frames = 5**((10**-4) *
                                         A_clamped * pitch_derivative)
                num_samples = len(render)
                audio_time_vector = np.linspace(
                    new_start, new_end, num=num_samples, endpoint=False)

                interpolated_gain = np.interp(audio_time_vector,
                                              t,  # Time points for gain_at_mel_frames
                                              gain_at_mel_frames,
                                              # Value for time < t[0]
                                              left=gain_at_mel_frames[0],
                                              right=gain_at_mel_frames[-1])  # Value for time > t[-1]

                render = render * interpolated_gain
                logging.info('Amplitude modulation applied.')
            else:
                logging.warning(
                    "Not enough pitch points (>1) to calculate derivative for Amplitude Modulation.")

        render = render / scale
        new_max = np.max(np.abs(render))

        # normalize using loudness_norm
        if Config.wave_norm:
            if "P" in self.flags.keys():
                p_strength = self.flags['P']
                if p_strength is not None:
                    render = loudness_norm(
                        render, Config.sample_rate, peak=-1, loudness=-16.0, block_size=0.400, strength=p_strength)
                else:
                    render = loudness_norm(
                        render, Config.sample_rate, peak=-1, loudness=-16.0, block_size=0.400)

        if new_max > Config.peak_limit:
            render = render / new_max

        volume_scale = self.volume / 100.0
        render = render * volume_scale

        save_wav(self.out_file, render)


def split_arguments(input_string):
    # Regular expression to match two file paths at the beginning
    otherargs = input_string.split(' ')[-11:]
    file_path_strings = ' '.join(input_string.split(' ')[:-11])

    first_file, second_file = file_path_strings.split('.wav ')
    return [first_file+".wav", second_file] + otherargs


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if server_ready:
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server Ready')
            # logging.info("Responded 200 OK to readiness check.") # Optional: Verbose logging
        else:
            # 503 Service Unavailable is appropriate
            self.send_response(503)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server Initializing')
            logging.info(
                "Responded 503 Service Unavailable to readiness check (server not ready).")
        return

    def do_POST(self):
        if not server_ready:
            logging.warning(
                "Received POST request before server was fully ready. Sending 503.")
            self.send_response(503)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'Server initializing, please retry.')
            return
        
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        post_data_string = post_data.decode('utf-8')
        logging.info(f"post_data_string: {post_data_string}")
        try:
            sliced = split_arguments(post_data_string)
            in_file_path = Path(sliced[0])
            out_file_path = Path(sliced[1])
            note_info_for_log = f"'{in_file_path.stem}' -> '{out_file_path.name}'"
            logging.info(f"Processing {note_info_for_log} begins...")

            # === Execute Resampler within try...except ===
            Resampler(*sliced)
            # If Resampler completes without exception, it's considered successful *by the server*

            logging.info(f"Processing {note_info_for_log} successful.")
            self.send_response(200)  # Send 200 OK
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"Success: {note_info_for_log}".encode('utf-8'))

        except FileNotFoundError:
            error_msg = f"Error processing {note_info_for_log}: Input file not found."
            logging.error(error_msg, exc_info=True)  # Log full traceback
            self.send_response(404)  # Not Found
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(
                f"{error_msg}\n{traceback.format_exc()}".encode('utf-8'))

        except Exception:
            # Catch any other exception during Resampler execution
            error_msg = f"[Error processing {note_info_for_log}: An internal error occurred."
            # Log the full traceback for debugging
            logging.error(error_msg, exc_info=True)
            self.send_response(500)  # Internal Server Error
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            # Send error details back (optional, consider security if sensitive info might leak)
            self.wfile.write(
                f"{error_msg}\n{traceback.format_exc()}".encode('utf-8'))

        '''
        except Exception as e:
            trcbk = traceback.format_exc()
            self.send_response(500)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(f"An error occurred.\n{trcbk}".encode('utf-8'))
        self.send_response(200)
        self.end_headers()
        '''


class ThreadPoolHTTPServer(HTTPServer):
    def __init__(self, server_address, RequestHandlerClass, max_workers):
        super().__init__(server_address, RequestHandlerClass)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def process_request(self, request, client_address):
        self.executor.submit(self.process_request_thread,
                             request, client_address)

    def process_request_thread(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def run(server_class=ThreadPoolHTTPServer, handler_class=RequestHandler, port=8572, max_workers=1):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class,
                         max_workers=max_workers)
    logging.info(
        f'Listening on port {port} with {max_workers} worker threads...')
    global server_ready
    server_ready = True
    httpd.serve_forever()


if __name__ == '__main__':
    lock_file_path = Path(tempfile.gettempdir()) / 'server.lock'

    try:
        with FileLock(str(lock_file_path), timeout=0.5) as server_lock:
            logging.info(
                f"Successfully acquired server lock: {lock_file_path}")
            logging.info("This process will start the HifiSampler server.")

            global hnsep_model, melAnalysis, vocoder, ort_session

            if Config.wave_norm:
                try:
                    import pyloudnorm as pyln
                    logging.info("pyloudnorm imported for wave normalization.")
                except ImportError:
                    logging.warning(
                        "pyloudnorm not found, wave normalization disabled.")
                    Config.wave_norm = False  # Disable if import fails

            logging.info(f'hachisampler {version}')

            # Load HifiGAN
            vocoder_path = Path(Config.vocoder_path)
            onnx_default_path = Path(
                r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")
            ckpt_default_path = Path(
                r"pc_nsf_hifigan_44.1k_hop512_128bin_2025.02\model.ckpt")

            # Determine actual vocoder path based on existence and defaults
            actual_vocoder_path = None
            if vocoder_path.exists():
                actual_vocoder_path = vocoder_path
            elif ckpt_default_path.exists():
                actual_vocoder_path = ckpt_default_path
                logging.info(
                    f"Configured vocoder path not found, using default: {ckpt_default_path}")
            elif onnx_default_path.exists():
                actual_vocoder_path = onnx_default_path
                logging.info(
                    f"Configured vocoder path not found, using default: {onnx_default_path}")
            else:
                # Raise error only if no vocoder can be found at all
                raise FileNotFoundError(
                    f"No HifiGAN model found. Checked configured path '{Config.vocoder_path}' and defaults.")

            # Load the determined model
            if actual_vocoder_path.suffix == '.ckpt':
                from util.nsf_hifigan import NsfHifiGAN
                Config.model_type = 'ckpt'
                vocoder = NsfHifiGAN(model_path=actual_vocoder_path)
                vocoder.to_device(Config.device)
                logging.info(f'Loaded HifiGAN (ckpt): {actual_vocoder_path}')
                logging.info(f'Using device: {Config.device}')
            elif actual_vocoder_path.suffix == '.onnx':
                import onnxruntime
                Config.model_type = 'onnx'
                Config.max_workers = 1
                # Determine available providers, prioritize DML/CUDA over CPU
                available_providers = onnxruntime.get_available_providers()
                preferred_providers = []
                if 'DmlExecutionProvider' in available_providers:
                    preferred_providers.append('DmlExecutionProvider')
                elif 'CUDAExecutionProvider' in available_providers:
                    preferred_providers.append('CUDAExecutionProvider')
                preferred_providers.append('CPUExecutionProvider')  # Fallback

                ort_session = onnxruntime.InferenceSession(
                    str(actual_vocoder_path), providers=preferred_providers)
                logging.info(
                    f'Loaded HifiGAN (onnx): {actual_vocoder_path} using providers {ort_session.get_providers()}')
                logging.info(f'Using provider: {ort_session.get_providers()[0]}')
            else:
                Config.model_type = actual_vocoder_path.suffix
                raise ValueError(
                    f'Invalid model type: {Config.model_type} for path {actual_vocoder_path}')

            # Load HN-SEP model
            hnsep_model, hnsep_model_args = load_sep_model(
                Config.hnsep_model_path, Config.device)
            logging.info(f'Loaded HN-SEP: {Config.hnsep_model_path}')

            # Initialize Mel Spectrogram tool
            melAnalysis = PitchAdjustableMelSpectrogram(
                sample_rate=Config.sample_rate,
                n_fft=Config.n_fft,
                win_length=Config.win_size,
                hop_length=Config.origin_hop_size,
                f_min=Config.mel_fmin,
                f_max=Config.mel_fmax,
                n_mels=Config.n_mels
            )
            logging.info(
                f'Initialized Mel Analysis with hop_size={Config.origin_hop_size}.')

            logging.info("Starting the HTTP server...")
            run(max_workers=Config.max_workers)
            logging.info("Server has stopped.")

    except Timeout:
        # This block is executed if server_lock.acquire failed (lock already held)
        logging.info(
            f"Another instance of the server seems to be running (lock file '{lock_file_path}' is held). Exiting.")
        sys.exit(0)

    except Exception as e:
        # Catch any *other* exception during setup (model loading, etc.)
        logging.error(
            f"Failed to initialize or start the server: {e}", exc_info=True)
        sys.exit(1)  # Exit with error status

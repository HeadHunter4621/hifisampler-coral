import logging
from pathlib import Path
import numpy as np
import resampy
import torch
from config import CONFIG
from scipy import signal
import soundfile as sf

if CONFIG.wave_norm:
    try:
        import pyloudnorm as pyln
        logging.info("pyloudnorm imported for wave normalization.")
    except ImportError:
        logging.warning(
            "pyloudnorm not found, wave normalization disabled.")
        CONFIG.wave_norm = False  # Disable if import fails


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

    if CONFIG.trim_silence:
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
        voiced_frames = [i for i, rms in enumerate(
            rms_values) if rms > CONFIG.silence_threshold]

        if voiced_frames:
            first_voiced = voiced_frames[0]
            last_voiced = voiced_frames[-1]

            # 添加一些余量，避免突然截断
            padding_frames = int(rate * 0.1) // hop_length  # 添加100ms的余量

            # 确保不超出边界
            start_sample = max(0, first_voiced * hop_length)
            end_sample = min(len(audio), (last_voiced + 1 +
                             padding_frames) * hop_length + frame_length)

            trimmed_audio = audio[start_sample:end_sample]
            logging.info(
                f'Trimmed silence: {len(audio)} -> {len(trimmed_audio)} samples')

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
    if CONFIG.trim_silence:
        # 创建一个全零数组作为输出
        output = np.zeros(original_length)

        # 将标准化后的音频放回原位置，并添加平滑过渡
        if voiced_frames:  # 确保有声音帧存在
            start_sample = max(0, first_voiced * int(hop_length))

            # 计算需要放回的音频长度
            available_length = min(len(audio), original_length - start_sample)

            # 创建一个逐渐衰减的窗函数，用于音频的尾部淡出
            # 最多200ms或音频长度的1/4
            fade_length = min(int(rate * 0.2), available_length // 4)
            fade_out = np.ones(available_length)

            if fade_length > 0:
                # 在末尾应用淡出效果
                fade_out[-fade_length:] = np.linspace(1.0, 0.0, fade_length)

            # 应用淡出效果并放回原位置
            output[start_sample:start_sample +
                   available_length] = audio[:available_length] * fade_out
            # 如果原始音频有后续部分，应用交叉淡入淡出
            if start_sample + available_length < original_length:
                remain_length = original_length - \
                    (start_sample + available_length)
                crossfade_length = min(fade_length, remain_length)

                if crossfade_length > 0:
                    crossfade_start = start_sample + available_length
                    # 从原始音频获取剩余部分
                    remain_audio = original_audio[crossfade_start:original_length]

                    # 应用淡入效果到原始音频的剩余部分
                    fade_in = np.ones(remain_length)
                    fade_in[:crossfade_length] = np.linspace(
                        0.0, 1.0, crossfade_length)

                    # 填充剩余部分
                    output[crossfade_start:original_length] = remain_audio * fade_in
        else:  # 如果没有声音帧，直接返回原始音频
            output = audio[:original_length]

        audio = output

    # 如果原始音频短于block_size，裁剪回原始长度
    if original_length < int(rate * block_size):
        audio = audio[:original_length]

    return audio


def growl(audio: np.ndarray, sample_rate: int, frequency: float = 80.0, strength: float = 0.5, waveform: str = 'square') -> np.ndarray:
    """
    Apply high frequency vibrato effect to the audio signal using square wave modulation.

    Args:
        audio (np.ndarray): Input audio signal.
        sample_rate (int): Sample rate of the audio signal.
        frequency (float): Frequency of the vibrato effect in Hz.
        strength (float): Strength of the vibrato effect (0 to 1).

    Returns:
        np.ndarray: Audio signal with growl effect applied.
    """
    if strength == 0 and frequency <= 0:
        return audio

    # --- 1. Generate Base Pitch Modulation (Vibrato + Jitter) ---
    num_samples = len(audio)
    t = np.arange(num_samples) / float(sample_rate)
    # Generate LFO for pitch modulation based on the selected waveform
    pitch_lfo = signal.square(2 * np.pi * frequency * t)

    # Combine LFO and Jitter for total pitch modulation
    max_pitch_dev_cents = 50.0  # Max deviation for vibrato at full strength
    cents_deviation = (pitch_lfo * strength) * max_pitch_dev_cents
    pitch_ratio = 2**(cents_deviation / 1200.0)

    # --- 2. Apply Pitch Modulation via Resampling ---
    read_indices = np.cumsum(pitch_ratio) - pitch_ratio[0]
    original_indices = np.arange(num_samples)
    growl_audio = np.interp(read_indices, original_indices, audio)

    return growl_audio


def pre_emphasis_base_tension(wave, b):
    """
    Args:
        wave: [1, 1, t]
    """
    original_length = wave.size(-1)
    pad_length = (CONFIG.hop_size - (original_length %
                  CONFIG.hop_size)) % CONFIG.hop_size
    wave = torch.nn.functional.pad(
        wave, (0, pad_length), mode='constant', value=0)
    wave = wave.squeeze(1)

    spec = torch.stft(
        wave,
        CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=torch.hann_window(CONFIG.win_size).to(wave.device),
        return_complex=True
    )
    spec_amp = torch.abs(spec)
    spec_phase = torch.atan2(spec.imag, spec.real)

    spec_amp_db = torch.log(torch.clamp(spec_amp, min=1e-9))

    fft_bin = CONFIG.n_fft // 2 + 1
    x0 = fft_bin / ((CONFIG.sample_rate / 2) / 1500)
    freq_filter = (-b / x0) * torch.arange(0, fft_bin, device=wave.device) + b
    spec_amp_db = spec_amp_db + \
        torch.clamp(freq_filter, min=-2, max=2).unsqueeze(0).unsqueeze(2)

    spec_amp = torch.exp(spec_amp_db)

    filtered_wave = torch.istft(
        torch.complex(spec_amp * torch.cos(spec_phase),
                      spec_amp * torch.sin(spec_phase)),
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=torch.hann_window(CONFIG.win_size).to(wave.device)
    )

    original_max = torch.max(torch.abs(wave))
    filtered_max = torch.max(torch.abs(filtered_wave))
    filtered_wave = filtered_wave * \
        (original_max / filtered_max) * (np.clip(b/(-15), 0, 0.33) + 1)
    filtered_wave = filtered_wave.unsqueeze(1)
    filtered_wave = filtered_wave[:, :, :original_length]

    return filtered_wave


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
    if type(loc) is str:  # make sure input is Path
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

    if fs != CONFIG.sample_rate:
        x = resampy.resample(x, fs, CONFIG.sample_rate)

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
        sf.write(str(loc), x, CONFIG.sample_rate, 'PCM_16')
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")

import logging
from pathlib import Path
import numpy as np
import resampy
import torch
from config import CONFIG
import soundfile as sf

if CONFIG.wave_norm:
    try:
        import pyloudnorm as pyln

        logging.info("pyloudnorm imported for wave normalization.")
    except ImportError:
        logging.warning("pyloudnorm not found, wave normalization disabled.")
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
    audio: np.ndarray,
    rate: int,
    peak=-1.0,
    loudness=-23.0,
    block_size=0.400,
    strength=100,
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
    original_audio = (
        audio.copy()
    )  # Preserve the original audio for subsequent processing

    if CONFIG.trim_silence:

        def get_rms_db(audio_segment):
            if len(audio_segment) == 0:
                return -np.inf
            rms = np.sqrt(np.mean(np.square(audio_segment)))
            if rms < 1e-10:  # Avoid log(0) errors
                return -np.inf
            return 20 * np.log10(rms)

        frame_length = int(rate * 0.02)  # 20ms window
        hop_length = int(rate * 0.01)  # 10ms step size

        rms_values = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i : i + frame_length]
            rms_db = get_rms_db(frame)
            rms_values.append(rms_db)

        # Detect audio frames using threshold detection
        voiced_frames = [
            i for i, rms in enumerate(rms_values) if rms > CONFIG.silence_threshold
        ]

        if voiced_frames:
            first_voiced = voiced_frames[0]
            last_voiced = voiced_frames[-1]

            # Add some extra margin to avoid abrupt truncation
            padding_frames = int(rate * 0.1) // hop_length  # Add a 100ms margin

            # Ensure that boundaries are not exceeded
            start_sample = max(0, first_voiced * hop_length)
            end_sample = min(
                len(audio),
                (last_voiced + 1 + padding_frames) * hop_length + frame_length,
            )

            trimmed_audio = audio[start_sample:end_sample]
            logging.info(
                f"Trimmed silence: {len(audio)} -> {len(trimmed_audio)} samples"
            )

            # Perform loudness normalization on the clipped audio
            audio = trimmed_audio

    # If the audio length is shorter than the minimum block size, padding is applied
    if len(audio) < int(rate * block_size):
        padding_length = int(rate * block_size) - len(audio)
        audio = np.pad(audio, (0, padding_length), mode="reflect")

    # Measure the loudness first
    meter = pyln.Meter(rate, block_size=block_size)  # create BS.1770 meter
    _loudness = meter.integrated_loudness(audio)

    # Apply strength to calculate the target loudness
    final_loudness = _loudness + (loudness - _loudness) * strength / 100

    # Loudness normalize audio to [loudness] LUFS
    audio = pyln.normalize.loudness(audio, _loudness, final_loudness)

    # If silent capture is enabled, the original length must be restored
    if CONFIG.trim_silence:
        # Create an array filled entirely with zeros as the output
        output = np.zeros(original_length)

        # Return the normalized audio to its original position and add smooth transitions
        if voiced_frames:  # Ensure that an audio frame exists
            start_sample = max(0, first_voiced * int(hop_length))

            # Calculate the length of audio to be restored
            available_length = min(len(audio), original_length - start_sample)

            # Create a gradually decaying window function for audio fade-out
            # Maximum 200ms or 1/4 of audio length
            fade_length = min(int(rate * 0.2), available_length // 4)
            fade_out = np.ones(available_length)

            if fade_length > 0:
                # Apply a fade-out effect at the end
                fade_out[-fade_length:] = np.linspace(1.0, 0.0, fade_length)

            # Apply the fade-out effect and return it to its original position
            output[start_sample : start_sample + available_length] = (
                audio[:available_length] * fade_out
            )
            # If the original audio has a subsequent section, apply a crossfade
            if start_sample + available_length < original_length:
                remain_length = original_length - (start_sample + available_length)
                crossfade_length = min(fade_length, remain_length)

                if crossfade_length > 0:
                    crossfade_start = start_sample + available_length
                    # Extract the remaining portion from the original audio
                    remain_audio = original_audio[crossfade_start:original_length]

                    # Apply a fade-in effect to the remainder of the original audio
                    fade_in = np.ones(remain_length)
                    fade_in[:crossfade_length] = np.linspace(0.0, 1.0, crossfade_length)

                    # Fill in the remaining portion
                    output[crossfade_start:original_length] = remain_audio * fade_in
        else:  # If there is no audio frame, return the raw audio directly
            output = audio[:original_length]

        audio = output

    # If the original audio is shorter than block_size, trim it back to its original length
    if original_length < int(rate * block_size):
        audio = audio[:original_length]

    return audio


def pre_emphasis_base_tension(wave, b):
    """
    Args:
        wave: [1, 1, t]
    """
    original_length = wave.size(-1)
    pad_length = (
        CONFIG.hop_size - (original_length % CONFIG.hop_size)
    ) % CONFIG.hop_size
    wave = torch.nn.functional.pad(wave, (0, pad_length), mode="constant", value=0)
    wave = wave.squeeze(1)

    spec = torch.stft(
        wave,
        CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=torch.hann_window(CONFIG.win_size).to(wave.device),
        return_complex=True,
    )
    spec_amp = torch.abs(spec)
    spec_phase = torch.atan2(spec.imag, spec.real)

    spec_amp_db = torch.log(torch.clamp(spec_amp, min=1e-9))

    fft_bin = CONFIG.n_fft // 2 + 1
    x0 = fft_bin / ((CONFIG.sample_rate / 2) / 1500)
    freq_filter = (-b / x0) * torch.arange(0, fft_bin, device=wave.device) + b
    spec_amp_db = spec_amp_db + torch.clamp(freq_filter, min=-2, max=2).unsqueeze(
        0
    ).unsqueeze(2)

    spec_amp = torch.exp(spec_amp_db)

    filtered_wave = torch.istft(
        torch.complex(
            spec_amp * torch.cos(spec_phase), spec_amp * torch.sin(spec_phase)
        ),
        n_fft=CONFIG.n_fft,
        hop_length=CONFIG.hop_size,
        win_length=CONFIG.win_size,
        window=torch.hann_window(CONFIG.win_size).to(wave.device),
    )

    original_max = torch.max(torch.abs(wave))
    filtered_max = torch.max(torch.abs(filtered_wave))
    filtered_wave = (
        filtered_wave
        * (original_max / filtered_max)
        * (np.clip(b / (-15), 0, 0.33) + 1)
    )
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
            loc = loc.with_suffix("." + ext.lower())
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
        sf.write(str(loc), x, CONFIG.sample_rate, "PCM_16")
    except Exception as e:
        logging.error(f"Error saving WAV file: {e}")

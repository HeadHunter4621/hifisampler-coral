import dataclasses
from pathlib import Path
import torch
from util.load_config_from_yaml import load_config_from_yaml
from typing import Any


@dataclasses.dataclass
@load_config_from_yaml(script_path=Path(__file__))
class Config:
    sample_rate: int = 44100  # UTAU only really likes 44.1khz
    win_size: int = 2048  # Must be consistent with the vocoder training
    hop_size: int = 512  # Must be consistent with the vocoder training
    origin_hop_size: int = 128  # The hop size before interpolation can be appropriately reduced to improve the electronic sound of long notes
    n_mels: int = 128  # Must be consistent with the vocoder training
    n_fft: int = 2048  # Must be consistent with the vocoder training
    mel_fmin: float = 40  # Must be consistent with the vocoder training
    mel_fmax: float = 16000  # Must be consistent with the vocoder training
    fill: int = 6
    vocoder_path: str = r"\path\to\your\vocoder\pc_nsf_hifigan\model.ckpt"
    model_type: str = "ckpt"  # or 'onnx'
    hnsep_model_path: str = r"\path\to\your\hnsep\model.pt"
    wave_norm: bool = False
    trim_silence: bool = (
        True  # Should silent sections be trimmed before loudness normalization?
    )
    silence_threshold: float = -52.0
    loop_mode: bool = False
    peak_limit: float = 1.0
    # max_workers can be an int or 'auto' (resolved to physical cores at runtime)
    max_workers: Any = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = Config()

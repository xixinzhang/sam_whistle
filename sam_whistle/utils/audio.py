import torchaudio
import librosa
import numpy as np
import torchaudio.functional as F
import torch
import tyro

from sam_whistle.config import SpecConfig
from sam_whistle import utils

def load_wave_file(file_path, type='tensor'):
    """Load one wave file."""
    if type == 'numpy':
        waveform, sample_rate = librosa.load(file_path, sr=None)
    elif type == 'tensor':
        waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def wave_to_spectrogram(waveform, sample_rate=None, frame_ms=None, hop_ms=None, pad=0, n_fft=None, hop_length=None, top_db=80.0, **kwargs):
    """Convert waveform to spectrogram."""
    # fft params
    if n_fft is None:
        if frame_ms is not None and sample_rate is not None:
            n_fft = int(frame_ms * sample_rate / 1000)
        else:
            raise ValueError("n_fft or frame_ms must be provided.")
    if hop_length is None:
        if hop_ms is not None and sample_rate is not None:
            hop_length = int(hop_ms * sample_rate / 1000)
        else:
            raise ValueError("hop_length or hop_ms must be provided.")

    # power scale spectrogram
    spec = F.spectrogram(
        waveform,
        pad=pad,
        window=torch.hamming_window(n_fft),
        n_fft=n_fft, 
        hop_length=hop_length,
        win_length=n_fft,
        power=2.0,
        normalized=False,
        center=True,
        pad_mode='reflect',
        onesided=True,
    )
    # decibel scale spectrogram with cutoff
    spec_db = F.amplitude_to_DB(spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=top_db)
    spec_db = torch.flip(spec_db, [-2])
    # normalzed spectrogram
    spec_db = (spec_db - spec_db.min()) / (spec_db.max() - spec_db.min())  # normalize to [0, 1]

    return spec_db # (C, freq, time)


if __name__ == '__main__':
    sample_wave = 'data/dclde/audio/palmyra092007FS192-070924-210000.wav'
    sample_wave = 'data/dclde/audio/Qx-Dc-CC0411-TAT11-CH2-041114-154040-s.wav'
    waveform, sr = load_wave_file(sample_wave)
    args = tyro.cli(SpecConfig)
    spec_db = wave_to_spectrogram(waveform, sr, **vars(args))
    print(spec_db.shape, spec_db.max(), spec_db.min())
    utils.visualize_array(spec_db[:,:,142500: 144000],)

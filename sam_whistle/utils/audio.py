import torchaudio
import librosa
import numpy as np
import torchaudio.functional as F
import torch
import tyro
from scipy.signal import medfilt2d

from sam_whistle.config import SpectConfig
from sam_whistle import utils

def load_wave_file(file_path, type='tensor'):
    """Load one wave file."""
    if type == 'numpy':
        waveform, sample_rate = librosa.load(file_path, sr=None)
    elif type == 'tensor':
        waveform, sample_rate = torchaudio.load(file_path)
    return waveform, sample_rate

def wave_to_spect(waveform, sample_rate=None, frame_ms=None, hop_ms=None, pad=0, n_fft=None, hop_length=None, top_db=80.0, center = True, amin = 1e-10, **kwargs):
    """Convert waveform to raw spectrogram in power dB scale."""
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
        center=center,
        pad_mode='reflect',
        onesided=True,
    )
    # decibel scale spectrogram with cutoff specified by top_db
    spect_power_db = F.amplitude_to_DB(spec, multiplier=10.0, amin=amin, db_multiplier=0.0, top_db=top_db)
    return spect_power_db # (C, freq, time)

def normalize_spect(spect_db):
    spect_db = (spect_db - spect_db.min()) / (spect_db.max() - spect_db.min())  # normalize to [0, 1]
    return spect_db

def snr_spect(spect_db, click_thr_db, broadband_thr_n):
    meanf_db = np.mean(spect_db, axis=1, keepdims=True)
    click_p = np.sum((spect_db - meanf_db) > click_thr_db, axis=0) > broadband_thr_n
    use_p = ~click_p
    spect_db = medfilt2d(spect_db, kernel_size=[3,3])
    if np.sum(use_p) == 0:
        # Qx-Dc-SC03-TAT09-060516-173000.wav 4500 no use_p
        use_p = np.ones_like(click_p)
    meanf_db = np.mean(spect_db[:, use_p], axis=1, keepdims=True)
    snr_spect_db = spect_db - meanf_db
    return snr_spect_db


if __name__ == '__main__':
    sample_wave = 'data/dclde/audio/palmyra092007FS192-070924-210000.wav'
    sample_wave = 'data/dclde/audio/Qx-Dc-CC0411-TAT11-CH2-041114-154040-s.wav'
    sample_wave = 'data/dclde/audio/Qx-Tt-SCI0608-Ziph-060819-074737.wav'
    waveform, sr = load_wave_file(sample_wave)
    args = tyro.cli(SpectConfig)
    spec_db = wave_to_spect(waveform, sr, **vars(args))
    print(spec_db.shape, spec_db.max(), spec_db.min())
    spec_db = flip_and_normalize_spect(spec_db)
    utils.visualize_array(spec_db[:,:,:1500],)

from __future__ import annotations

import numpy as np
import librosa


AUDIO_SR = 8000
N_FFT = 256
HOP_LENGTH = 128
N_MELS = 32
N_MFCC = 20
MAX_MEL_FRAMES = 40


def pad_or_trim(matrix: np.ndarray, target_frames: int = MAX_MEL_FRAMES) -> np.ndarray:
    if matrix.shape[1] >= target_frames:
        return matrix[:, :target_frames].astype(np.float32)
    pad_width = target_frames - matrix.shape[1]
    return np.pad(matrix, ((0, 0), (0, pad_width)), mode="constant").astype(np.float32)


def audio_to_mfcc_frames(audio: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    return np.vstack([mfcc, delta, delta2]).T.astype(np.float32)


def stats_from_frames(frames: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [frames.mean(axis=0), frames.std(axis=0), frames.min(axis=0), frames.max(axis=0)]
    ).astype(np.float32)


def audio_to_logmel(audio: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
        fmax=sr // 2,
    )
    return pad_or_trim(np.log1p(mel))


def audio_to_logmel_flat(audio: np.ndarray, sr: int = AUDIO_SR) -> np.ndarray:
    return audio_to_logmel(audio, sr).reshape(-1).astype(np.float32)


def logmel_to_audio(logmel: np.ndarray, sr: int = AUDIO_SR, iterations: int = 16) -> np.ndarray:
    mel = np.expm1(np.maximum(logmel, 0.0)).astype(np.float32)
    audio = librosa.feature.inverse.mel_to_audio(
        mel,
        sr=sr,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=N_FFT,
        power=2.0,
        n_iter=iterations,
        fmax=sr // 2,
    )
    if audio.size == 0:
        return np.zeros(sr // 2, dtype=np.float32)
    audio = audio / (np.max(np.abs(audio)) + 1e-6)
    return audio.astype(np.float32)

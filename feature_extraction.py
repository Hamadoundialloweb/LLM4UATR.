import os
from typing import List, Tuple, Dict, Any
import numpy as np
import torch

try:
    import librosa
    _HAS_LIBROSA = True
except:
    import torchaudio
    _HAS_LIBROSA = False

from beats_wrapper import BEaTsCollate


# ============ Paramètres dataset ============
def _get_dataset_params(dataset_name: str) -> Dict[str, Any]:
    """Paramètres selon dataset"""
    if dataset_name in ("shipsear9_5s", "shipsear9_3s", "shipsear9_10s"):
        return {
            "sr": 52734,
            "fmin": 100.0,
            "fmax": 26367.0,
            "n_mels": 300,
            "n_barks": 300,
            "cqt_bins": 340
        }
    else:  # oceanship, deepship, etc.
        return {
            "sr": 32000,
            "fmin": 100.0,
            "fmax": 8000.0,
            "n_mels": 256,
            "n_barks": 300,
            "cqt_bins": 290
        }


TARGET_FRAMES_MAP = {
    "stft": 1200,
    "mel": 1200,
    "bark": 1200,
    "cqt": 900,
    "beats": 1200
}


# ============ Fonctions extraction ============
def _load_audio(path: str, sr: int) -> np.ndarray:
    """Charge audio"""
    if _HAS_LIBROSA:
        y, _ = librosa.load(path, sr=sr, mono=True)
        return y.astype(np.float32)
    else:
        waveform, file_sr = torchaudio.load(path)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if file_sr != sr:
            resampler = torchaudio.transforms.Resample(file_sr, sr)
            waveform = resampler(waveform)
        return waveform.squeeze(0).numpy().astype(np.float32)


def _mel_spec(y: np.ndarray, sr: int, n_fft: int, hop: int, win: int, 
              n_mels: int, fmin: float, fmax: float) -> np.ndarray:
    """Mel spectrogram"""
    if _HAS_LIBROSA:
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop, win_length=win,
            n_mels=n_mels, fmin=fmin, fmax=fmax, power=2.0, window="hann"
        )
        return np.log1p(mel)
    else:
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        spec = torch.stft(y_t, n_fft=n_fft, hop_length=hop, win_length=win,
                          window=torch.hann_window(win), return_complex=True)
        power = (spec.abs() ** 2).squeeze(0)
        
        fb = torchaudio.functional.create_fb_matrix(
            n_freqs=(n_fft // 2) + 1,
            f_min=fmin, f_max=fmax,
            n_mels=n_mels,
            sample_rate=sr
        )
        mel = torch.matmul(fb, power).numpy()
        return np.log1p(mel)


def _pad_or_trim(spec: np.ndarray, target: int) -> np.ndarray:
    """Pad ou trim à target frames"""
    F, T = spec.shape
    if T == target:
        return spec
    if T < target:
        return np.pad(spec, ((0, 0), (0, target - T)), mode="constant")
    start = (T - target) // 2
    return spec[:, start:start + target]


def _extract_mel(path: str, params: Dict, target_frames: int) -> np.ndarray:
    """Extraction MEL complète"""
    sr = params['sr']
    win = int(0.05 * sr)
    hop = int(0.025 * sr)
    n_fft = win
    
    y = _load_audio(path, sr)
    spec = _mel_spec(y, sr, n_fft, hop, win, 
                     params['n_mels'], params['fmin'], params['fmax'])
    spec = _pad_or_trim(spec, target_frames)
    
    # Normalisation
    mean = spec.mean()
    std = spec.std() + 1e-6
    spec = (spec - mean) / std
    
    return spec.astype(np.float32)


# ============ Collate Classes ============
class CollateSpectral:
    """Collate pour features spectrales"""
    
    def __init__(self, feature_type: str, dataset_name: str):
        self.feature_type = feature_type.strip().lower()
        self.dataset_name = dataset_name
        self.params = _get_dataset_params(dataset_name)
        self.target_frames = TARGET_FRAMES_MAP[self.feature_type]
        
        print(f"📊 CollateSpectral: {feature_type} | Target frames: {self.target_frames}")
    
    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = []
        labels = []
        
        for path, label in batch:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Audio introuvable: {path}")
            
            # Extraction MEL (extensible pour STFT/Bark/CQT)
            spec = _extract_mel(path, self.params, self.target_frames)
            feats.append(spec)
            labels.append(int(label))
        
        # Aligner hauteur
        min_H = min(f.shape[0] for f in feats)
        feats_aligned = []
        for f in feats:
            if f.shape[0] > min_H:
                feats_aligned.append(f[:min_H, :])
            else:
                feats_aligned.append(f)
        
        x = np.stack(feats_aligned, axis=0)[:, np.newaxis, :, :]  # [B, 1, H, W]
        X = torch.from_numpy(x)
        y = torch.tensor(labels, dtype=torch.long)
        
        return X, y


# ============ Fonction principale ============
def get_collate_fn(feature_type: str, dataset_name: str, use_beats: bool = False):
    """
    Retourne collate function selon feature type
    
    Args:
        feature_type: stft|mel|bark|cqt|beats
        dataset_name: Nom du dataset
        use_beats: Si True, utilise BEATs
        
    Returns:
        Collate function appropriée
    """
    if feature_type == "beats" or use_beats:
        beats_path = "/remote-home/Diallo/Beat/BEATs_iter3_plus_AS2M.pt"
        
        if not os.path.exists(beats_path):
            raise FileNotFoundError(
                f"❌ Checkpoint BEATs introuvable: {beats_path}\n"
                f"Téléchargez depuis: https://github.com/microsoft/unilm/tree/master/beats"
            )
        
        return BEaTsCollate(
            checkpoint_path=beats_path,
            target_sr=16000,
            extract_layer=None  # Utilise toutes les couches
        )
    else:
        return CollateSpectral(feature_type, dataset_name)

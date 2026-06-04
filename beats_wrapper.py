"""
BEATs Wrapper - Extraction simplifiée pour classification
"""

import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np
from typing import Optional, List, Tuple

from beats import BEATs, BEATsConfig


class BEaTsFeatureExtractor(nn.Module):
    """Extracteur de features BEATs simplifié"""
    
    def __init__(
        self, 
        checkpoint_path: str = "/remote-home/Diallo/Beat/BEATs_iter3_plus_AS2M.pt",
        target_sr: int = 16000,
        freeze: bool = True,
        extract_layer: Optional[int] = None,
    ):
        super().__init__()
        
        self.target_sr = target_sr
        self.freeze = freeze
        self.extract_layer = extract_layer
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"❌ Checkpoint BEATs introuvable: {checkpoint_path}\n"
                f"Téléchargez depuis: https://github.com/microsoft/unilm/tree/master/beats"
            )
        
        print(f"📦 Chargement BEATs depuis: {checkpoint_path}")
        
        # Charger checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        cfg = BEATsConfig(checkpoint['cfg'])
        
        # Créer modèle
        self.model = BEATs(cfg)
        self.model.load_state_dict(checkpoint['model'])
        
        self.hidden_dim = cfg.encoder_embed_dim
        
        print(f"✅ BEATs chargé!")
        print(f"   Layers: {cfg.encoder_layers} | Dim: {self.hidden_dim}")
        
        if self.freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
            print(f"   🔒 Paramètres gelés (freeze=True)")
    
    def _load_and_resample(self, audio_path: str) -> torch.Tensor:
        """Charge et rééchantillonne à 16kHz"""
        waveform, sr = torchaudio.load(audio_path)
        
        # Mono
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample si nécessaire
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)
    
    def forward(
        self, 
        audio_path: str = None, 
        waveform: torch.Tensor = None,
        padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extrait features BEATs
        
        Args:
            audio_path: Chemin vers fichier audio
            waveform: Waveform direct [samples] ou [batch, samples]
            padding_mask: Masque de padding
            
        Returns:
            features: [batch, seq_len, hidden_dim]
        """
        if waveform is None:
            if audio_path is None:
                raise ValueError("Fournir audio_path ou waveform")
            waveform = self._load_and_resample(audio_path)
        
        device = next(self.model.parameters()).device
        waveform = waveform.to(device)
        
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Extraction
        if self.freeze:
            with torch.no_grad():
                features, _, _ = self.model.extract_features(
                    source=waveform,
                    padding_mask=padding_mask,
                    max_layer=self.extract_layer
                )
        else:
            features, _, _ = self.model.extract_features(
                source=waveform,
                padding_mask=padding_mask,
                max_layer=self.extract_layer
            )
        
        return features


class BEaTsCollate:
    """Collate function pour batching avec BEATs"""
    
    def __init__(
        self, 
        checkpoint_path: str = "/remote-home/Diallo/Beat/BEATs_iter3_plus_AS2M.pt",
        target_sr: int = 16000,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        extract_layer: Optional[int] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.target_sr = target_sr
        self.device_str = device
        self.extract_layer = extract_layer
        self.extractor = None
        
        print(f"🔧 BEaTsCollate configuré")
    
    def _init_extractor(self):
        """Initialize extractor (appelé au premier batch)"""
        if self.extractor is None:
            device = torch.device(self.device_str if torch.cuda.is_available() else "cpu")
            self.extractor = BEaTsFeatureExtractor(
                checkpoint_path=self.checkpoint_path,
                target_sr=self.target_sr,
                freeze=True,
                extract_layer=self.extract_layer
            ).to(device)
            print(f"   ✅ Extractor initialisé sur {device}")
    
    def __call__(self, batch: List[Tuple[str, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            batch: [(audio_path, label), ...]
            
        Returns:
            features: [batch, seq_len, hidden_dim] sur CPU
            labels: [batch] sur CPU
        """
        # Initialize extractor au premier batch
        if self.extractor is None:
            self._init_extractor()
        
        features_list = []
        labels = []
        
        for audio_path, label in batch:
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio introuvable: {audio_path}")
            
            with torch.no_grad():
                feat = self.extractor(audio_path=audio_path)
                # ⚠️ CRITIQUE: Retourner sur CPU pour pin_memory du DataLoader
                features_list.append(feat.squeeze(0).cpu())
            
            labels.append(label)
        
        # Padding à la longueur max
        max_len = max(f.size(0) for f in features_list)
        hidden_dim = features_list[0].size(1)
        
        padded = []
        for feat in features_list:
            if feat.size(0) < max_len:
                # Padding sur CPU
                pad = torch.zeros(max_len - feat.size(0), hidden_dim)
                feat = torch.cat([feat, pad], dim=0)
            padded.append(feat)
        
        # Stack sur CPU
        features = torch.stack(padded, dim=0)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return features, labels


def test_beats():
    """Test rapide BEATs"""
    print("\n" + "="*60)
    print("TEST BEATs")
    print("="*60)
    
    # Waveform test (5 secondes)
    test_wave = torch.randn(1, 16000 * 5)
    
    extractor = BEaTsFeatureExtractor()
    
    with torch.no_grad():
        feats = extractor(waveform=test_wave)
    
    print(f"\n✅ Test OK!")
    print(f"   Input: {test_wave.shape}")
    print(f"   Output: {feats.shape}")
    print(f"   Expected: [1, ~500, 768]")
    
    return feats


if __name__ == "__main__":
    test_beats()

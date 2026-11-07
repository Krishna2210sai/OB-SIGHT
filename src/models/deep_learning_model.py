"""PyTorch models for CTG + maternal clinical data.

The architecture is intentionally compact: the dataset contains 552 recordings
and only 46 cesarean deliveries, so a large network would overfit immediately.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

SIGNAL_CHANNELS = 4
WINDOW_SAMPLES = 1800
CLINICAL_FEATURES = 15


class SignalEncoder(nn.Module):
    def __init__(self, in_channels: int = SIGNAL_CHANNELS, embedding_dim: int = 96):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 24, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(24, 48, kernel_size=5, padding=2),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(48, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 96, kernel_size=3, padding=1),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.projection = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(192, embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.cat([self.pool(x).flatten(1), self.max_pool(x).flatten(1)], dim=1)
        return self.projection(x)


class ClinicalEncoder(nn.Module):
    def __init__(self, in_features: int = CLINICAL_FEATURES, embedding_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 48),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(48, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultimodalCTGNet(nn.Module):
    """Signal encoder + clinical MLP + late fusion."""

    def __init__(self, clinical_features: int = CLINICAL_FEATURES):
        super().__init__()
        self.signal_encoder = SignalEncoder()
        self.clinical_encoder = ClinicalEncoder(clinical_features)
        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(96 + 32, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, signal: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.signal_encoder(signal), self.clinical_encoder(clinical)], dim=1)).squeeze(-1)


class SignalOnlyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.signal_encoder = SignalEncoder()
        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Linear(48, 1),
        )

    def forward(self, signal: torch.Tensor, clinical: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(self.signal_encoder(signal)).squeeze(-1)


class CNNBiLSTMNet(nn.Module):
    """Compact CTG-only CNN followed by a bidirectional LSTM.

    The convolutional stack reduces the 4 x 1800 input to a 48-channel
    sequence of 225 steps.  A single-layer BiLSTM then models temporal
    dependencies in both directions.  The architecture is deliberately small:
    the training split contains only 32 cesarean recordings.
    """

    def __init__(self, in_channels: int = SIGNAL_CHANNELS, lstm_hidden: int = 32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 48, kernel_size=3, padding=1),
            nn.BatchNorm1d(48),
            nn.ReLU(inplace=True),
        )
        self.lstm = nn.LSTM(
            input_size=48,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(lstm_hidden * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(32, 1),
        )

    def forward(self, signal: torch.Tensor, clinical: torch.Tensor | None = None) -> torch.Tensor:
        # (batch, channels, time) -> (batch, time, channels) for the LSTM.
        x = self.cnn(signal).transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        # h_n is (2, batch, hidden); concatenate forward and backward states.
        features = torch.cat([h_n[0], h_n[1]], dim=1)
        return self.head(features).squeeze(-1)


class ClinicalOnlyNet(nn.Module):
    def __init__(self, clinical_features: int = CLINICAL_FEATURES) -> None:
        super().__init__()
        self.clinical_encoder = ClinicalEncoder(clinical_features)
        self.head = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, signal: torch.Tensor | None, clinical: torch.Tensor) -> torch.Tensor:
        return self.head(self.clinical_encoder(clinical)).squeeze(-1)


def build_model(variant: str, clinical_features: int = CLINICAL_FEATURES) -> nn.Module:
    if variant == "multimodal":
        return MultimodalCTGNet(clinical_features)
    if variant == "signal":
        return SignalOnlyNet()
    if variant == "bilstm":
        return CNNBiLSTMNet()
    if variant == "clinical":
        return ClinicalOnlyNet(clinical_features)
    raise ValueError(f"Unknown model variant: {variant}")


def forward_model(model: nn.Module, signal: torch.Tensor, clinical: torch.Tensor) -> torch.Tensor:
    """Uniform forward call for all four variants."""
    if isinstance(model, (SignalOnlyNet, ClinicalOnlyNet)):
        return model(signal, clinical)
    return model(signal, clinical)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

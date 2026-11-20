"""Anomaly detectors with one interface: fit on benign rows, score any rows.

Both models are trained on benign traffic only and return an anomaly score where
higher means more unusual.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol, Self

import joblib
import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from torch import nn

from detection.dataset import FloatArray

_STANDARDIZED_CLIP = 10.0


class Detector(Protocol):
    """What the training loop and the agent need from a model."""

    name: str

    def params(self) -> dict[str, Any]:
        """Hyperparameters, for experiment tracking."""
        ...

    def fit(self, features: FloatArray) -> Self:
        """Fit on benign rows only."""
        ...

    def score(self, features: FloatArray) -> FloatArray:
        """Anomaly score per row; higher is more anomalous."""
        ...

    def save(self, directory: Path) -> None:
        """Persist the fitted model into ``directory``."""
        ...


class IsolationForestDetector:
    """Classical baseline: isolation depth in random trees."""

    name = "isolation_forest"

    def __init__(self, n_estimators: int = 200, max_samples: int = 4096, seed: int = 42) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.seed = seed
        self._model: IsolationForest | None = None

    def params(self) -> dict[str, Any]:
        return {
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "seed": self.seed,
        }

    def fit(self, features: FloatArray) -> Self:
        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=min(self.max_samples, len(features)),
            random_state=self.seed,
            n_jobs=-1,
        ).fit(features)
        return self

    def score(self, features: FloatArray) -> FloatArray:
        if self._model is None:
            raise RuntimeError("IsolationForestDetector is not fitted")
        scores: FloatArray = (-self._model.score_samples(features)).astype(np.float32)
        return scores

    def save(self, directory: Path) -> None:
        if self._model is None:
            raise RuntimeError("IsolationForestDetector is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "params.json").write_text(json.dumps(self.params()), encoding="utf-8")
        joblib.dump(self._model, directory / "model.joblib")

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Load a saved model. joblib is pickle: only load artifacts you produced."""
        detector = cls(**json.loads((directory / "params.json").read_text(encoding="utf-8")))
        detector._model = joblib.load(directory / "model.joblib")
        return detector


class _Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: tuple[int, ...], latent: int) -> None:
        super().__init__()
        sizes = [n_features, *hidden, latent]
        encoder: list[nn.Module] = []
        for left, right in pairwise(sizes):
            encoder += [nn.Linear(left, right), nn.ReLU()]
        decoder: list[nn.Module] = []
        for left, right in pairwise(reversed(sizes)):
            decoder += [nn.Linear(left, right), nn.ReLU()]
        self.encoder = nn.Sequential(*encoder[:-1])
        self.decoder = nn.Sequential(*decoder[:-1])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output: torch.Tensor = self.decoder(self.encoder(inputs))
        return output


class AutoencoderDetector:
    """Deep model: a bottleneck autoencoder scored by reconstruction error.

    It learns to compress and rebuild benign flows; flows it cannot rebuild well
    are unlike anything in the benign baseline.
    """

    name = "autoencoder"

    def __init__(
        self,
        hidden: tuple[int, ...] = (64, 32),
        latent: int = 12,
        epochs: int = 30,
        batch_size: int = 1024,
        learning_rate: float = 1e-3,
        patience: int = 4,
        holdout_fraction: float = 0.1,
        seed: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden = tuple(hidden)
        self.latent = latent
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.patience = patience
        self.holdout_fraction = holdout_fraction
        self.seed = seed
        self.device = device
        self.history_: list[dict[str, float]] = []
        self._network: _Autoencoder | None = None
        self._mean: torch.Tensor | None = None
        self._scale: torch.Tensor | None = None

    def params(self) -> dict[str, Any]:
        return {
            "hidden": list(self.hidden),
            "latent": self.latent,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "patience": self.patience,
            "holdout_fraction": self.holdout_fraction,
            "seed": self.seed,
            "device": self.device,
        }

    def _standardize(self, features: FloatArray) -> torch.Tensor:
        if self._mean is None or self._scale is None:
            raise RuntimeError("AutoencoderDetector is not fitted")
        tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        standardized = (tensor - self._mean) / self._scale
        # Benign-fitted scaling can send attack values to enormous z-scores.
        return standardized.clamp(-_STANDARDIZED_CLIP, _STANDARDIZED_CLIP)

    def fit(self, features: FloatArray) -> Self:
        torch.manual_seed(self.seed)
        generator = torch.Generator().manual_seed(self.seed)
        tensor = torch.from_numpy(np.ascontiguousarray(features, dtype=np.float32))
        self._mean = tensor.mean(dim=0)
        scale = tensor.std(dim=0)
        self._scale = torch.where(scale > 1e-6, scale, torch.ones_like(scale))
        data = self._standardize(features)

        order = torch.randperm(len(data), generator=generator)
        n_holdout = max(1, int(len(data) * self.holdout_fraction))
        holdout = data[order[:n_holdout]].to(self.device)
        train = data[order[n_holdout:]].to(self.device)

        network = _Autoencoder(data.shape[1], self.hidden, self.latent).to(self.device)
        optimizer = torch.optim.Adam(network.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()
        best_loss, best_state, stale = float("inf"), network.state_dict(), 0
        self.history_ = []
        for epoch in range(self.epochs):
            network.train()
            shuffled = train[torch.randperm(len(train), generator=generator).to(self.device)]
            total = 0.0
            for start in range(0, len(shuffled), self.batch_size):
                batch = shuffled[start : start + self.batch_size]
                optimizer.zero_grad()
                loss = loss_fn(network(batch), batch)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(batch)
            network.eval()
            with torch.no_grad():
                holdout_loss = float(loss_fn(network(holdout), holdout).item())
            self.history_.append({"epoch": epoch, "train_loss": total / len(train), "holdout_loss": holdout_loss})
            if holdout_loss < best_loss - 1e-5:
                best_loss, stale = holdout_loss, 0
                best_state = {k: v.detach().clone() for k, v in network.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        network.load_state_dict(best_state)
        self._network = network.eval()
        return self

    def score(self, features: FloatArray) -> FloatArray:
        if self._network is None:
            raise RuntimeError("AutoencoderDetector is not fitted")
        data = self._standardize(features)
        errors: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, len(data), 65_536):
                batch = data[start : start + 65_536].to(self.device)
                errors.append(((self._network(batch) - batch) ** 2).mean(dim=1).cpu())
        return torch.cat(errors).numpy().astype(np.float32)

    def save(self, directory: Path) -> None:
        if self._network is None or self._mean is None or self._scale is None:
            raise RuntimeError("AutoencoderDetector is not fitted")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "params.json").write_text(json.dumps(self.params()), encoding="utf-8")
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in self._network.state_dict().items()},
                "mean": self._mean,
                "scale": self._scale,
            },
            directory / "model.pt",
        )

    @classmethod
    def load(cls, directory: Path) -> Self:
        """Load a saved model. Uses ``weights_only=True``: tensors only, no code."""
        params = json.loads((directory / "params.json").read_text(encoding="utf-8"))
        params["hidden"] = tuple(params["hidden"])
        detector = cls(**params)
        payload = torch.load(directory / "model.pt", weights_only=True, map_location="cpu")
        detector._mean, detector._scale = payload["mean"], payload["scale"]
        network = _Autoencoder(len(payload["mean"]), detector.hidden, detector.latent)
        network.load_state_dict(payload["state_dict"])
        detector._network = network.to(detector.device).eval()
        return detector


DETECTOR_TYPES: dict[str, type[IsolationForestDetector] | type[AutoencoderDetector]] = {
    IsolationForestDetector.name: IsolationForestDetector,
    AutoencoderDetector.name: AutoencoderDetector,
}

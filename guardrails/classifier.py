"""Fine-tuned DistilBERT prompt-injection classifier (``sentinel-guard``).

``sentinel-guard train`` fine-tunes ``distilbert-base-uncased`` (pinned) on the
corpus from :mod:`guardrails.corpus` and saves it; ``sentinel-guard evaluate``
compares it with the regex detector on held-out text.

At run time :class:`InjectionClassifier` returns, per text, the probability that
it contains instructions aimed at an AI. Long text is split into overlapping
windows and the highest window counts, so a payload at the end of a long
command line is not truncated away. The threshold is a fixed 0.5: the
validation split shares templates with training, so tuning on it would only
fit those templates.

Saved models are safetensors plus tokenizer files and a ``metadata.json`` with
the SHA-256 of every file; :meth:`InjectionClassifier.load` refuses files that
changed and never unpickles anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from common.settings import get_settings
from guardrails.corpus import TEST_ONLY_FAMILIES, build_corpus, harness_payloads, load_deepset
from guardrails.injection import pattern_matches
from guardrails.sanitize import sanitize_text

logger = logging.getLogger("guardrails.classifier")

BASE_MODEL = "distilbert/distilbert-base-uncased"
BASE_REVISION = "12040accade4e8a0f71eabdb258fecc2e7e948be"
METADATA_FILE = "metadata.json"
THRESHOLD = 0.5
EVALUATION_FILE = "evaluation.md"
SAMPLE_PAYLOAD_MARK = "SYSTEM NOTICE TO AI ANALYST"


@dataclass(frozen=True)
class TrainConfig:
    epochs: int = 3
    batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    max_length: int = 256
    validation_fraction: float = 0.1
    seed: int = 42


def pick_device(name: str | None = None) -> torch.device:
    """``name`` if given, else Apple's GPU when present, else CPU."""
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class InjectionClassifier:
    """Scores text for prompt injection with a fine-tuned sequence classifier."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        threshold: float = THRESHOLD,
        max_length: int = 256,
        stride: int | None = None,
        max_windows: int = 32,
        batch_size: int = 32,
        device: torch.device | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.device = device or torch.device("cpu")
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.max_length = max_length
        # Tokens shared by consecutive windows; must leave room for the two special tokens.
        self.stride = max_length // 4 if stride is None else stride
        if not 0 <= self.stride < max_length - 2:
            raise ValueError(f"stride {self.stride} must be below max_length - 2 ({max_length - 2})")
        self.max_windows = max_windows
        self.batch_size = batch_size
        self.metadata = metadata or {}

    @torch.no_grad()
    def score(self, texts: Sequence[str]) -> list[float]:
        """Injection probability per text: the highest over its first ``max_windows`` windows."""
        if not texts:
            return []
        encoded = self.tokenizer(
            [sanitize_text(text, 20_000) for text in texts],
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding="longest",
            return_tensors="pt",
        )
        owners = encoded.pop("overflow_to_sample_mapping").tolist()
        windows: dict[int, int] = {}
        keep: list[int] = []
        for index, owner in enumerate(owners):
            windows[owner] = windows.get(owner, 0) + 1
            if windows[owner] <= self.max_windows:
                keep.append(index)
        best = [0.0] * len(texts)
        for start in range(0, len(keep), self.batch_size):
            rows = keep[start : start + self.batch_size]
            batch = {key: value[rows].to(self.device) for key, value in encoded.items()}
            probabilities = torch.softmax(self.model(**batch).logits.float(), dim=-1)[:, 1].cpu().tolist()
            for row, probability in zip(rows, probabilities, strict=True):
                best[owners[row]] = max(best[owners[row]], float(probability))
        return best

    def save(self, directory: Path, metadata: dict[str, Any]) -> None:
        """Write safetensors weights, tokenizer and checksummed metadata."""
        if directory.exists():
            shutil.rmtree(directory)
        directory.mkdir(parents=True)
        self.model.to("cpu").save_pretrained(directory, safe_serialization=True)
        self.model.to(self.device)
        self.tokenizer.save_pretrained(directory)
        files = {path.name: _sha256(path) for path in sorted(directory.iterdir())}
        record = {**metadata, "threshold": self.threshold, "max_length": self.max_length, "stride": self.stride,
                  "files_sha256": files}  # fmt: skip
        (directory / METADATA_FILE).write_text(json.dumps(record, indent=2), encoding="utf-8")
        self.metadata = record

    @classmethod
    def load(cls, directory: Path, *, device: torch.device | None = None) -> InjectionClassifier:
        """Load a saved classifier, refusing any file whose checksum changed or that is not recorded."""
        metadata = json.loads((directory / METADATA_FILE).read_text(encoding="utf-8"))
        recorded = metadata["files_sha256"]
        present = {path.name for path in directory.iterdir()} - {METADATA_FILE}
        if unexpected := present - set(recorded):
            raise ValueError(f"unrecorded files in the model directory: {sorted(unexpected)}")
        for name, expected in recorded.items():
            if _sha256(directory / name) != expected:
                raise ValueError(f"model file {name} does not match its recorded checksum")
        model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, use_safetensors=True
        )
        tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
        return cls(
            model,
            tokenizer,
            threshold=float(metadata["threshold"]),
            max_length=int(metadata["max_length"]),
            stride=int(metadata["stride"]),
            device=device,
            metadata=metadata,
        )

    def summary(self) -> dict[str, Any]:
        """What /health reports about this classifier."""
        return {"model": self.metadata.get("base_model", "unknown"), "threshold": self.threshold}


def load_base(revision: str = BASE_REVISION) -> tuple[Any, Any]:
    """The pinned pre-trained DistilBERT with a fresh two-class head."""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, revision=revision, num_labels=2, use_safetensors=True
    )
    return model, tokenizer


def train_classifier(
    train: pd.DataFrame, model: Any, tokenizer: Any, config: TrainConfig, *, device: torch.device
) -> tuple[InjectionClassifier, dict[str, Any]]:
    """Fine-tune ``model`` on ``train`` (``text``/``label``) and report validation metrics."""
    torch.manual_seed(config.seed)
    fit, validation = train_test_split(
        train, test_size=config.validation_fraction, stratify=train["label"], random_state=config.seed
    )
    texts, labels = fit["text"].tolist(), torch.tensor(fit["label"].to_numpy(), dtype=torch.long)
    steps = config.epochs * ((len(texts) + config.batch_size - 1) // config.batch_size)
    warmup = max(1, int(steps * config.warmup_fraction))
    model.to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0.0, (steps - step) / max(1, steps - warmup)))
    )
    generator = torch.Generator().manual_seed(config.seed)
    history: list[dict[str, float]] = []
    best_f1, best_epoch, best_state = -1.0, 0, None
    started = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        order = torch.randperm(len(texts), generator=generator).tolist()
        total = 0.0
        for start in range(0, len(order), config.batch_size):
            rows = order[start : start + config.batch_size]
            batch = tokenizer([texts[i] for i in rows], truncation=True, max_length=config.max_length,
                              padding="longest", return_tensors="pt").to(device)  # fmt: skip
            loss = model(**batch, labels=labels[rows].to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            optimizer.zero_grad()
            total += loss.item() * len(rows)
        classifier = InjectionClassifier(model, tokenizer, max_length=config.max_length, device=device)
        scores = np.asarray(classifier.score(validation["text"].tolist()))
        metrics = binary_metrics(validation["label"].to_numpy() == 1, scores >= THRESHOLD)
        history.append({"epoch": epoch + 1, "train_loss": total / len(texts), **metrics})
        logger.info("epoch %d: loss %.4f, validation %s", epoch + 1, total / len(texts), metrics)
        f1 = 0.0 if np.isnan(metrics["f1"]) else metrics["f1"]  # undefined when nothing was flagged
        if f1 > best_f1:  # keep the best epoch, not simply the last
            best_f1, best_epoch = f1, epoch + 1
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    report = {"config": asdict(config), "history": history, "best_epoch": best_epoch,
              "train_seconds": round(time.perf_counter() - started, 1), "train_rows": len(fit),
              "validation_rows": len(validation)}  # fmt: skip
    return InjectionClassifier(model, tokenizer, max_length=config.max_length, device=device), report


def binary_metrics(truth: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1 and false-positive rate; ``nan`` where undefined."""
    true_positive = int(np.sum(truth & predicted))
    false_positive = int(np.sum(~truth & predicted))
    positives, negatives = int(truth.sum()), int((~truth).sum())
    flagged = true_positive + false_positive
    precision = true_positive / flagged if flagged else float("nan")
    recall = true_positive / positives if positives else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if flagged and positives and (precision + recall) else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4) if positives and flagged else float("nan"),
        "false_positive_rate": round(false_positive / negatives, 4) if negatives else float("nan"),
        "n": len(truth),
    }


def compare(frame: pd.DataFrame, classifier: InjectionClassifier) -> pd.DataFrame:
    """Regex, classifier, and both combined (either flags), on ``frame`` (``text``/``label``)."""
    texts = frame["text"].tolist()
    regex = np.array([bool(pattern_matches(sanitize_text(text, 20_000))) for text in texts])
    model = np.asarray(classifier.score(texts)) >= classifier.threshold
    truth = frame["label"].to_numpy() == 1
    rows = {"regex": regex, "distilbert": model, "regex or distilbert": regex | model}
    return pd.DataFrame({name: binary_metrics(truth, predicted) for name, predicted in rows.items()}).T


def evaluation_slices(corpus: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The held-out test sets the README reports."""
    test = corpus[corpus["split"] == "test"]
    generated = test[test["source"] == "generated"]
    benign = generated[generated["label"] == 0]
    seen = generated[(generated["label"] == 1) & ~generated["family"].isin(TEST_ONLY_FAMILIES)]
    unseen = generated[(generated["label"] == 1) & generated["family"].isin(TEST_ONLY_FAMILIES)]
    harness = pd.DataFrame({"text": list(harness_payloads()), "label": 1})
    return {
        "log text, seen families (held-out templates)": pd.concat([benign, seen]),
        "log text, unseen families": pd.concat([benign, unseen]),
        "hard negatives only": benign[benign["family"] == "hard_negative"],
        "chat prompts (deepset test split)": test[test["source"] == "deepset"],
        "harness and sample payloads (never trained on)": harness,
    }


def family_recall(corpus: pd.DataFrame, classifier: InjectionClassifier) -> pd.DataFrame:
    """Recall per injection family on the test split."""
    injected = corpus[(corpus["split"] == "test") & (corpus["label"] == 1) & (corpus["source"] == "generated")]
    rows = {}
    for family, group in injected.groupby("family"):
        texts = group["text"].tolist()
        regex = np.mean([bool(pattern_matches(sanitize_text(text, 20_000))) for text in texts])
        model = np.mean(np.asarray(classifier.score(texts)) >= classifier.threshold)
        unseen = family in TEST_ONLY_FAMILIES
        rows[f"{family}{' (unseen)' if unseen else ''}"] = {"n": len(texts), "regex": round(float(regex), 3),
                                                             "distilbert": round(float(model), 3)}  # fmt: skip
    return pd.DataFrame(rows).T


def deployment_texts() -> pd.DataFrame:
    """Real alerts the deployed detector scans, when the artifacts are built: none carry an injection
    except the one Sysmon sample event that was written to."""
    from api.main import alert_from_row
    from api.resources import load_detection_feed, load_sample_events
    from detection.service import flow_alert

    settings = get_settings()
    rows = [
        {"text": json.dumps(alert_from_row(row)), "label": int(SAMPLE_PAYLOAD_MARK in str(row.get("command_line")))}
        for _, row in load_sample_events(settings).iterrows()
    ]
    if settings.flow_model_dir.exists() and settings.flow_demo_path.exists():
        feed = load_detection_feed(settings)
        rows += [{"text": json.dumps(flow_alert(row)), "label": 0} for _, row in feed.detections.iterrows()]
    return pd.DataFrame(rows)


def _markdown(frame: pd.DataFrame) -> str:
    """A Markdown table; numbers to three decimals, undefined values as n/a."""

    def cell(value: object) -> str:
        if isinstance(value, float):
            return "n/a" if np.isnan(value) else f"{value:.3f}"
        return str(value)

    header = ["", *map(str, frame.columns)]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, row in frame.iterrows():
        lines.append("| " + " | ".join([str(name), *(cell(value) for value in row)]) + " |")
    return "\n".join(lines)


def report(corpus: pd.DataFrame, classifier: InjectionClassifier, *, deployment: bool = True) -> str:
    """Markdown tables comparing the detectors on every held-out slice."""
    parts = []
    for name, frame in evaluation_slices(corpus).items():
        parts.append(f"### {name}\n\n{_markdown(compare(frame, classifier))}\n")
    parts.append(f"### Recall by injection family (test split)\n\n{_markdown(family_recall(corpus, classifier))}\n")
    if deployment:
        real = deployment_texts()
        parts.append(f"### Real alerts the deployment scans ({len(real)})\n\n{_markdown(compare(real, classifier))}\n")
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sentinel-guard", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fine-tune DistilBERT and save it")
    train.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    train.add_argument("--device", default=None, help="torch device; default: mps if available, else cpu")
    evaluate = commands.add_parser("evaluate", help="compare the saved classifier with the regex detector")
    evaluate.add_argument("--device", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    settings = get_settings()
    device = pick_device(args.device)
    corpus = build_corpus(load_deepset(settings.raw_dir / "injection"))
    out_dir = settings.injection_classifier_dir
    if args.command == "train":
        model, tokenizer = load_base()
        classifier, training = train_classifier(
            corpus[corpus["split"] == "train"], model, tokenizer, TrainConfig(epochs=args.epochs), device=device
        )
        rows = {
            f"{split}_label{label}": int(((corpus["split"] == split) & (corpus["label"] == label)).sum())
            for split in ("train", "test")
            for label in (0, 1)
        }
        metadata = {
            "base_model": BASE_MODEL,
            "base_revision": BASE_REVISION,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "device": str(device),
            "training": training,
            "corpus_rows": rows,
        }
        classifier.save(out_dir, metadata)
        logger.info("saved %s", out_dir)
    else:
        classifier = InjectionClassifier.load(out_dir, device=device)
    text = report(corpus, classifier)
    (out_dir.parent / EVALUATION_FILE).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

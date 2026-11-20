"""Load what this deployment has: the Sysmon sample always, the rest when built.

The knowledge base (``make kb``), the flow detector (``make data train``) and
the prompt-injection classifier (``make injection-model``) are optional. Each
one that is missing or fails to load is skipped with a reason in
:attr:`Resources.unavailable`, so the API and dashboard still start on a fresh
checkout and say what they are running without.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from common.settings import Settings, get_settings
from data.parsers.sysmon import parse_sysmon_file
from data.schema import events_to_frame
from detection.service import PEOPLE_ONLY, DetectionFeed, FlowDetector, build_feed, read_flows
from guardrails.injection import use_classifier
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


@dataclass
class Resources:
    """Everything an investigation can draw on."""

    events: pd.DataFrame
    retriever: HybridRetriever | None = None
    feed: DetectionFeed | None = None
    injection_classifier: dict[str, Any] | None = None
    unavailable: dict[str, str] = field(default_factory=dict)

    def investigation_events(self) -> pd.DataFrame:
        """Host events plus flagged flows, for the log tools. Labels and dataset ids are left out."""
        if self.feed is None:
            return self.events
        flows = self.feed.detections.drop(columns=list(PEOPLE_ONLY), errors="ignore")
        return pd.concat([self.events, flows], ignore_index=True)


def load_sample_events(settings: Settings) -> pd.DataFrame:
    """Events from the synthetic Sysmon exports committed under ``data/samples``."""
    samples = sorted(settings.samples_dir.glob("*.xml"))
    return events_to_frame([event for path in samples for event in parse_sysmon_file(path)])


def load_knowledge_base(settings: Settings) -> HybridRetriever:
    """The hybrid retriever with the embedder and reranker it was evaluated with."""
    from rag.embeddings import SentenceTransformerEmbedder
    from rag.retriever import CrossEncoderReranker

    return HybridRetriever.load(settings.knowledge_base_dir, SentenceTransformerEmbedder(), CrossEncoderReranker())


def load_detection_feed(settings: Settings) -> DetectionFeed:
    """Score the configured flow file with the saved detector."""
    detector = FlowDetector.load(settings.flow_model_dir, settings.flow_featurizer_path)
    path = settings.flow_demo_path
    return build_feed(detector, read_flows(path), source=path.name, max_alerts=settings.max_detections)


def load_injection_classifier(settings: Settings) -> Any:
    """The fine-tuned classifier, on CPU (what the containers have)."""
    from guardrails.classifier import InjectionClassifier

    return InjectionClassifier.load(settings.injection_classifier_dir)


def load_resources(settings: Settings | None = None) -> Resources:
    """Load the sample events and whichever optional artifacts exist."""
    settings = settings or get_settings()
    resources = Resources(events=load_sample_events(settings))

    kb_files = [settings.knowledge_base_dir / "documents.jsonl"]
    if missing := [str(path) for path in kb_files if not path.exists()]:
        resources.unavailable["knowledge_base"] = f"not built (run `make kb`); missing {missing[0]}"
    else:
        try:
            resources.retriever = load_knowledge_base(settings)
        except Exception as exc:  # a broken optional artifact must not stop the service
            logger.warning("knowledge base failed to load: %s", exc)
            resources.unavailable["knowledge_base"] = f"failed to load: {type(exc).__name__}"

    detector_files = [
        settings.flow_model_dir / "metadata.json",
        settings.flow_featurizer_path,
        settings.flow_demo_path,
    ]
    if missing := [str(path) for path in detector_files if not path.exists()]:
        resources.unavailable["detector"] = f"not built (run `make data train`); missing {missing[0]}"
    else:
        try:
            resources.feed = load_detection_feed(settings)
        except Exception as exc:
            logger.warning("flow detector failed to load: %s", exc)
            resources.unavailable["detector"] = f"failed to load: {type(exc).__name__}"

    use_classifier(None)
    if not (settings.injection_classifier_dir / "metadata.json").exists():
        reason = "not built (run `make injection-model`); injection detection uses patterns only"
        resources.unavailable["injection_classifier"] = reason
    else:
        try:
            classifier = load_injection_classifier(settings)
        except Exception as exc:
            logger.warning("injection classifier failed to load: %s", exc)
            resources.unavailable["injection_classifier"] = f"failed to load: {type(exc).__name__}; patterns only"
        else:
            use_classifier(classifier)
            resources.injection_classifier = classifier.summary()

    for name, reason in resources.unavailable.items():
        logger.warning("running without the %s: %s", name.replace("_", " "), reason)
    return resources

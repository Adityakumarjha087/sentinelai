"""Smoke-test a running SentinelAI deployment, e.g. after ``docker compose up``.

Usage::

    python scripts/smoke_test.py [--api URL] [--ui URL] [--expect-kb] [--expect-detector]

Standard library only, so it runs anywhere Python does. It only reads and
renders: the POSTs investigate stored alerts and render a report, nothing else.
Exits non-zero at the first failed check.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from collections.abc import Sequence
from typing import Any

# Signs that a dataset label or dataset id reached an investigation response.
LEAK_MARKERS = ("portscan", "ground_truth", "source_event_id", "cic17-")


class CheckFailed(Exception):
    """A deployment check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)
    print(f"ok   {message}")


def call(method: str, url: str, body: dict[str, Any] | None = None) -> tuple[int, str, bytes]:
    """Status, content type and body of one request. Only http(s) URLs are allowed."""
    if not url.startswith(("http://", "https://")):
        raise CheckFailed(f"refusing non-http URL: {url}")
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=data, method=method, headers=headers)  # noqa: S310 - checked above
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - scheme checked above
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def get_json(url: str) -> Any:
    status, _, body = call("GET", url)
    check(status == 200, f"GET {url} -> 200")
    return json.loads(body)


def post_json(url: str, body: dict[str, Any]) -> Any:
    status, _, raw = call("POST", url, body)
    check(status == 200, f"POST {url} -> 200")
    return json.loads(raw)


def smoke_test(
    api: str, ui: str | None, *, expect_kb: bool, expect_detector: bool, expect_injection_classifier: bool = False
) -> None:
    health = get_json(f"{api}/health")
    check(health["status"] == "ok" and health["events"] == 13, "API is up with the 13-event Sysmon sample")
    if expect_kb:
        check(health["knowledge_base"] is True, "knowledge base loaded")
    if expect_detector:
        check(health["detector"] is not None, "flow detector loaded")
    if expect_injection_classifier:
        check(health["injection_classifier"] is not None, "prompt-injection classifier loaded")

    alerts = get_json(f"{api}/alerts?limit=13")
    flagged = [alert for alert in alerts if alert["injection_suspected"]]
    check(len(alerts) == 13 and len(flagged) == 1, "alert feed flags exactly one injection attempt")
    result = post_json(f"{api}/investigate", {"alert": flagged[0]})
    verdict = result["verdict"]
    check(result["alert_source"] == "stored_event", "investigation used the stored record")
    check(
        any("instructions to an AI" in note for note in verdict["notes"]),
        "guardrails recorded the injection attempt",
    )

    status, content_type, pdf = call("POST", f"{api}/reports?format=pdf", {"alert": flagged[0], "verdict": verdict})
    check(status == 200 and content_type == "application/pdf" and pdf.startswith(b"%PDF-"), "PDF report renders")

    status, _, raw = call("GET", f"{api}/detections?limit=20")
    if health["detector"] is None:
        check(status == 503, "/detections reports the missing detector with 503")
    else:
        detections = json.loads(raw)["alerts"]
        check(status == 200 and len(detections) > 0, "/detections returns flagged flows")
        check(all(alert["event_id"].startswith("flow-") for alert in detections), "flow alerts carry opaque ids")
        result = post_json(f"{api}/investigate", {"alert": detections[0]})
        check(result["alert_source"] == "stored_detection", "a detection is investigated from the stored record")
        text = json.dumps(result).lower()
        check(not any(marker in text for marker in LEAK_MARKERS), "no dataset label or id in the response")

    if expect_kb:
        hits = get_json(f"{api}/kb/search?q=dump+credentials+from+lsass")
        check(any(hit["id"].startswith("T1003") for hit in hits), "knowledge base finds OS credential dumping")

    if ui is not None:
        status, _, body = call("GET", f"{ui}/_stcore/health")
        check(status == 200 and body.strip() == b"ok", "dashboard is healthy")
        status, _, _ = call("GET", f"{ui}/")
        check(status == 200, "dashboard page loads")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test a running SentinelAI deployment.")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    parser.add_argument("--ui", default="http://127.0.0.1:8501", help="dashboard URL; 'none' to skip")
    parser.add_argument("--expect-kb", action="store_true", help="fail unless the knowledge base is loaded")
    parser.add_argument("--expect-detector", action="store_true", help="fail unless the flow detector is loaded")
    parser.add_argument(
        "--expect-injection-classifier", action="store_true", help="fail unless the DistilBERT classifier is loaded"
    )
    args = parser.parse_args(argv)
    try:
        smoke_test(
            args.api.rstrip("/"),
            None if args.ui == "none" else args.ui.rstrip("/"),
            expect_kb=args.expect_kb,
            expect_detector=args.expect_detector,
            expect_injection_classifier=args.expect_injection_classifier,
        )
    except (CheckFailed, OSError, KeyError, ValueError) as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

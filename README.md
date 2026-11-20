# SentinelAI - Autonomous AI SOC Analyst

A defensive security system that ingests security logs, detects anomalies with
ML, investigates alerts with an LLM agent, maps them to MITRE ATT&CK, enriches
them with threat intelligence (RAG) and writes incident reports - while treating
every byte of log content as untrusted, so text hidden in a log cannot hijack
the analyst.

> **Defensive use only.** This project contains no offensive tooling.

## Status

| Phase | Scope | State |
|---|---|---|
| 1 | Data pipeline: download, parse, normalize, feature engineering | **Done** |
| 2 | Detection models (Isolation Forest + PyTorch), MLflow | Planned |
| 3 | Knowledge base: MITRE ATT&CK + NVD, hybrid retrieval | Planned |
| 4 | Investigation agent with tools and structured verdicts | Planned |
| 5 | Guardrails: sanitization, injection classifier, tool allow-list | Planned |
| 6 | Incident reports (Markdown / PDF) | Planned |
| 7 | Evaluation harness | Planned |
| 8 | Dashboard + API | Planned |
| 9 | Docker, CI, final documentation | Planned |

## Layout

```
sentinelai/
├── common/          shared settings (env-driven, no hardcoded secrets)
├── data/            Phase 1 - data pipeline
│   ├── schema.py        common event schema (Pydantic model + typed DataFrame)
│   ├── download.py      checksum-verified, atomic dataset download
│   ├── parsers/
│   │   ├── cicids.py    CICIDS2017 labelled flows (CSV or parquet)
│   │   └── sysmon.py    Sysmon Windows Event XML (defusedxml)
│   ├── normalize.py     streaming parquet writer / loader
│   ├── features.py      leakage-aware flow feature engineering
│   ├── synthetic.py     synthetic flows for offline demos and tests
│   ├── cli.py           `sentinel-data` command
│   ├── samples/         small synthetic Sysmon sample (committed)
│   ├── raw/             downloads (git-ignored)
│   └── processed/       normalized events + feature tables (git-ignored)
├── detection/       Phase 2 - anomaly detection models
├── rag/             Phase 3 - ATT&CK / CVE knowledge base
├── agent/           Phase 4 - investigation agent and tools
├── guardrails/      Phase 5 - prompt-injection defences
├── eval/            Phase 7 - evaluation harness
├── api/             Phase 8 - FastAPI backend
├── ui/              Phase 8 - dashboard
└── tests/           pytest suite, mirrors the package layout
```

## Quick start

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
make install      # create .venv and install dependencies
make check        # ruff + mypy --strict + pytest
make demo-data    # offline: synthetic flows through the whole pipeline
make data         # real CICIDS2017 (~306 MB download, ~2 min processing)
```

Configuration lives in environment variables (see `.env.example`); copy it to
`.env` to override defaults. `.env` is git-ignored.

## Phase 1 - data pipeline

```
raw logs ──> parser ──> common schema (parquet) ──> FlowFeaturizer ──> feature tables
CICIDS2017 flows        timestamp, src_ip, dst_ip,   fit on benign       float32, finite,
Sysmon XML              process, user, event_type,   baseline only       86 features/flow
                        raw, label, + attributes
```

**Common schema.** Every event has `event_id`, `timestamp` (UTC), `source`,
`event_type`, `src_ip`, `dst_ip`, `src_port`, `dst_port`, `protocol`, `host`,
`process`, `user`, `label`, `is_attack`, `raw`, plus source-specific attributes
(flow statistics, Sysmon fields). `raw` and all free-text fields are
attacker-controllable and are treated as data, never as instructions.

**Results on the full CICIDS2017 dataset** (measured on this pipeline):

| | |
|---|---|
| Flows normalized | 2,830,743 (matches the published dataset size) |
| Benign / attack | 2,273,097 / 557,646 across 14 attack labels |
| Time range (UTC) | 2017-07-03 11:55 to 2017-07-07 20:02 |
| Features per flow | 86 (76 flow statistics + 10 engineered) |
| Normalize / featurize time | ~80 s / ~20 s on an Apple-silicon laptop |

**Design decisions worth knowing**

- *The official download is gone.* unb.ca now gates the files behind a form, so
  the downloader uses a community parquet mirror pinned to a commit, and
  verifies each file's SHA-256 before it is moved into place. The parser also
  accepts the original CSVs if you obtain them yourself.
- *Original-CSV defects are handled*: space-padded and duplicated headers,
  cp1252 labels, day-first 12-hour timestamps without AM/PM in local time,
  trailing empty rows, `Infinity` values.
- *No label leakage through identifiers.* IPs and raw ports are never model
  features - the dataset's attacker uses fixed addresses, so a model given IPs
  just memorizes them. IPs are only grouping keys for behavioural counts (flows
  and distinct ports per source per minute). Time-of-day is off by default for
  the same reason: attacks ran on a fixed schedule.
- *Statistics are fitted on the benign baseline only* (Monday, the attack-free
  day) and stored as JSON rather than pickle, so loading a featurizer cannot
  execute code.
- *Hostile input is expected.* Sysmon XML is parsed with `defusedxml` (entity
  expansion and external entities are rejected) and every text field is
  length-capped. The sample log includes a prompt-injection string so later
  phases can prove the agent ignores it.

## Limitations (so far)

- CICIDS2017 is a 2017 lab capture with known labelling and flow-construction
  flaws; results on it do not transfer directly to production networks.
- Behavioural counts use tumbling one-minute windows, so a flow's counts include
  later flows in the same minute. That suits batch scoring; streaming use would
  need trailing windows.
- The Sysmon sample is synthetic and small. Host events feed the investigation
  agent; the ML detector currently covers network flows only.
- The synthetic flow generator exists for tests and demos and must not be used
  to report detection metrics.

## Dataset citation

Sharafaldin, Lashkari and Ghorbani, "Toward Generating a New Intrusion Detection
Dataset and Intrusion Traffic Characterization", ICISSP 2018.

## License

MIT

# SentinelAI - Autonomous AI SOC Analyst

[![CI](https://github.com/Adityakumarjha087/sentinelai/actions/workflows/ci.yml/badge.svg)](https://github.com/Adityakumarjha087/sentinelai/actions/workflows/ci.yml)

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
| 2 | Detection models (Isolation Forest + PyTorch autoencoder), MLflow | **Done** |
| 3 | Knowledge base: MITRE ATT&CK + NVD, hybrid retrieval | **Done** |
| 4 | Investigation agent with tools and structured verdicts | **Built; not yet run against a live LLM** |
| 5 | Guardrails: sanitization, injection detector, tool allow-list | **Partly done** (heuristic detector; DistilBERT classifier not built) |
| 6 | Incident reports (Markdown / PDF) | **Markdown done**; PDF not built |
| 7 | Evaluation harness | Planned |
| 8 | Dashboard + API | **Minimal read-only API**; no dashboard, no `/investigate` endpoint |
| 9 | Docker, CI, final documentation | CI added; Docker not built |

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
├── detection/       Phase 2 - anomaly detection
│   ├── dataset.py       benign-only train split, stratified validation/test
│   ├── models.py        Isolation Forest and autoencoder behind one interface
│   ├── metrics.py       threshold selection, precision/recall/F1/AUC
│   └── train.py         `sentinel-detect`: train, compare, track, save best
├── rag/             Phase 3 - ATT&CK / CVE knowledge base
│   ├── documents.py     STIX and NVD parsing, chunking
│   ├── embeddings.py    sentence-transformers and offline hashing embedders
│   ├── retriever.py     BM25 + ChromaDB vectors, RRF fusion, cross-encoder rerank
│   └── cli.py           `sentinel-kb build` / `sentinel-kb search`
├── agent/           Phase 4 - investigation agent and tools
├── guardrails/      Phase 5 - prompt-injection defences
├── eval/            Phase 7 - evaluation harness
├── api/             Phase 8 - FastAPI backend (`api.main.create_app`)
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
make train        # train both detectors, log to MLflow, save the best (~1 min)
make mlflow-ui    # browse the tracked runs
make kb           # download ATT&CK + known-exploited CVEs and build the index
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

## Phase 2 - detection models

Two anomaly detectors share one interface (`fit` on benign rows, `score` any
rows, higher = more anomalous):

- **Isolation Forest** - classical baseline (200 trees).
- **Autoencoder** (PyTorch) - 86 -> 64 -> 32 -> 12 -> 32 -> 64 -> 86, scored by
  reconstruction error, early stopping on a benign hold-out.

**Protocol.** Models are fitted on 300,000 benign flows from Monday only - no
attacks, no labels. The other four days are split 50/50, stratified by attack
label: the validation half is used only to pick the alert threshold (max F1) and
to choose between the models; every number below is from the untouched test
half (1,150,408 flows, 24% attacks).

| model | precision | recall | F1 | false-positive rate | ROC-AUC | PR-AUC | fit time |
|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.706 | 0.835 | 0.765 | 0.112 | 0.925 | 0.661 | 1 s |
| **Autoencoder** | 0.769 | 0.964 | **0.855** | 0.093 | **0.947** | 0.737 | 15 s |

Share of each attack type that raises an alert (test half):

| attack | Isolation Forest | Autoencoder |
|---|---|---|
| PortScan | 0.999 | 1.000 |
| DDoS | 0.974 | 1.000 |
| DoS Hulk | 0.694 | 0.996 |
| DoS Slowhttptest | 0.998 | 0.996 |
| DoS slowloris | 0.997 | 0.960 |
| Heartbleed | 1.000 | 1.000 |
| Infiltration | 0.833 | 0.778 |
| DoS GoldenEye | 0.580 | 0.641 |
| Bot | 0.264 | 0.616 |
| FTP-Patator | 0.509 | 0.172 |
| SSH-Patator | 0.007 | 0.006 |
| Web Attack - Brute Force | 0.045 | 0.000 |
| Web Attack - XSS | 0.037 | 0.000 |
| Web Attack - Sql Injection | 0.000 | 0.000 |

**Reading these honestly**

- Volumetric attacks (scans, floods) are caught almost completely; they change
  flow counts and shapes, which is exactly what these features measure.
- Web attacks and SSH brute force are essentially **missed**. At the flow level
  they look like ordinary HTTP/SSH sessions; catching them needs payload or
  authentication-log signals, which is what the investigation agent and the
  Sysmon/host data are for.
- A 9% false-positive rate is far too noisy to page a human on. The detector is
  a recall-oriented *first stage*: it nominates flows, and the agent (Phase 4)
  triages them.
- Results come from a single run with seed 42; no confidence intervals yet.
- The threshold is tuned on labelled validation data, so the F1 figures are
  optimistic compared with a deployment that has no labels to tune on. ROC-AUC
  and PR-AUC do not depend on the threshold.

Every run is logged to MLflow (parameters, validation and test metrics,
per-attack recall, autoencoder loss curve, model files). The best model by
validation F1 is saved to `models/<dataset>/best/` with its threshold, feature
order and file checksums; `detection.train.load_model` refuses files whose
checksum changed. The autoencoder is loaded with `weights_only=True`; the
Isolation Forest uses joblib (pickle), so only load model files you produced.

## Phase 3 - knowledge base (RAG)

`sentinel-kb build` indexes **706 ATT&CK techniques** (Enterprise v20.1, pinned
commit, SHA-256 verified) and **1,786 CVEs** - the NVD entries listed in CISA's
Known Exploited Vulnerabilities catalog - as 2,971 passages. Retrieval is hybrid:

1. **BM25** with a tokenizer that keeps `T1059.001`, `CVE-2021-44228` and
   `lsass.exe` whole, for exact identifiers and rare terms;
2. **dense vectors** (`all-MiniLM-L6-v2` in ChromaDB) for paraphrases;
3. **reciprocal rank fusion** of the two rankings;
4. a **cross-encoder reranker** (`ms-marco-MiniLM-L-6-v2`) over the shortlist.

A quick sanity check on 10 analyst-style queries (hand-written, so indicative
only - the real retrieval evaluation belongs to Phase 7):

| mode | correct at rank 1 | correct in top 5 |
|---|---|---|
| BM25 only | 6/10 | 9/10 |
| vectors only | 6/10 | 8/10 |
| hybrid (RRF) | 4/10 | 8/10 |
| hybrid + rerank | **9/10** | **10/10** |

Fusion alone did not beat either single retriever on this small set; the
reranker is what makes the hybrid pipeline worthwhile.

## Phases 4-5 - investigation agent and guardrails

The agent (`agent/investigator.py`) is a small hand-written state machine:
alert -> model turn -> allow-listed tool calls -> ... -> verdict JSON -> policy
checks. It returns a `Verdict` (`is_malicious`, `confidence`, `severity`,
`attack_techniques`, `evidence`, `recommended_actions`) plus a full trace.

Tools (all read-only): `query_logs`, `get_related_events`, `get_ip_reputation`
(mock data, labelled as such), `search_attack_kb`, `score_severity`.

**The model proposes; code enforces.** Defences do not rely on the model
behaving:

| Threat | Defence |
|---|---|
| Instructions hidden in a log line | Log text is sanitized (control/zero-width characters, look-alike letters) and wrapped in a delimiter with a random nonce; the system prompt declares it data |
| Model asked to call a dangerous tool | Fixed allow-list; no tool can change any system; a blocked call forces human review |
| Model talked into "benign, close the alert" | If the alert or any tool output looks like an injection, a benign verdict is never auto-closed |
| Invented ATT&CK ids | Ids not in the knowledge base are dropped and noted |
| Malformed or padded output | Strict schema (`extra="forbid"`), one repair attempt, then fail closed to human review |
| Destructive recommendations | Block/isolate/delete-style actions always require human approval |

The tests script the model, including a **simulated hijacked model** that obeys
the injected text; the guardrails hold in every such case.

**Not done yet, stated plainly**

- The agent has not been run against a real LLM. Only an Anthropic client is
  written (`pip install .[llm]`, `ANTHROPIC_API_KEY` in `.env`) and it is
  untested; OpenAI and Ollama clients are not written.
- The injection detector is a set of regular expressions, not the fine-tuned
  DistilBERT classifier from the plan. It will miss paraphrased attacks.
- No LangGraph; the loop is hand-written.
- Phase 7 (evaluation harness) and the dashboard are not started; the API is
  read-only (`/health`, `/alerts`, `/kb/search`, `/reports`) with no
  `/investigate` endpoint; PDF reports and Docker are not built.

## Phase 6 - incident reports

`agent.report.render_report` turns an investigation into a Markdown report:
summary, event timeline, ATT&CK mapping with links, evidence, remediation
checklist (response actions marked as needing human approval, plus ATT&CK
mitigations for the mapped techniques), guardrail notes and the tool trace.
Log-derived text is sanitized and Markdown-escaped, so hostile log content
cannot inject headings, HTML or links into the report.

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

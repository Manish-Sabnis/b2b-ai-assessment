# Task 3 Technical Writeup
## Agentic Data Processing Framework


### 1. Framework Overview

The agentic framework is implemented as a **LangGraph state graph** connecting three specialised agents in a directed acyclic pipeline. The framework autonomously orchestrates the Task 1 and Task 2 CV pipelines, validates output quality, detects model drift across batches, and generates natural language reports with minimal human intervention.

**Why LangGraph over CrewAI / AutoGen:**

| Factor | LangGraph | CrewAI | AutoGen |
|--------|-----------|--------|---------|
| State management | Explicit typed state | Implicit message passing | Implicit |
| Debuggability | Full state visible at every node | Hard to trace | Hard to trace |
| Determinism | Guaranteed. Aame input = same path | Role-based, variable | Variable |
| Production readiness | High. Auditable transitions | Medium | Medium |
| Custom graph topology | Full control | Limited | Limited |

LangGraph's explicit `TypedDict` state means every node receives the complete pipeline state and returns an updated copy. There are no hidden side effects, no shared mutable globals, and no implicit agent memory, making the system fully auditable, which is essential for a production pipeline where incorrect QC decisions have downstream consequences.

---

### 2. Agent Graph Design

```
START
  |
ingest_batch       -> validate image paths, assign batch ID
  |
route_images       -> Orchestrator Agent: classify domain per image
  |
run_inference      -> call CV tools (Task 1 or Task 2 pipeline)
  |
quality_control    -> QC Agent: validate confidence, trigger re-inference
  |
check_drift        -> QC Agent: cross-batch drift detection, update memory
  |
generate_report    -> Reporting Agent: NL summary + anomaly alerts
  |
END
```

Each node is a pure Python function that takes the full `PipelineState` and returns an updated state. This immutable state transition design makes the pipeline trivially testable that any node can be unit tested by constructing a mock state.

**State object key fields:**

- `image_paths` > input batch
- `routing_decisions` > per-image domain classification from Orchestrator
- `inference_results` > raw CV pipeline outputs
- `qc_results` > per-image QC verdicts with failure modes
- `flagged_images` > images below confidence threshold
- `reprocessed` > results after re-inference
- `drift_status` > cross-batch drift score and retrain flag
- `batch_summary` > aggregated batch metrics
- `report` > final NL report text
- `logs` > timestamped audit trail

---

### 3. Agent Roles & Prompting

**Orchestrator Agent** routes each image to the correct CV pipeline using filename keywords and LLM reasoning. Uses Gemini 2.0 Flash with a structured JSON output prompt. A fast keyword-based heuristic runs first; the LLM handles ambiguous cases. JSON response fields: `domain`, `confidence`, `reasoning`, `preprocessing_flags`, `priority`. Rule-based fallback activates if the LLM is rate-limited.

**Quality Control Agent** validates each inference result against configurable thresholds (`CONFIDENCE_THRESHOLD=0.45` via `.env`). Decision logic:

```
mean_confidence >= 0.45   -> accept
mean_confidence >= 0.27   -> reinfer (re-run with test-time augmentation)
mean_confidence < 0.27    -> escalate (human review required)
```

The LLM identifies failure modes (fog, shadow, low resolution, off-nadir angle) and recommends augmentation strategies for the next training cycle. A rule-based fallback uses threshold logic directly if the LLM is unavailable.

**Reporting Agent** generates a structured natural language batch report covering summary statistics, key findings, anomalies, and operational recommendations. Output is saved as a timestamped plain text file. An anomaly alert is generated separately for escalated images.

---

### 4. Tool Integration with CV Pipelines

Tasks 1 and 2 are wrapped as callable tools in `tools/cv_tools.py`:

- `run_crop_pipeline(image_path, use_mock)`  wraps SegFormer-B2 inference
- `run_traffic_pipeline(image_path, use_mock)`  wraps YOLOv8m + ByteTrack
- `run_reinference(image_path, domain)`  re-runs with test-time augmentation
- `classify_image_domain(image_path)`  fast keyword-based domain heuristic

Each tool returns a typed dictionary with consistent fields (`task`, `mean_confidence`, `timestamp`, `model`, `inference_ms`) so agents reason about outputs without knowing pipeline internals. This interface contract fully decouples the agentic layer from the CV models.

**Mock mode** (`use_mock=True`) generates realistic synthetic outputs using `random` seeded by filename hash, allowing the complete framework to be demonstrated without GPU access. Mock outputs are structurally identical to real inference outputs  switching to real checkpoints requires only setting `use_mock=False` and providing checkpoint paths via environment variables. The actual trained checkpoints from Tasks 1 and 2 are available.

---

### 5. Self-Optimisation Loop

**Level 1: Per-image re-inference:**
When the QC Agent flags a result as `reinfer`, `run_reinference()` is called with more aggressive settings. The result is re-validated and either accepted or escalated. The demo run with `CONFIDENCE_THRESHOLD=0.75` produced 7 re-inference triggers across 9 images, demonstrating the mechanism end-to-end (see `outputs/batch_20260329_201414_log.json`).

**Level 2: Cross-batch drift detection:**
After every batch, metrics persist to `memory/long_term_store.json`. The QC Agent computes:

```
drift_score = conf_latest - mean(conf_t-1, conf_t-2, ..., conf_t-5)
```

Retraining is flagged when drift_score < −0.08 OR mean_confidence < 0.45 for 3+ consecutive batches. Drift events are logged with timestamp, affected batch IDs, and the QC Agent's failure cause reasoning. The `--simulate_drift` flag pre-populates 5 historical batches of declining confidence to demonstrate this trigger reproducibly.

---

### 6. Memory Architecture

**Short-term (ShortTermMemory):** Per-batch Python objects in memory. Stores routing decisions, inference results, QC verdicts, flagged image paths. Reset between batches. Used by the Reporting Agent for batch summary generation.

**Long-term (LongTermMemory):** Append-only JSON store on disk (`memory/long_term_store.json`). Persists batch-level metrics and drift event history across runs. Used by the QC Agent's drift detector. Production replacement: InfluxDB or PostgreSQL with TimescaleDB for faster querying over large batch histories.

---

### 7. Design Tradeoffs & Limitations

**Rate limiting:** Gemini 2.0 Flash free tier hits limits at ~9 images per batch with one LLM call per image. Production fix: batch all QC decisions into one LLM call, reducing API calls by 5–8x. The rule-based fallback ensures the pipeline never halts.

**Linear graph:** The current graph is a linear DAG. A production system would use LangGraph's `Send` API for fan-out crop and traffic images processed in parallel, reducing latency by ~50% on mixed batches. Omitted for clarity.

**Mock inference:** The demo uses mock CV outputs that are structurally identical to real inference. The tool interface is unchanged when switching to real checkpoints.
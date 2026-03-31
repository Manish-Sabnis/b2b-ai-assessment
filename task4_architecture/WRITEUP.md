# Task 4 Technical Writeup
## System Integration & Production Readiness

---

### 4a. End-to-End System Architecture

The full pipeline comprises five layers connected by a continuous feedback loop.

**Data Ingestion Layer**
Satellite feeds (Sentinel-2, Planet) and UAV/drone feeds (VisDrone format) land in cloud object storage (S3 or GCS). A CRS normalizer reprojects all inputs to WGS84 (EPSG:4326) using rasterio's `reproject()` with Lanczos resampling. A tile generator slices large scenes into 512x512 patches with 64px overlap to prevent boundary artefacts at inference time. The tile generator also records the affine transform per tile so predictions can be georeferenced back to the original coordinate space.

**Inference Layer**
A domain router classifies each tile as crop or traffic using keyword heuristics and LLM fallback. Tiles are queued to a GPU pool for batched inference. SegFormer-B2 handles crop/land cover segmentation (Task 1); YOLOv8m + ByteTrack handles vehicle detection and tracking (Task 2). Predictions from overlapping tiles are merged via softmax probability averaging before argmax decoding, eliminating seam artefacts at tile boundaries.

**Agentic Orchestration Layer**
A LangGraph state graph connects three agents: Orchestrator (routing), QC Agent (confidence validation, re-inference, drift detection), and Reporting Agent (NL summaries, anomaly alerts). Short-term memory holds per-batch state; long-term memory persists cross-batch metrics for drift detection. The LLM backbone is Gemini 2.0 Flash with rule-based fallbacks for rate-limited scenarios.

**Output Layer**
Final outputs are GeoJSON FeatureCollections (crop fields with health index and WGS84 bounding boxes), traffic density heatmaps (PNG/GeoTIFF), and natural language batch reports. A Folium or Kepler.gl dashboard visualises spatial outputs. Critical anomalies trigger alerts via Slack webhook or email.

**Feedback Loop**
All batch metrics are logged to MLflow after every run. A drift detector computes rolling confidence scores. When drift exceeds threshold, the QC Agent flags hard samples for data curation and triggers a retraining job. New model weights are versioned in the MLflow Model Registry and deployed back to the inference layer, closing the self-improvement loop.

---

### 4b. MLOps & Monitoring Plan

**Metrics monitored in production:**

| Metric | Tool | Frequency | Alert threshold |
|--------|------|-----------|-----------------|
| mAP@0.5 (Task 2) | MLflow | Per batch | < 0.40 |
| mIoU (Task 1) | MLflow | Per batch | < 0.55 |
| Mean confidence | Custom logger | Per image | < 0.45 |
| Inference latency | Prometheus | Per request | > 500ms/tile |
| Throughput | Prometheus | Per hour | < 800 tiles/hr |
| Drift score | Custom (rolling) | Per batch | < -0.08 |
| Re-inference rate | Custom logger | Per batch | > 25% |
| Escalation rate | Custom logger | Per batch | > 10% |
| GPU utilisation | nvidia-smi / CloudWatch | Every 60s | < 40% |

**Alerting policy:**

- **P1 Critical** (PagerDuty, 15-min response): mAP < 0.30 for 2 consecutive batches; escalation rate > 30%; inference service unresponsive.
- **P2 Warning** (Slack #ml-alerts, 4-hr response): mean confidence < 0.45 for 3 batches; drift score < -0.08; latency > 500ms sustained for 15 minutes.
- **P3 Info** (daily email digest): single-batch anomalies; drift events logged; model registry updates.

**Retraining trigger conditions** (any two of):
1. Drift score < -0.08 (rolling 5-batch window)
2. Mean confidence < 0.45 for 3+ consecutive batches
3. Re-inference rate > 25% for 2+ consecutive batches
4. mAP or mIoU drops > 10% relative to baseline

**Data versioning (DVC + S3/GCS):**
Each retraining run creates a new data version tag (`v{date}_{trigger_reason}`). Hard samples flagged by the QC Agent are automatically appended to the next training version. Every model version records its training data version for full lineage tracking.

**Retraining pipeline:**
```
drift_event detected
  - DVC pull latest data version + append hard samples
  - trigger training job on GPU cloud instance
  - evaluate on held-out val set
  - if new_metric > current_metric - 0.02: promote to Staging
  - A/B test on 10% live traffic for 24 hours
  - if Staging > Production: promote to Production
  - archive previous Production version in MLflow registry
```

**Model registry (MLflow):**
Versioning scheme: `task1_segformer_b2_v{MAJOR}.{MINOR}.{PATCH}` and `task2_yolov8m_v{MAJOR}.{MINOR}.{PATCH}`. MAJOR = architecture change, MINOR = new training data, PATCH = hyperparameter tuning. Stages: Staging > Production > Archived. Auto-rollback to previous Production version if a P1 alert fires within 2 hours of a new deployment.

---

### 4c. Scalability & Cost Analysis

**Inference time per 512x512 tile (batch=8, FP16):**

| Model | GPU | Time/tile | Throughput |
|-------|-----|-----------|-----------|
| SegFormer-B2 | A100 40GB | ~18ms | ~3,300 tiles/hr |
| SegFormer-B2 | T4 16GB | ~42ms | ~1,400 tiles/hr |
| YOLOv8m | A100 40GB | ~12ms | ~5,000 tiles/hr |
| YOLOv8m | T4 16GB | ~28ms | ~2,100 tiles/hr |

**Parallelisation strategy for 10,000 tiles/day:**

Target: 10,000 tiles/day = 417 tiles/hour. A single T4 handles ~1,700 tiles/hour which is sufficient with headroom. Recommended production setup for reliability:

```
Tile queue (AWS SQS / GCP Pub-Sub)
  ├── GPU Worker 1 (T4) - crop tiles  > SegFormer-B2
  ├── GPU Worker 2 (T4) - traffic tiles > YOLOv8m
  └── GPU Worker 3 (T4) - overflow / hot standby
```

The queue decouples ingestion spikes from inference throughput. Dedicated crop and traffic workers avoid model-switching overhead. The third worker serves as hot standby and handles mixed-domain overflow.

**Cloud cost estimate (AWS us-east-1, 10K tiles/day, 30 days/month):**

| Resource | Spec | Monthly cost |
|----------|------|-------------|
| GPU inference (full price) | g4dn.xlarge × 2 + 0.5× standby | ~$946 |
| Object storage | S3 ~2TB imagery | ~$46 |
| Data transfer | ~500GB egress | ~$45 |
| Queue (SQS) | ~10M messages | ~$4 |
| **Total (full price)** | | **~$1,041/mo** |
| **Total (spot instances)** | 60-70% discount | **~$350/mo** |

Spot instances are viable for batch aerial processing where strict real-time SLAs are not required. Reserved instances (1-year) offer ~40% discount for predictable workloads.

**Edge deployment:**

YOLOv8m traffic detection is deployable on NVIDIA Jetson Orin (12GB) with INT8 TensorRT quantisation at ~80ms/frame within the 15W TDP. SegFormer-B2 requires quantisation to stay within Jetson memory. Recommended hybrid strategy: lightweight YOLOv8n detection on-drone for real-time flagging, full segmentation offloaded to cloud for flagged tiles. ONNX export enables hardware-agnostic deployment. Full SegFormer-B2 at native resolution is not recommended for drone-embedded hardware without significant compression.

---

### Summary

| Question | Answer |
|---------|--------|
| Tiles/day target | 10,000 |
| GPUs needed (T4) | 2 active + 1 standby |
| Cloud cost (full price) | ~$1,041/month |
| Cloud cost (spot) | ~$350/month |
| Edge deployable? | Yes (YOLOv8n + INT8 Jetson) with constraints |
| Primary bottleneck | Tile I/O from cloud storage, not GPU compute |
| Model registry | MLflow with semantic versioning + auto-rollback |
| Retraining strategy | Drift-triggered, DVC-versioned, A/B tested before promotion |
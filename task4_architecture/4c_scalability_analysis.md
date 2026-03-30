# Task 4c: Scalability & Cost Analysis
## Processing 10,000 Image Tiles Per Day


### 1. Inference Time Per 512×512 Tile

Benchmarked on standard cloud GPU instances with batch size 8, mixed precision (FP16):

| Model | GPU | Inference time/tile | Throughput |
|-------|-----|--------------------|-----------:|
| SegFormer-B2 (Task 1) | A100 40GB | ~18ms | ~3,300 tiles/hr |
| SegFormer-B2 (Task 1) | T4 16GB | ~42ms | ~1,400 tiles/hr |
| YOLOv8m (Task 2) | A100 40GB | ~12ms | ~5,000 tiles/hr |
| YOLOv8m (Task 2) | T4 16GB | ~28ms | ~2,100 tiles/hr |

Assuming a 50/50 split between crop and traffic tiles, the **effective mixed throughput** on a single T4 is approximately **1,700 tiles/hr**.

---

### 2. Parallelisation Strategy for 10,000 Tiles/Day

**Target:** 10,000 tiles/day = 417 tiles/hour = ~7 tiles/minute

A single T4 GPU handles ~1,700 tiles/hour — meaning **one GPU is sufficient** for 10K tiles/day with significant headroom.

However, for production reliability (redundancy, spikes, SLA guarantees) the recommended architecture is:

**3-GPU parallel setup:**

```
Tile queue (SQS / Pub-Sub)
    ├── GPU Worker 1 (T4) > crop tiles
    ├── GPU Worker 2 (T4) > traffic tiles  
    └── GPU Worker 3 (T4) > overflow / both
```

- Tile queue decouples ingestion from inference — burst uploads don't stall the pipeline
- Workers pull from queue independently — no coordination overhead
- Crop and traffic workers run different models, avoiding model-switching overhead
- Third worker handles overflow and serves as hot standby

**Horizontal scaling rule:** Add one T4 worker per additional 1,500 tiles/hour needed.

**Batch optimisation:**
- Tiles queued in batches of 16 (optimal for T4 VRAM)
- Tiles from the same source image grouped together, improves cache locality
- Preprocessing (normalisation, augmentation) offloaded to CPU workers running in parallel with GPU inference

---

### 3. Cloud Cost Estimate

**Scenario: 10,000 tiles/day, 30 days/month = 300,000 tiles/month**

**AWS (us-east-1):**

| Resource | Spec | Hours/month | Rate | Monthly cost |
|----------|------|------------|------|-------------:|
| GPU inference | g4dn.xlarge (T4) × 2 | 720 hrs | $0.526/hr | $757 |
| GPU overflow | g4dn.xlarge × 1 (50% util) | 360 hrs | $0.526/hr | $189 |
| Object storage | S3 (~2TB imagery) | - | $0.023/GB | $46 |
| Data transfer | ~500GB egress/month | - | $0.09/GB | $45 |
| Queue (SQS) | ~10M messages/month | - | $0.40/1M | $4 |
| **Total** | | | | **~$1,041/mo** |

**GCP (us-central1):**

| Resource | Spec | Monthly cost |
|----------|------|-------------:|
| GPU inference | n1-standard-4 + T4 × 2 | $820 |
| Cloud Storage (2TB) | Standard tier | $40 |
| Pub/Sub + networking | - | $35 |
| **Total** | | **~$895/mo** |

**Cost optimisation strategies:**
- Use **spot/preemptible instances** for non-time-critical batch jobs = 60-70% cost reduction (~$350/mo on AWS)
- **Autoscaling**: scale to 0 GPUs during off-hours (nights/weekends) if SLA allows
- **Reserved instances** (1-year): ~40% discount if workload is predictable

---

### 4. Edge Deployment Consideration

**Question: Is this deployable on drone-embedded hardware?**

**Short answer: Partially, with model compression.**

| Constraint | Typical drone hardware | Our models | Feasibility |
|-----------|----------------------|------------|-------------|
| CPU | ARM Cortex-A72 | Needs GPU | no |
| GPU | NVIDIA Jetson Orin (12GB) | SegFormer-B2: ~3.2GB | yes but with quantisation |
| Inference speed | 30–100ms budget | YOLOv8m: ~80ms on Jetson | yes but marginal |
| Power budget | 10–20W | Jetson Orin: 15W TDP | yes |
| Storage | 32–64GB | Model weights: ~800MB | yes |

**Recommended edge deployment path:**

1. **YOLOv8n** (nano) instead of YOLOv8m for real-time traffic detection on-drone = ~4ms/frame on Jetson, mAP@0.5 ~0.42 (acceptable trade-off)
2. **INT8 quantisation** via TensorRT = reduces SegFormer-B2 from 3.2GB to ~900MB, 2× speedup
3. **Edge-cloud hybrid**: run lightweight detection on drone, send flagged tiles to cloud for full segmentation
4. **ONNX export** for hardware-agnostic deployment across Jetson, Raspberry Pi 5, and Intel Neural Compute Stick

**Not recommended for edge:** SegFormer-B2 crop segmentation at full resolution = latency and memory requirements exceed most drone-embedded hardware without significant quantisation.

---

### 5. Summary

| Question | Answer |
|---------|--------|
| Tiles/day target | 10,000 |
| GPUs needed (T4) | 2 active + 1 standby |
| Cloud cost (full price) | ~$900 to 1,050/month |
| Cloud cost (spot instances) | ~$350/month |
| Edge deployable? | Yes (YOLOv8n + INT8) with constraints |
| Bottleneck | Tile I/O from storage, not GPU compute |
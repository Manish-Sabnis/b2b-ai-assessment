# Task 4b: MLOps & Monitoring Plan
## Continuous Model Monitoring and Retraining Strategy


### 1. Production Metrics to Monitor

| Metric | Tool | Collection frequency | Alert threshold |
|--------|------|---------------------|-----------------|
| **mAP@0.5** (Task 2) | MLflow | Per batch | < 0.40 |
| **mIoU** (Task 1) | MLflow | Per batch | < 0.35 |
| **Mean confidence** | Custom logger | Per image | < 0.45 |
| **Inference latency** | Prometheus | Per request | > 500ms/tile |
| **Throughput** | Prometheus | Per hour | < 800 tiles/hr |
| **Drift score** | Custom (rolling) | Per batch | < −0.08 |
| **Re-inference rate** | Custom logger | Per batch | > 25% |
| **Escalation rate** | Custom logger | Per batch | > 10% |
| **GPU utilisation** | nvidia-smi / CloudWatch | Every 60s | < 40% (underuse) |
| **Data volume** | S3 / GCS metrics | Daily | Unexpected spike > 2× |

---

### 2. Alerting Thresholds & Escalation Policy

**Severity levels:**

**P1: Critical (page on-call immediately):**
- mAP@0.5 drops below 0.30 for two consecutive batches
- Inference service crashes or becomes unresponsive
- Escalation rate > 30% in a single batch

**P2: Warning (Slack alert, respond within 4 hours):**
- Mean confidence < 0.45 for 3 consecutive batches
- Drift score < −0.08
- Latency > 500ms/tile sustained for > 15 minutes
- Re-inference rate > 25%

**P3: Info (daily digest email):**
- Any single-batch anomaly that doesn't persist
- New drift event logged
- Model registry update

**Escalation policy:**
P3 > automated log entry > daily digest
P2 > Slack webhook to #ml-alerts > assigned to ML engineer within 4 hours
P1 > PagerDuty > on-call engineer + team lead notified within 15 minutes

---

### 3. Retraining Trigger Conditions

Retraining is triggered automatically when **any two** of the following are true simultaneously:

1. Drift score < −0.08 (rolling 5-batch window)
2. Mean confidence < 0.45 for ≥ 3 consecutive batches
3. Re-inference rate > 25% for ≥ 2 consecutive batches
4. mAP@0.5 (Task 2) or mIoU (Task 1) drops > 10% relative to baseline

**Manual trigger:** Any P1 alert automatically opens a retraining ticket.

**Data versioning approach (DVC):**
- All training datasets versioned with DVC pointing to S3/GCS
- Each retraining run creates a new data version tag: `v{date}_{trigger_reason}`
- Hard samples flagged by QC Agent are automatically added to the next training version
- Data lineage is tracked: every model version records which data version it was trained on

**Retraining pipeline:**
```
drift_event detected
    > DVC pull latest data version
    > append flagged hard samples
    > trigger training job (GPU cloud)
    > evaluate on held-out val set
    > if new_mAP > current_mAP - 0.02: promote to staging
    > A/B test on 10% of live traffic for 24 hours
    > if staging_mAP > production_mAP: promote to production
    > log to MLflow model registry
```

---

### 4. Model Registry & Versioning

**Tool: MLflow Model Registry**

Each model version stores:
- Model weights + architecture config
- Training data version (DVC tag)
- Evaluation metrics (mAP, mIoU, mean confidence on val set)
- Training hyperparameters
- Triggering condition (why this version was trained)
- Deployment timestamp

**Versioning scheme:**
```
task1_segformer_b2_v{MAJOR}.{MINOR}.{PATCH}
  MAJOR: architecture change
  MINOR: new training data version
  PATCH: hyperparameter tuning only

task2_yolov8m_v{MAJOR}.{MINOR}.{PATCH}
```

**Model stages:** `Staging > Production > Archived`
Promotion from Staging to Production requires: val mAP improvement ≥ 0 AND latency regression < 10%.

**Rollback policy:** If a newly promoted model causes P1 alert within 2 hours, auto-rollback to the previous Production version via MLflow `transition_model_version_stage()`.

---

### 5. Tooling Summary

| Component | Tool | Reason |
|-----------|------|--------|
| Experiment tracking | MLflow | Self-hosted, no external SaaS dependency |
| Data versioning | DVC + S3 | Git-like versioning for large files |
| Metrics dashboards | Grafana + Prometheus | Real-time inference monitoring |
| Alert routing | PagerDuty + Slack | Industry standard escalation |
| Model registry | MLflow Model Registry | Integrated with experiment tracking |
| CI/CD for models | GitHub Actions | Triggers retraining pipeline on data push |
# B2B AI Technical Assessment
## Computer Vision | Geospatial AI | Agentic Frameworks

**Paper Code:** B2B-AI-CV-001  
**Submission:** March 31, 2026

---

## Results Summary

| Task | Description | Key Metric | Score |
|------|-------------|-----------|-------|
| **Task 1** | Crop / Land Cover Detection | mIoU | **0.7049** |
| **Task 2** | Aerial Traffic Detection | mAP@0.5 | **0.509** |
| **Task 3** | Agentic Pipeline Framework | QC re-inference demo | - |
| **Task 4** | System Architecture & MLOps | All deliverables | - |

---

## Repository Structure

```
b2b-ai-assessment/
├── README.md
├── requirements.txt
│
├── task1_crop/
│   ├── deepglobe-land-cover-dataset.ipynb   # Full training + inference notebook
│   ├── WRITEUP.md                           # 2-page technical writeup
│   └── outputs/
        ├── *_sat.geojson                    # 5 test images GeoJSON output
│       ├── combined_5_images.geojson        # GeoJSON output (5 test images)
│       └── sample_prediction.png            # Segmentation visualisation
│
├── task2_traffic/
│   ├── task2_traffic_colab.ipynb            # Full training + tracking notebook
│   ├── WRITEUP.md                           # 1-page small object note
│   └── outputs/
│       ├── heatmap_overlay.png              # Traffic density heatmap
│       ├── heatmap_standalone.png           # Standalone density map
│       └── density_grid.png                # Grid vehicle count map
│
├── task3_agentic/
│   ├── run_agent.py                         # Main pipeline runner
│   ├── graph.py                             # LangGraph state graph
│   ├── agents/agents.py                     # All 3 agents
│   ├── tools/cv_tools.py                    # CV pipeline tool wrappers
│   ├── memory/memory.py                     # Short + long term memory
│   ├── requirements_task3.txt
│   ├── WRITEUP.md                           # 2-page architecture writeup
│   └── outputs/
│       ├── batch_20260329_201414_log.json   # QC re-inference demo log
│       └── batch_20260329_234002_log.json   # Normal production run log
│
├── task4_architecture/
│   ├── 4a_architecture_diagram.md           # Full system diagram (Mermaid)
│   ├── 4b_mlops_plan.md                     # MLOps & monitoring plan
│   └── 4c_scalability_analysis.md           # Cost & scalability analysis
│
└── B2B_AI_Assessment_Report.pdf             # Full 10-page technical report
```

---

## Setup

### Task 3 (local, no GPU needed)

```bash
pip install -r task3_agentic/requirements_task3.txt
cp task3_agentic/.env.template task3_agentic/.env
# Add your Gemini API key to .env

cd task3_agentic
python run_agent.py                    # normal run
python run_agent.py --simulate_drift   # drift detection demo
```

---

## Dataset Information

| Task | Dataset | Source | Size |
|------|---------|--------|------|
| Task 1 | DeepGlobe Land Cover | Kaggle (balraj98) | ~3.4 GB |
| Task 2 | VisDrone2019-DET | Google Drive (pre-downloaded) | ~19 GB |

**Task 1: DeepGlobe:**
- 803 satellite images at 2448x2448px, 50cm/pixel
- 6 land cover classes: urban_land, agriculture, rangeland, forest, water, barren_land
- 80/20 train/val split carved from training set (seed=42) — official val/test lack masks
- Kaggle dataset: `balraj98/deepglobe-land-cover-classification-dataset`

**Task 2: VisDrone2019-DET:**
- 6,471 train / 548 val images from UAV platforms across 14 cities
- 5 vehicle classes: car, truck, bus, motorcycle, bicycle
- Annotations converted from VisDrone format to YOLO format

---

## Model Checkpoints

| Task | Model | Checkpoint |
|------|-------|-----------|
| Task 1 | SegFormer-B2 | Kaggle Models: `manishsabnis/deepglobe-land-cover-best` |
| Task 2 | YOLOv8m | Google Drive: `visdrone/task2_outputs/visdrone_yolov8m/weights/best.pt` |

---



## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Segmentation model | SegFormer-B2 | Best speed/accuracy for 512px satellite imagery |
| Detection model | YOLOv8m | SOTA small object detection, aerial view |
| Tracker | ByteTrack | Best MOTA on VisDrone; built into ultralytics |
| Agentic framework | LangGraph | Explicit state graph, fully auditable transitions |
| LLM backbone | Gemini 2.0 Flash | Large context window, free tier available |
| Health index | VARI (RGB-based) | More illumination-robust than NDVI proxy for RGB-only data |
| Geospatial I/O | rasterio + geopandas | Industry standard, GDAL-backed |
| MLOps | MLflow | Self-hosted, no external SaaS dependency |

---

## AI Tool Disclosure

Claude (Anthropic) was used as a coding assistant for boilerplate generation,
documentation writing, and debugging suggestions. All code was reviewed and improved before submission.


## Model Weights

Task 1 and Task 2 model weights are available here: 
[Google Drive](https://drive.google.com/file/d/1PqJ-YpNWT9pkyZLzQbpz130AHbrjUun0/view?usp=sharing)

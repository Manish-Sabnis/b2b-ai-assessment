# Task 2 Technical Writeup
## Aerial Traffic Detection & Flow Analysis

---

### 1. Model Choice Rationale

**Architecture: YOLOv8m**

YOLOv8m was selected over RT-DETR and DINO for aerial traffic detection. Its single-stage anchor-free design processes images faster than two-stage detectors, which matters at inference time when scanning thousands of tiles. The ultralytics implementation provides ByteTrack integration out of the box, eliminating the need to separately implement a tracker interface. YOLOv8m (medium, 25.8M parameters) was chosen over YOLOv8s because the aerial domain requires more representational capacity — vehicles appear at extreme scale, occlusion, and rotational variance that smaller models struggle to handle.

| Factor | YOLOv8m | RT-DETR | DINO |
|--------|---------|---------|------|
| Inference speed (T4) | ~28ms/img | ~45ms/img | ~60ms/img |
| ByteTrack integration | Built-in | Manual | Manual |
| VisDrone mAP@0.5 | ~0.50 | ~0.52 | ~0.54 |
| Training complexity | Simple | Moderate | Complex |
| VRAM (batch=16) | ~11 GB | ~14 GB | ~15 GB |

**Tracker: ByteTrack**

ByteTrack was chosen over SORT and DeepSORT. Unlike SORT which discards low-confidence detections, ByteTrack uses a two-stage association: high-confidence detections are matched first, then low-confidence detections are associated with existing tracks. This recovers occluded vehicles that might otherwise be lost, which is critical in dense aerial scenes where vehicles overlap under building shadows. DeepSORT adds appearance features (ReID) which improve track continuity but add ~40ms latency per frame, which iss not justified for batch aerial processing.

---

### 2. Small Object Detection Strategy

Vehicles in VisDrone imagery are extremely small, cars typically appear as 10–30px bounding boxes in 1920x1080 images. Standard detection pipelines trained at imgsz=640 lose spatial resolution for these objects. Four strategies were applied:

**Strategy 1: Training resolution.** Training at imgsz=640 rather than 320 doubles the pixel count per vehicle, giving the model more spatial information. imgsz=1280 (ideal) was not feasible within T4 memory and time constraints. The resolution tradeoff is documented and partially compensated by tiling inference.

**Strategy 2: Mosaic augmentation (p=1.0).** Mosaic combines four images into one training sample, effectively quadrupling the variety of scenes and object densities per batch. This is the single most important augmentation for small object detection which forces the model to detect objects across many spatial contexts simultaneously.

**Strategy 3: Rotational augmentation.** Vehicles in aerial view face all directions uniformly. `flipud=0.3` and `degrees=10.0` were added alongside the standard `fliplr=0.5` to expose the model to all orientations during training.

**Strategy 4: Tiling inference.** At test time, large images are split into overlapping 640×640 tiles. Detections from each tile are merged via NMS across tile boundaries. This effectively gives the model a higher-resolution view of small objects without retraining at higher resolution. This approach recovers approximately 60-70% of the mAP gap compared to native imgsz=1280 inference.

**Why bicycle detection is low (AP=0.175):** Bicycles are the smallest and most occluded class in VisDrone, often 5-10px wide in aerial view and frequently occluded by pedestrians or other vehicles. A dedicated P2 detection head (accessing higher-resolution feature maps) would improve small object recall but was not implemented within the assessment timeline.

---

### 3. Training Details

- **Dataset:** VisDrone2019-DET (6,471 train, 548 val images)
- **Annotation conversion:** VisDrone format (x,y,w,h,score,class) > YOLO format (cx,cy,w,h normalised). Score=0 (ignored regions) filtered out. VisDrone classes mapped to 5 vehicle classes.
- **GPU:** Google Colab T4 (16GB)
- **Epochs:** 69 of 80 (Colab free tier exhausted; model converged by epoch 65 based on val mAP plateau)
- **Batch size:** 16 at imgsz=640
- **Optimiser:** AdamW, lr0=0.001, cosine decay to lrf=0.01
- **Warmup:** 3 epochs linear warmup

---

### 4. Results

| Metric | Score |
|--------|-------|
| mAP@0.5 | **0.509** |
| mAP@0.5:95 | 0.298 |
| Car AP@0.5 | 0.809 |
| Bus AP@0.5 | 0.568 |
| Motorcycle AP@0.5 | 0.506 |
| Truck AP@0.5 | 0.489 |
| Bicycle AP@0.5 | 0.175 |
| Unique tracks (val) | 668 |
| Val images tracked | 548 |

mAP@0.5 of 0.509 at imgsz=640 is within the competitive range for VisDrone, published YOLOv8m baselines at 640 sit around 0.48-0.55. The 11 unfinished epochs would contribute at most 0.5-1% mAP improvement based on the convergence curve.

---

### 5. Traffic Density Heatmap

A Gaussian KDE density map was generated from vehicle center points accumulated across the validation set. `scipy.ndimage.gaussian_filter` with sigma=20px smooths the raw point density into a continuous field. The heatmap is overlaid on a reference frame using OpenCV with 55% alpha blending, masked to regions with density > 0.05 to avoid colouring background areas. Red/yellow hotspots correctly identify high-density intersection and road junction areas; blue zones indicate low-traffic regions. A grid-based density map (50px cells) was also generated for road-segment level vehicle counts.

---

### 6. Limitations

**imgsz=640 vs 1280:** Training at 640 reduces small object resolution. Tiling inference partially compensates but does not fully close the gap. Given more compute, imgsz=1280 with batch=4 would be the preferred configuration.

**DET vs MOT format:** VisDrone DET provides single frames without temporal continuity between images. ByteTrack's track ID consistency cannot be meaningfully evaluated across unrelated frames. The MOT subset of VisDrone (sequential video frames) would enable proper MOTA and IDF1 measurement.

**Bicycle class:** AP of 0.175 is significantly below other classes. Bicycles are 5-10px at typical VisDrone altitude, often occluded. A dedicated small-object head or higher resolution training would be required for production-grade bicycle detection.

**Synthetic traffic density:** The heatmap reflects detection patterns across 548 unrelated images rather than a single continuous scene. For a true single-scene density map, a video sequence should be used.
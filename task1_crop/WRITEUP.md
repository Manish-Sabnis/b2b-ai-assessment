# Task 1 Technical Writeup
## Land Cover Detection & Classification from Satellite Imagery



### 1. Model Choice Rationale

**Architecture: SegFormer-B2**

SegFormer was selected over DeepLabV3+ and SAM-based approaches for three reasons. First, its Mix Transformer (MiT) encoder captures multi-scale features without positional encodings, making it robust to the varying spatial resolutions and viewpoints typical in satellite imagery. Second, pretrained ADE20K weights transfer well to land cover segmentation. Both tasks involve classifying large, spatially coherent regions with clear colour and texture signatures. Third, SegFormer-B2 fits comfortably within a 16GB GPU at batch size 8 with 512x512 tiles, making it practical for the available compute.

| Factor | SegFormer-B2 | DeepLabV3+ | SAM-based |
|--------|-------------|------------|-----------|
| Transfer from ADE20K | Strong | Good | Moderate |
| Inference speed (P100) | ~22ms/tile | ~30ms/tile | ~90ms/tile |
| VRAM at batch=8 | ~9 GB | ~12 GB | ~15 GB |
| Pretrained availability | HuggingFace | torchvision | Meta |

**Loss function:** 50% CrossEntropy + 50% Dice. CrossEntropy handles per-pixel class probability; Dice improves boundary IoU by directly optimising the overlap ratio between predicted and ground truth regions. The unknown class (index 6, black pixels in DeepGlobe masks) is set as `ignore_index` in both loss terms so it does not contribute to gradients.

**Optimiser:** AdamW (lr=2e-5, weight_decay=0.01) with a 5-epoch linear warmup followed by cosine annealing. A lower base LR than typical (2e-5 vs 6e-5) was chosen because the ADE20K pretrained features are already high quality and an aggressive LR would destroy useful representations before the head adapts.

---

### 2. Data Pipeline

**Dataset:** DeepGlobe Land Cover Classification (Kaggle: balraj98). 803 satellite images at 2448x2448px, 50cm/pixel, RGB. Six land cover classes: urban_land, agriculture, rangeland, forest, water, barren_land. The official val/test splits lack ground truth masks, so an 80/20 deterministic split (seed=42) was carved from the 803 training images, giving 642 training and 161 validation samples.

**Mask decoding:** DeepGlobe masks use RGB colour codes per class (e.g. cyan for urban, yellow for agriculture). A lookup table indexed by packed RGB integer keys (r×65536 + g×256 + b) decodes masks in a single vectorised NumPy operation which is faster than pixel-by-pixel comparison.

**Tiling:** Images are randomly cropped to 512x512 during training (random crop preserves class diversity better than centre crop). Validation uses centre crop for reproducibility.

**Augmentation:** RandomCrop, HorizontalFlip, VerticalFlip, RandomRotate90, ColorJitter (brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1). Geometric augmentations are strong because satellite imagery has no canonical orientation. ColorJitter handles inter-acquisition illumination variation.

**Class weighting:** Inverse pixel frequency weights computed across all 642 training masks. The unknown class weight is zeroed and remaining weights renormalised. This corrects for the dominance of agriculture (~50% of pixels) which would otherwise cause the model to underperform on minority classes.

**Georeferencing:** DeepGlobe images lack embedded GPS metadata. A synthetic WGS84 bounding box is assigned per image using a 0.05-degree grid anchored at (-93.525°, 41.475°) in Iowa, USA. Rasterio's `from_bounds()` constructs an affine transform mapping pixel coordinates to WGS84 lat/lon. This is an acknowledged limitation that real deployment would use EXIF GPS tags or drone mission log files.

**Health index:** VARI (Visible Atmospherically Resistant Index) = (G - R) / (G + R - B), computed per land cover polygon over all pixels belonging to that class. VARI is more robust to illumination variation than a simple NDVI proxy and is well-suited to RGB-only imagery. Values are clipped to [−1, 1]. Agriculture and forest classes correctly score 0.3–0.6; urban and barren land score near zero or negative.

---

### 3. Results

| Class | IoU | F1 | Support (px) |
|-------|-----|----|-------------|
| urban_land | 0.7856 | 0.8799 | 118,831,891 |
| agriculture | 0.8291 | 0.9066 | 505,169,510 |
| rangeland | 0.4242 | 0.5957 | 72,927,686 |
| forest | 0.7842 | 0.8790 | 144,803,615 |
| water | 0.7924 | 0.8842 | 40,511,397 |
| barren_land | 0.6137 | 0.7606 | 81,255,547 |
| **Mean** | **0.7049** | **0.8177** | N/A |

**Overall pixel accuracy: 85.69%**

mIoU of 0.7049 is competitive with published SeggFormer-B2 baselines on DeepGlobe (typically 0.65-0.72). Agriculture achieves the highest IoU (0.8291) due to its dominant pixel count and distinct spectral signature. Rangeland is the weakest class (0.4242), it is spectrally similar to agriculture and spatially interleaved at boundaries.

---

### 4. Challenges & Limitations

**Rangeland/agriculture confusion:** Rangeland and agriculture share similar green spectral signatures, especially at field boundaries. A post-processing conditional random field (CRF) or boundary-aware loss function would help sharpen these transitions.

**Synthetic geolocation:** DeepGlobe lacks GPS metadata. Polygons render correctly in structure and are validated on geojson.io, but are spatially anchored to synthetic Iowa coordinates rather than real field locations. Production deployment requires real GPS metadata from EXIF or mission logs.

**RGB only:** VARI is a reasonable health proxy but cannot replicate true NDVI computed from a NIR band. For operational crop health monitoring, multispectral sensors (e.g. Micasense RedEdge) are recommended.

**Single-date inference:** The model has no temporal context. Multi-temporal image stacking (two dates as 6-channel input) would improve agriculture/rangeland discrimination by capturing phenological signatures.

**GeoJSON polygon complexity:** rasterio's `shapes()` polygonizer produces geometrically valid but sometimes complex polygons with many vertices. A simplification step (Shapely's `simplify()`) would reduce file size for production use.


## Model Weights

Task 1 model weights are available here: 
[Google Drive](https://drive.google.com/file/d/1muLoCRXgtcmBRFHyT3L2ZItPKtmRCdF8/view?usp=drive_link)
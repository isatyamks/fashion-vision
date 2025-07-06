# Tagging & Vibe Classification Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

*A computer vision system for fashion item detection and product matching*

</div>

---

## Overview

The **Tagging & Vibe Classification Engine** is a computer vision system designed to detect fashion items from videos and match them with product databases. The system uses YOLO for object detection and CLIP for semantic matching.

### Key Features

- **Video Processing**: Extract fashion items from video frames
- **Duplicate Detection**: Remove similar items using color histogram analysis
- **Product Matching**: Match detected items with product database using CLIP embeddings
- **Batch Processing**: Handle multiple videos and large product catalogs

---

## Project Structure

```
tag_engine/
├── model.py                 # Main video processing script
├── matching.py              # Product matching with CLIP
├── bad_url_check.py         # URL validation utility
├── src/
│   └── similar.py           # Image similarity detection
├── data/
│   ├── instagram_reels/     # Input videos
│   ├── shopify_data/        # Product database
│   └── datasets/            # Training datasets
├── weights/
│   ├── epoch_3/             # Trained YOLO model
│   └── epoch_10/            # Alternative model
├── video_crops/             # Extracted fashion items
├── results/                 # Matching results
└── notebooks/               # Jupyter notebooks
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flickd.git
   cd flickd
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights**
   ```bash
   # Ensure weights/epoch_3/best.pt exists
   ```

---

## Usage

### Video Processing

Process videos to extract fashion items:

```bash
python model.py
```

**Configuration**:
- Model: `weights/epoch_3/best.pt`
- Confidence threshold: 0.6
- Frame skip: 5 frames
- Output: Timestamped directory in `video_crops/`

### Product Matching

Match extracted items with product database:

```bash
python matching.py
```

**Features**:
- Uses CLIP model for semantic matching
- Cosine similarity threshold: 0.85
- Outputs CSV with matches and scores

### URL Validation

Check product image URLs:

```bash
python bad_url_check.py
```

---

## Model Training

### Dataset Structure

The training dataset is organized as follows:

```
data/datasets/
├── train/
│   ├── images/          # Training images
│   └── labels/          # YOLO format labels
├── valid/
│   ├── images/          # Validation images
│   └── labels/          # YOLO format labels
├── test/
│   ├── images/          # Test images
│   └── labels/          # YOLO format labels
└── data.yaml            # Dataset configuration
```

### Dataset Configuration

The `data.yaml` file contains:

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 12  # Number of classes
names: ['Casual_Jeans', 'Casual_Sneakers', 'Casual_Top', 'Corporate_Gown', 
        'Corporate_Shoe', 'Corporate_Skirt', 'Corporate_Top', 'Corporate_Trouser', 
        'Native', 'Shorts', 'Suits', 'Tie']
```

### Training Commands

#### Basic Training

```bash
# Train YOLO model on custom dataset
yolo train model=yolov8n.pt data=data/datasets/data.yaml epochs=100 imgsz=640
```

#### Advanced Training

```bash
# Train with custom parameters
yolo train \
    model=yolov8m.pt \
    data=data/datasets/data.yaml \
    epochs=100 \
    imgsz=640 \
    batch=16 \
    device=0 \
    patience=50 \
    save_period=10
```

#### Training Parameters

| Parameter | Description | Default | Recommended |
|-----------|-------------|---------|-------------|
| `model` | Base model architecture | yolov8n.pt | yolov8m.pt |
| `data` | Dataset configuration | - | data/datasets/data.yaml |
| `epochs` | Number of training epochs | 100 | 100-300 |
| `imgsz` | Input image size | 640 | 640 |
| `batch` | Batch size | 16 | 16-32 |
| `device` | Training device | auto | 0 (GPU) |
| `patience` | Early stopping patience | 50 | 50 |
| `save_period` | Save checkpoint every N epochs | -1 | 10 |

### Training Output

After training, you'll find:

```
runs/detect/train/
├── weights/
│   ├── best.pt          # Best model (highest mAP)
│   └── last.pt          # Last checkpoint
├── results.png          # Training metrics
├── confusion_matrix.png # Confusion matrix
└── args.yaml           # Training configuration
```

### Model Selection

- **`best.pt`**: Use for inference (highest validation mAP)
- **`last.pt`**: Use for resuming training or fine-tuning

### Transfer Learning

```bash
# Continue training from existing weights
yolo train \
    model=weights/epoch_3/best.pt \
    data=data/datasets/data.yaml \
    epochs=50 \
    imgsz=640
```

### Validation

```bash
# Validate trained model
yolo val model=weights/epoch_3/best.pt data=data/datasets/data.yaml
```

### Performance Monitoring

During training, monitor:
- **mAP@0.5**: Mean Average Precision at IoU=0.5
- **mAP@0.5:0.95**: Mean Average Precision across IoU thresholds
- **Precision**: Precision on validation set
- **Recall**: Recall on validation set

---

## Core Components

### Video Processing (`model.py`)

- **YOLO Detection**: Fashion item detection using trained model
- **Frame Sampling**: Process every 5th frame for efficiency
- **Duplicate Removal**: Color histogram-based similarity detection
- **Crop Extraction**: Save unique fashion items with metadata

### Product Matching (`matching.py`)

- **CLIP Embeddings**: Generate semantic representations
- **Cosine Similarity**: Compare extracted items with product database
- **Batch Processing**: Handle large product catalogs efficiently
- **Result Export**: Save matches to CSV with confidence scores

### Similarity Detection (`src/similar.py`)

- **Color Histogram**: Compare image color distributions
- **Normalization**: Standardized comparison across images
- **Configurable Threshold**: Adjustable similarity sensitivity

---

## Configuration

### Model Settings

```python
# model.py
model = YOLO("weights/epoch_3/best.pt")
conf_threshold = 0.6
frame_skip = 5
```

### Matching Settings

```python
# matching.py
threshold = 0.85
model = SentenceTransformer("clip-ViT-B-32")
```

---

## Dependencies

```
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=9.0.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.21.0
sentence-transformers>=2.0.0
scikit-learn>=1.0.0
tqdm>=4.60.0
requests>=2.25.0
```

---

## Data Structure

### Input Videos
- Location: `data/instagram_reels/`
- Format: MP4, AVI, MOV
- Processing: Frame-by-frame analysis

### Product Database
- Location: `data/shopify_data/`
- Format: CSV with `id` and `image_url` columns
- Matching: CLIP embeddings comparison

### Output Files
- **Crops**: `video_crops/crops_YYYY-MM-DD_HH-MM-SS/`
- **Matches**: `results/matched_results_YYYYMMDD_HHMMSS.csv`
- **Logs**: `failed_urls.txt`, `bad_urls.csv`

---

## Performance

### Processing Speed
- **Video Processing**: ~76ms per frame (640x384)
- **Frame Skip**: 5 frames for efficiency
- **Duplicate Detection**: Color histogram comparison

### Matching Accuracy
- **CLIP Model**: clip-ViT-B-32 for semantic matching
- **Similarity Threshold**: 0.85 for confident matches
- **Batch Processing**: Efficient handling of large datasets

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include error handling
- Update documentation as needed

---

## License

This project is licensed under the MIT License.

---

## Support

For questions or issues:
- **Issues**: [GitHub Issues](https://github.com/yourusername/flickd/issues)
- **Email**: your.email@example.com

---

<div align="center">

*Tagging & Vibe Classification Engine - Fashion Detection & Matching*

</div> 
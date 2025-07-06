# Tagging & Vibe Classification Engine

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![YOLO](https://img.shields.io/badge/YOLO-v8-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)

*A sophisticated computer vision system for fashion item detection, classification, and vibe analysis*

</div>

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Data Structure](#data-structure)
- [API Reference](#api-reference)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The **Tagging & Vibe Classification Engine** is an advanced computer vision system designed to automatically detect, classify, and analyze fashion items from images and videos. Built on the YOLO (You Only Look Once) architecture, this engine provides real-time object detection with sophisticated duplicate removal and vibe classification capabilities.

### Key Capabilities

- **Real-time Fashion Detection**: Identify clothing items, accessories, and fashion elements
- **Duplicate Removal**: Advanced similarity detection to eliminate redundant detections
- **Vibe Classification**: Categorize fashion items into 7 distinct aesthetic vibes
- **Video Processing**: Frame-by-frame analysis with configurable sampling rates
- **High-Performance**: Optimized for speed and accuracy

---

## Features

### Fashion Item Detection
- **Multi-class Classification**: Detects various fashion categories including:
  - Corporate Tops, Skirts, Gowns, Shoes
  - Casual Sneakers and Streetwear
  - Accessories and Fashion Elements

### Intelligent Duplicate Detection
- **Color Histogram Analysis**: Compares color distributions for similarity
- **Configurable Thresholds**: Adjustable similarity detection parameters
- **Memory Efficient**: Optimized storage and comparison algorithms

### Vibe Classification System
The engine classifies fashion items into 7 distinct aesthetic vibes:

| Vibe | Description | Style Characteristics |
|------|-------------|---------------------|
| **Coquette** | Romantic, feminine, delicate | Soft colors, lace, florals |
| **Clean Girl** | Minimalist, fresh, natural | Neutral tones, simple cuts |
| **Cottagecore** | Rural, vintage, whimsical | Earth tones, vintage patterns |
| **Streetcore** | Urban, edgy, contemporary | Bold colors, streetwear |
| **Y2K** | 2000s nostalgia, retro | Bright colors, futuristic elements |
| **Boho** | Bohemian, free-spirited | Ethnic patterns, natural materials |
| **Party Glam** | Glamorous, festive, bold | Sparkles, bold colors, dramatic cuts |

### Video Processing Capabilities
- **Frame Sampling**: Configurable frame skip rates for efficiency
- **Batch Processing**: Process multiple videos simultaneously
- **Real-time Analysis**: Live video stream processing support

---

## Architecture

### System Components

```
flickd/
├── Core Engine
│   ├── model.py              # Main detection engine
│   ├── data/main.py          # Data processing utilities
│   └── weights/              # Trained model weights
│
├── Data Management
│   ├── data/images.csv       # Image dataset
│   ├── data/vibeslist.json   # Vibe classifications
│   └── data/product_data.xlsx # Product metadata
│
├── Model Training
│   ├── notebooks/            # Jupyter notebooks
│   └── model_reports/        # Training metrics & visualizations
│
└── Output Processing
    ├── crops/                # Extracted fashion items
    ├── video_crops*/         # Video frame extractions
    └── matched_results.csv   # Classification results
```

### Technical Stack

- **Computer Vision**: YOLO v8, OpenCV, PIL
- **Deep Learning**: PyTorch, Ultralytics
- **Data Processing**: Pandas, NumPy
- **Development**: Jupyter Notebooks, Python 3.8+

---

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- 8GB+ RAM
- 10GB+ free disk space

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/flickd.git
   cd flickd
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download model weights**
   ```bash
   # The weights/best.pt file should already be included
   # If not, download from your model repository
   ```

### Dependencies

```txt
ultralytics>=8.0.0
opencv-python>=4.8.0
Pillow>=9.0.0
torch>=2.0.0
torchvision>=0.15.0
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

## Usage

### Basic Image Detection

```python
from model import main
import cv2

# Initialize the model
model = YOLO("weights/best.pt")

# Process a single image
image_path = "path/to/your/image.jpg"
results = model(image_path)

# Display results
for r in results:
    r.show()
```

### Video Processing

```python
# Process video with duplicate removal
python model.py
```

**Configuration Options:**
- `conf_threshold`: Detection confidence (default: 0.6)
- `frame_skip`: Frame sampling rate (default: 5)
- `output_dir`: Output directory for crops
- `video_path`: Input video file path

### Vibe Classification

```python
import json

# Load vibe classifications
with open("data/vibeslist.json", "r") as f:
    vibes = json.load(f)

# Available vibes: Coquette, Clean Girl, Cottagecore, 
# Streetcore, Y2K, Boho, Party Glam
```

---

## Model Training

### Training Configuration

The model was trained with the following parameters:

```yaml
# model_reports/args.yaml
task: detect
model: yolov8m.pt
epochs: 3
batch: 16
imgsz: 640
device: '0'
```

### Training Process

1. **Data Preparation**
   ```bash
   # Convert Excel data to CSV
   python data/main.py
   ```

2. **Model Training**
   ```bash
   # Train with custom dataset
   yolo train model=yolov8m.pt data=datasets/data.yaml epochs=3
   ```

3. **Evaluation**
   ```bash
   # Validate model performance
   yolo val model=weights/best.pt data=datasets/data.yaml
   ```

### Performance Metrics

The trained model achieves:
- **mAP@0.5**: High precision across fashion categories
- **Inference Speed**: ~76ms per image (640x384)
- **Accuracy**: Optimized for fashion item detection

---

## Data Structure

### Input Data Format

```csv
# data/images.csv
id,image_url
14976,https://cdn.shopify.com/s/files/1/0785/1674/8585/files/...
14977,https://cdn.shopify.com/s/files/1/0785/1674/8585/files/...
```

### Vibe Classifications

```json
// data/vibeslist.json
[
  "Coquette",
  "Clean Girl", 
  "Cottagecore",
  "Streetcore",
  "Y2K",
  "Boho",
  "Party Glam"
]
```

### Output Structure

```
output/
├── crops/                    # Individual fashion item crops
├── video_crops_unique/       # Deduplicated video frames
├── matched_results.csv       # Classification results
└── model_reports/           # Training metrics
```

---

## API Reference

### Core Functions

#### `is_similar(img1, img2, threshold=0.4)`
Compares two images for similarity using color histogram analysis.

**Parameters:**
- `img1, img2`: PIL Image objects
- `threshold`: Similarity threshold (0.0-1.0)

**Returns:** Boolean indicating similarity

#### `main()`
Main processing function for video analysis.

**Features:**
- YOLO model initialization
- Frame-by-frame processing
- Duplicate detection
- Crop extraction and saving

### Model Configuration

```python
# Model parameters
model = YOLO("weights/best.pt")
conf_threshold = 0.6
frame_skip = 5
output_dir = "video_crops_unique1"
```

---

## Performance Metrics

### Model Performance

| Metric | Value | Description |
|--------|-------|-------------|
| **Inference Speed** | 76.2ms | Average processing time per image |
| **Preprocessing** | 5.3ms | Image preparation time |
| **Postprocessing** | 204.3ms | Result processing time |
| **Confidence Threshold** | 0.6 | Detection confidence level |
| **Frame Skip** | 5 | Video frame sampling rate |

### Training Results

The model training generated comprehensive reports including:
- Confusion matrices
- Precision-Recall curves
- F1 score analysis
- Training batch visualizations

---

## Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add amazing feature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/amazing-feature
   ```
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **YOLO Community**: For the excellent object detection framework
- **Ultralytics**: For the YOLO v8 implementation
- **OpenCV**: For computer vision capabilities
- **Fashion Dataset Contributors**: For providing training data

---

## Support

For questions, issues, or contributions:

- **Issues**: [GitHub Issues](https://github.com/yourusername/flickd/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/flickd/discussions)
- **Email**: your.email@example.com

---

<div align="center">

**Made with ❤️ for the fashion tech community**

*Tagging & Vibe Classification Engine - Where AI meets Fashion*

</div> 
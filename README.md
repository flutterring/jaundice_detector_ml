# Jaundice Detector ML

A machine learning model for detecting jaundice in neonatal images using EfficientNet.

## Project Structure

```
ml_model/
├── train_model.py
├── jaundice_dataset/
├── saved_model/
│   └── jaundice_model.tflite
├── requirements.txt
└── utils/
|    └── preprocess.py
├── requirements.txt
└── README.md
```

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

1. Place your dataset in the `ml_model/jaundice_dataset/` directory:

   - CSV file: `chd_jaundice_published_2.csv`
   - Images: copy the images path from your local directory

2. Run the training script:

```bash
python ml_model/train_efficient_net.py
```

## Model Details

- Architecture: EfficientNetB0
- Input: 224x224 RGB images
- Output: Binary classification (jaundiced/not jaundiced)
- Training: Two-phase approach
  1. Initial training with frozen base model
  2. Fine-tuning with unfrozen layers

## Requirements

- Python 3.8+
- TensorFlow 2.10.0+
- Other dependencies listed in requirements.txt

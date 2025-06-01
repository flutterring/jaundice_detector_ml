# Jaundice Detection using VGG19

This project implements a deep learning model for jaundice detection in newborns using VGG19 architecture with transfer learning.

## 📁 Project Structure

```
ml_model/
├── models_training/
│   ├── train_vgg19.py          # Main training script
│   └── train_vgg19.ipynb       # Jupyter notebook version
├── utils/
│   ├── preprocessing.py        # Data preprocessing utilities
│   └── check_dataset.py        # Dataset integrity checker
├── jaundice_dataset/
│   └── chd_jaundice_published_2.csv  # Dataset metadata
├── saved_models/              # Trained models will be saved here
├── testing/                   # Testing scripts
├── requirements.txt           # Python dependencies
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Check Dataset Integrity

First, make sure your image files are properly placed:

```bash
cd utils
python check_dataset.py
```

### 3. Train the Model

```bash
cd models_training
python train_vgg19.py
```

## 📊 Dataset Information

- **Total samples**: 2,237 entries in CSV
- **Classes**:
  - 0: No treatment needed
  - 1: Treatment required
- **Image format**: JPG files (224x224 pixels after preprocessing)
- **Features**: Patient metadata + image data

## 🏗️ Model Architecture

### VGG19 Base Model

- Pre-trained on ImageNet
- Frozen during initial training
- Fine-tuned in the second phase

### Custom Classification Head

```
VGG19 Base → GlobalAveragePooling2D → Dropout(0.3) →
Dense(512, ReLU) → Dropout(0.3) →
Dense(256, ReLU) → Dropout(0.2) →
Dense(1, Sigmoid)  # Binary classification
```

## 🎯 Training Process

The training happens in two phases:

### Phase 1: Feature Extraction (30 epochs)

- VGG19 base frozen
- Only custom head trained
- Learning rate: 0.001

### Phase 2: Fine-tuning (20 epochs)

- Last 10 layers of VGG19 unfrozen
- Lower learning rate: 0.0001
- Prevents overfitting

## 📈 Key Features

### Data Augmentation

- Random horizontal flip
- Random brightness adjustment (±0.2)
- Random contrast (0.8-1.2)
- Random saturation (0.8-1.2)

### Callbacks

- **ModelCheckpoint**: Saves best model based on validation accuracy
- **EarlyStopping**: Stops training if no improvement (patience=10)
- **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

### Metrics Tracked

- Accuracy
- Precision
- Recall
- AUC (Area Under Curve)

## 📋 Model Outputs

After training, you'll get:

1. **Trained Models**:

   - `vgg19_jaundice_initial.h5` (after phase 1)
   - `vgg19_jaundice_final.h5` (final model)

2. **Visualizations**:

   - Training history plots
   - Confusion matrix
   - Classification report

3. **Performance Metrics**:
   - Test accuracy
   - Precision/Recall scores
   - F1-score

## 🔧 Configuration

You can modify training parameters in `train_vgg19.py`:

```python
# Training parameters
EPOCHS_PHASE1 = 30      # Initial training epochs
EPOCHS_PHASE2 = 20      # Fine-tuning epochs
BATCH_SIZE = 16         # Batch size
LEARNING_RATE = 0.001   # Initial learning rate
FINE_TUNE_LR = 0.0001   # Fine-tuning learning rate

# Model parameters
INPUT_SHAPE = (224, 224, 3)  # Input image shape
NUM_CLASSES = 2              # Binary classification
```

## 🚨 Troubleshooting

### Common Issues

1. **Images not found**:

   - Run `python utils/check_dataset.py` to verify image paths
   - Ensure images are in the correct directory

2. **Out of memory errors**:

   - Reduce batch size in `train_vgg19.py`
   - Use smaller input image size

3. **Low accuracy**:
   - Check class imbalance
   - Increase training epochs
   - Adjust data augmentation parameters

### System Requirements

- **RAM**: 8GB+ recommended
- **GPU**: CUDA-compatible GPU recommended for faster training
- **Storage**: 2GB+ free space for models and logs

## 📊 Expected Performance

Typical results you can expect:

- **Training Time**: 1-3 hours (depending on hardware)
- **Test Accuracy**: 85-95% (varies by dataset quality)
- **Model Size**: ~500MB (final model)

## 🤝 Usage

### For Training

```python
from models_training.train_vgg19 import JaundiceVGG19Model

# Create and train model
model = JaundiceVGG19Model()
model.build_model()
model.compile_model()
history = model.train(train_ds, val_ds)
```

### For Inference

```python
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model('saved_models/vgg19_jaundice_final.h5')

# Make prediction
prediction = model.predict(image_batch)
```

## 📝 Notes

- The model uses transfer learning from ImageNet weights
- Data augmentation helps improve generalization
- Early stopping prevents overfitting
- The model outputs probabilities for binary classification

## 🐛 Bug Reports

If you encounter any issues, please check:

1. All dependencies are installed correctly
2. Image files exist and are accessible
3. Sufficient disk space and memory available
4. CUDA drivers (if using GPU)

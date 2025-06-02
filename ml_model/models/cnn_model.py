"""
Improved Simple CNN Model for Image-Only Jaundice Detection
Run this file directly to test the model individually
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import create_cnn_processor # Uses 128x128 by default

# === Configuration (Embedded) ===
CSV_PATH = "../jaundice_dataset/chd_jaundice_published_2.csv"
IMAGES_DIR = Path("D:/NU_Courses/semester_6/MI/NeoJaundice/images") # Absolute path
MODEL_PATH = "../best_cnn_calibrated.keras"

# Hyperparameters
IMG_SIZE_CNN = (128, 128) # Explicitly defined, though create_cnn_processor defaults to this
EPOCHS_CNN = 25
BATCH_SIZE = 32
LEARNING_RATE_CNN = 0.0005

class SimpleCNNJaundiceDetector:
    """Improved CNN for jaundice detection using skin images with calibration"""
    
    def __init__(self, use_calibration=True, use_augmentation=True):
        self.image_processor = create_cnn_processor(
            use_calibration=use_calibration, 
            use_augmentation=use_augmentation
        )
        self.model = self._build_improved_model()
        
    def _build_improved_model(self):
        model = keras.Sequential([
            layers.Input(shape=(IMG_SIZE_CNN[0], IMG_SIZE_CNN[1], 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.1),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.Conv2D(128, 3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(),
            layers.Dropout(0.2),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.4),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE_CNN),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_dataset(self, csv_path, images_dir):
        df = pd.read_csv(csv_path)
        images, labels = [], []
        print(f"🔍 Loading {len(df)} images with CNN preprocessing ({IMG_SIZE_CNN[0]}x{IMG_SIZE_CNN[1]})...")

        for idx, row in df.iterrows():
            if idx % 500 == 0: print(f"Processed {idx}/{len(df)} images")
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                image = self.image_processor.process_image(image_path, apply_augmentation=False) # Augmentation during training only
                images.append(image)
                labels.append(row['jaundiced'])
        
        print(f"✅ Dataset loaded. Calibration: {self.image_processor.use_calibration}")
        return np.array(images), np.array(labels)
    
    def train_model(self, X_train, y_train, X_val, y_val):
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, min_delta=0.001),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.3, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', verbose=1, mode='max')
        ]

        print("🧠 Training Calibrated CNN Model...")
        X_train_aug, y_train_aug = X_train, y_train
        if self.image_processor.use_augmentation:
            print("📈 Using runtime data augmentation")
            augmented_images, augmented_labels = [], []
            for i in range(len(X_train)):
                augmented_images.append(X_train[i])
                augmented_labels.append(y_train[i])
                augmented_image = self.image_processor.augmentation(tf.expand_dims(X_train[i], 0), training=True)[0].numpy()
                augmented_images.append(augmented_image)
                augmented_labels.append(y_train[i])
            X_train_aug, y_train_aug = np.array(augmented_images), np.array(augmented_labels)
            print(f"📊 Training set expanded: {len(X_train)} → {len(X_train_aug)} samples")

        history = self.model.fit(
            X_train_aug, y_train_aug, validation_data=(X_val, y_val),
            epochs=EPOCHS_CNN, batch_size=BATCH_SIZE, callbacks=callbacks,
            class_weight=class_weights_dict, verbose=1
        )
        return history

    def evaluate_model(self, X_test, y_test):
        print("📊 Evaluating Calibrated CNN Model...")
        predictions = self.model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Calibrated CNN Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Using calibration: {self.image_processor.use_calibration}")
        print(f"Using augmentation: {self.image_processor.use_augmentation}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Jaundice', 'Jaundice']))
        return accuracy
    
    def predict_proba(self, X_image_processed):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load weights first.")
        return self.model.predict(X_image_processed)

    def visualize_preprocessing_sample(self, csv_path, images_dir, num_samples=1):
        df = pd.read_csv(csv_path)
        sample_df = df.sample(n=num_samples, random_state=42)
        for _, row in sample_df.iterrows():
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                print(f"Visualizing: {row['image_idx']} (Jaundiced: {row['jaundiced']})")
                fig = self.image_processor.visualize_calibration(image_path)
                if fig:
                    import matplotlib.pyplot as plt
                    plt.show()
                    plt.close(fig) # Close the figure to free memory

def test_cnn_model():
    print("🧠 Testing Calibrated CNN Jaundice Detector")
    print("=" * 60)
    detector_calibrated = SimpleCNNJaundiceDetector(use_calibration=True, use_augmentation=True)
    print(f"Model created with {detector_calibrated.model.count_params():,} parameters")
    # detector_calibrated.visualize_preprocessing_sample(CSV_PATH, IMAGES_DIR, num_samples=1)
    X, y = detector_calibrated.load_dataset(CSV_PATH, IMAGES_DIR)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"\nDataset Summary: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")
    detector_calibrated.train_model(X_train, y_train, X_val, y_val)
    detector_calibrated.model.load_weights(MODEL_PATH)
    accuracy_calibrated = detector_calibrated.evaluate_model(X_test, y_test)
    print(f"\n🏆 Calibrated CNN Accuracy: {accuracy_calibrated:.4f}")
    return detector_calibrated, accuracy_calibrated

# Functions for comparing preprocessing (can be moved or kept here for direct testing)
# def compare_preprocessing_methods(): ... # Removed for brevity in this step

if __name__ == "__main__":
    test_cnn_model()

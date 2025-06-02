"""
EfficientNetB0 Model for Jaundice Detection with Two-Stage Training
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import create_efficientnet_processor # Uses 224x224 by default

# === Configuration (Embedded) ===
CSV_PATH = "ml_model/jaundice_dataset/chd_jaundice_published_2.csv"
IMAGES_DIR = Path("D:/CS Project/ML pro/NeoJaundice/NeoJaundice/images") # Absolute path
MODEL_PATH_HEAD_ONLY = "../best_efficientnet_calibrated_head_only.keras"
MODEL_PATH_FINE_TUNED = "../best_efficientnet_calibrated.keras"

# Hyperparameters
IMG_SIZE_EFFICIENTNET = (224, 224) # Explicitly defined
EPOCHS_EFFICIENTNET_HEAD = 10
EPOCHS_EFFICIENTNET_FINETUNE = 20 
BATCH_SIZE = 32 # Shared batch size
LEARNING_RATE_EFFICIENTNET_HEAD = 1e-3
LEARNING_RATE_EFFICIENTNET_FINETUNE = 5e-5

class EfficientNetJaundiceDetector:
    def __init__(self, use_calibration=True, use_augmentation=True):
        self.image_processor = create_efficientnet_processor(
            use_calibration=use_calibration,
            use_augmentation=use_augmentation
        )
        self.model = None # Will be built/loaded
        self.use_calibration = use_calibration
        self.use_augmentation = use_augmentation

    def _build_model(self, num_classes=1, learning_rate=1e-3, trainable_base_layers=0):
        base_model = EfficientNetB0(
            include_top=False, 
            weights='imagenet', 
            input_shape=(IMG_SIZE_EFFICIENTNET[0], IMG_SIZE_EFFICIENTNET[1], 3)
        )

        if trainable_base_layers == 0:
            base_model.trainable = False
        elif trainable_base_layers > 0:
            base_model.trainable = True
            # Freeze all layers except the last `trainable_base_layers`
            for layer in base_model.layers[:-trainable_base_layers]:
                layer.trainable = False
            print(f"Unfreezing last {trainable_base_layers} layers of EfficientNetB0.")
        else: # trainable_base_layers == -1 means unfreeze all
            base_model.trainable = True 
            print("Unfreezing all layers of EfficientNetB0.")

        inputs = keras.Input(shape=(IMG_SIZE_EFFICIENTNET[0], IMG_SIZE_EFFICIENTNET[1], 3))
        x = base_model(inputs, training=False if trainable_base_layers == 0 else True)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x) # Increased dropout
        x = layers.Dense(64, activation='relu')(x) # Smaller dense layer
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x) # More dropout
        outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)
        
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy' if num_classes == 1 else 'categorical_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_dataset(self, csv_path, images_dir):
        df = pd.read_csv(csv_path)
        images, labels = [], []
        print(f"🔍 Loading {len(df)} images with EfficientNet preprocessing ({IMG_SIZE_EFFICIENTNET[0]}x{IMG_SIZE_EFFICIENTNET[1]})...")

        for idx, row in df.iterrows():
            if idx % 500 == 0: print(f"Processed {idx}/{len(df)} images")
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                image = self.image_processor.process_image(image_path, apply_augmentation=False)
                images.append(image)
                labels.append(row['jaundiced'])
        
        print(f"✅ Dataset loaded. Calibration: {self.use_calibration}")
        return np.array(images), np.array(labels)

    def _train_phase(
        self, X_train, y_train, X_val, y_val, 
        epochs, learning_rate, model_save_path,
        initial_weights_path=None, trainable_base_layers=0,
        use_augmentation_for_phase=False
    ):
        self.model = self._build_model(
            learning_rate=learning_rate,
            trainable_base_layers=trainable_base_layers
        )
        if initial_weights_path and Path(initial_weights_path).exists():
            print(f"💾 Loading initial weights from: {initial_weights_path}")
            self.model.load_weights(initial_weights_path)
        
        class_weights_val = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights_val))

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, min_delta=0.001),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.2, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', verbose=1, mode='max')
        ]
        
        X_train_phase, y_train_phase = X_train, y_train
        if use_augmentation_for_phase and self.image_processor.use_augmentation:
            print("📈 Using runtime data augmentation for this training phase.")
            augmented_images, augmented_labels = [], []
            for i in range(len(X_train)):
                augmented_images.append(X_train[i])
                augmented_labels.append(y_train[i])
                augmented_image = self.image_processor.augmentation(tf.expand_dims(X_train[i], 0), training=True)[0].numpy()
                augmented_images.append(augmented_image)
                augmented_labels.append(y_train[i])
            X_train_phase, y_train_phase = np.array(augmented_images), np.array(augmented_labels)
            print(f"📊 Training set expanded: {len(X_train)} → {len(X_train_phase)} samples")

        history = self.model.fit(
            X_train_phase, y_train_phase, validation_data=(X_val, y_val),
            epochs=epochs, batch_size=BATCH_SIZE, callbacks=callbacks,
            class_weight=class_weights_dict, verbose=1
        )
        return history

    def train_two_stages(self, X_train, y_train, X_val, y_val):
        print("\n--- Stage 1: Training Head Only ---")
        history_head = self._train_phase(
            X_train, y_train, X_val, y_val,
            epochs=EPOCHS_EFFICIENTNET_HEAD,
            learning_rate=LEARNING_RATE_EFFICIENTNET_HEAD,
            model_save_path=MODEL_PATH_HEAD_ONLY,
            trainable_base_layers=0, # Base frozen
            use_augmentation_for_phase=False # No augmentation for head training
        )

        print("\n--- Stage 2: Fine-tuning --- ")
        history_finetune = self._train_phase(
            X_train, y_train, X_val, y_val,
            epochs=EPOCHS_EFFICIENTNET_FINETUNE,
            learning_rate=LEARNING_RATE_EFFICIENTNET_FINETUNE,
            model_save_path=MODEL_PATH_FINE_TUNED,
            initial_weights_path=MODEL_PATH_HEAD_ONLY, # Start from best head weights
            trainable_base_layers=60, # Unfreeze last 60 layers
            use_augmentation_for_phase=self.use_augmentation # Use augmentation if enabled for the detector
        )
        # Load the best fine-tuned model for subsequent evaluation
        if Path(MODEL_PATH_FINE_TUNED).exists():
            self.model.load_weights(MODEL_PATH_FINE_TUNED)
        return history_head, history_finetune

    def evaluate_model(self, X_test, y_test, model_path_to_load=None):
        if model_path_to_load and Path(model_path_to_load).exists():
            print(f"💾 Loading weights for evaluation from: {model_path_to_load}")
            self.model.load_weights(model_path_to_load)
        elif self.model is None or not self.model.weights: # Check if model has weights loaded
            if Path(MODEL_PATH_FINE_TUNED).exists():
                print(f"💾 No specific model path provided, loading best fine-tuned model: {MODEL_PATH_FINE_TUNED}")
                # Rebuild is necessary if model object DNE or is different arch
                self.model = self._build_model(learning_rate=LEARNING_RATE_EFFICIENTNET_FINETUNE, trainable_base_layers=60) 
                self.model.load_weights(MODEL_PATH_FINE_TUNED)
            else:
                raise ValueError("Model not trained or loaded, and no fine-tuned model found.")

        print("📊 Evaluating EfficientNet Model...")
        predictions = self.model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 EfficientNet Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Using calibration: {self.use_calibration}")
        print(f"Using augmentation: {self.use_augmentation}")
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
                    plt.close(fig)

def test_efficientnet_model():
    print("🧠 Testing EfficientNet Jaundice Detector (Two-Stage Training)")
    print("=" * 70)
    detector = EfficientNetJaundiceDetector(use_calibration=True, use_augmentation=True)
    print(f"Image Processor: Calibration={detector.use_calibration}, Augmentation={detector.use_augmentation}")
    
    # detector.visualize_preprocessing_sample(CSV_PATH, IMAGES_DIR, num_samples=1)
    
    X, y = detector.load_dataset(CSV_PATH, IMAGES_DIR)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"\nDataset Summary: Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")

    detector.train_two_stages(X_train, y_train, X_val, y_val)
    accuracy = detector.evaluate_model(X_test, y_test, model_path_to_load=MODEL_PATH_FINE_TUNED)
    print(f"\n🏆 EfficientNet Final Fine-tuned Accuracy: {accuracy:.4f}")
    return detector, accuracy

if __name__ == "__main__":
    test_efficientnet_model()

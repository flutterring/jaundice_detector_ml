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

# Ensure utils is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.preprocessing import create_efficientnet_processor, SimpleTabularProcessor

# === Configuration (Embedded) ===
CSV_PATH = "../jaundice_dataset/chd_jaundice_published_2.csv"
IMAGES_DIR = Path("D:/NU_Courses/semester_6/MI/NeoJaundice/images") # Absolute path
MODEL_PATH_HEAD_ONLY = "../best_efficientnet_tabular_head_only.keras"
MODEL_PATH_FINE_TUNED = "../best_efficientnet_tabular_fine_tuned.keras"

# Image Config
IMG_SIZE_EFFICIENTNET = (224, 224)
USE_CALIBRATION = True
USE_AUGMENTATION = True # For fine-tuning phase of image branch

# Tabular Config
CATEGORICAL_COLS = ['gender'] 
NUMERICAL_COLS = ['gestational_age', 'age(day)', 'weight', 'blood(mg/dL)']
TARGET_COL = 'jaundiced'

# Training Hyperparameters
BATCH_SIZE = 32 
# Stage 1: Head Training for Image Branch
EPOCHS_HEAD = 10
LEARNING_RATE_HEAD = 1e-3
# Stage 2: Fine-tuning for Image Branch + Combined Training
EPOCHS_FINETUNE = 20
LEARNING_RATE_FINETUNE = 5e-5 # For EfficientNet fine-tuning
LEARNING_RATE_COMBINED = 1e-4 # For the combined model after image branch is fine-tuned

class MultiInputEfficientNetTabularDetector:
    def __init__(self):
        self.image_processor = create_efficientnet_processor(use_calibration=USE_CALIBRATION, use_augmentation=USE_AUGMENTATION)
        self.tabular_processor = SimpleTabularProcessor()
        self.model = None
        self.tabular_input_shape = None

    def _build_model(self, learning_rate, trainable_base_layers=0):
        if self.tabular_input_shape is None:
            raise ValueError("Tabular input shape must be set before building model.")

        # Image Branch (EfficientNetB0)
        image_input = layers.Input(shape=(IMG_SIZE_EFFICIENTNET[0], IMG_SIZE_EFFICIENTNET[1], 3), name='image_input')
        base_model = EfficientNetB0(
            include_top=False, 
            weights='imagenet', 
            input_shape=(IMG_SIZE_EFFICIENTNET[0], IMG_SIZE_EFFICIENTNET[1], 3)
        )
        if trainable_base_layers == 0:
            base_model.trainable = False
        elif trainable_base_layers > 0:
            base_model.trainable = True
            for layer in base_model.layers[:-trainable_base_layers]:
                layer.trainable = False
        else: # Unfreeze all
             base_model.trainable = True
        
        x_img = base_model(image_input, training=(False if trainable_base_layers == 0 else True))
        x_img = layers.GlobalAveragePooling2D(name='image_features')(x_img)
        
        # Tabular Branch (MLP)
        tabular_input = layers.Input(shape=(self.tabular_input_shape,), name='tabular_input')
        x_tab = layers.Dense(64, activation='relu')(tabular_input)
        x_tab = layers.BatchNormalization()(x_tab)
        x_tab = layers.Dropout(0.3)(x_tab)
        x_tab = layers.Dense(32, activation='relu', name='tabular_features')(x_tab)

        # Concatenate features
        concatenated = layers.concatenate([x_img, x_tab], name='concatenated_features')
        
        # Combined Classification Head
        x = layers.Dropout(0.3)(concatenated)
        x = layers.Dense(64, activation='relu')(x) # Slightly larger head
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=[image_input, tabular_input], outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_and_preprocess_data(self, csv_path, images_dir):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        all_tabular_cols = CATEGORICAL_COLS + NUMERICAL_COLS
        # Basic validation
        if not all(col in df.columns for col in all_tabular_cols + [TARGET_COL, 'image_idx']):
            raise ValueError("Missing one or more required columns in CSV for multi-input EfficientNet.")

        images_processed, tabular_processed, labels = [], [], []
        X_tabular_raw_full = df[all_tabular_cols]
        self.tabular_processor.fit(X_tabular_raw_full, CATEGORICAL_COLS, NUMERICAL_COLS)
        self.tabular_input_shape = self.tabular_processor.transform(X_tabular_raw_full.iloc[[0]]).shape[1]
        print(f"🛠️ Tabular processor fitted. Expected tabular input shape: ({self.tabular_input_shape},)")

        for idx, row in df.iterrows():
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                image = self.image_processor.process_image(image_path, apply_augmentation=False)
                images_processed.append(image)
                tabular_data = self.tabular_processor.transform(pd.DataFrame([row[all_tabular_cols]]))
                tabular_processed.append(tabular_data.flatten())
                labels.append(row[TARGET_COL])
        
        if not images_processed: raise ValueError("No images processed.")
        print(f"✅ Dataset loaded. Images: {len(images_processed)}, Tabular: {len(tabular_processed)}, Labels: {len(labels)}")
        return [np.array(images_processed), np.array(tabular_processed)], np.array(labels)

    def _train_phase(
        self, X_train_inputs, y_train, X_val_inputs, y_val, 
        epochs, learning_rate, model_save_path,
        initial_weights_path=None, trainable_base_layers=0,
        use_augmentation_for_phase=False
    ):
        self.model = self._build_model(learning_rate=learning_rate, trainable_base_layers=trainable_base_layers)
        if initial_weights_path and Path(initial_weights_path).exists():
            print(f"💾 Loading initial weights from: {initial_weights_path}")
            self.model.load_weights(initial_weights_path)
        
        class_weights_val = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights_val))
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=8, restore_best_weights=True, min_delta=0.0005),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.2, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', verbose=1, mode='max')
        ]

        X_train_images, X_train_tabular = X_train_inputs[0], X_train_inputs[1]
        X_train_feed, y_train_feed = [X_train_images, X_train_tabular], y_train

        if use_augmentation_for_phase and self.image_processor.use_augmentation:
            print("📈 Applying runtime data augmentation to image branch for this phase.")
            aug_imgs, aug_tabs, aug_labels = [], [], []
            for i in range(len(X_train_images)):
                aug_imgs.append(X_train_images[i]); aug_tabs.append(X_train_tabular[i]); aug_labels.append(y_train[i])
                aug_img = self.image_processor.augmentation(tf.expand_dims(X_train_images[i],0), training=True)[0].numpy()
                aug_imgs.append(aug_img); aug_tabs.append(X_train_tabular[i]); aug_labels.append(y_train[i])
            X_train_feed = [np.array(aug_imgs), np.array(aug_tabs)]
            y_train_feed = np.array(aug_labels)
            print(f"📊 Training set expanded: {len(X_train_images)} -> {len(y_train_feed)} instances")

        history = self.model.fit(
            X_train_feed, y_train_feed, validation_data=(X_val_inputs, y_val),
            epochs=epochs, batch_size=BATCH_SIZE, callbacks=callbacks,
            class_weight=class_weights_dict, verbose=1
        )
        return history

    def train_two_stages(self, X_train_inputs, y_train, X_val_inputs, y_val):
        print("\n--- Stage 1: Training Head Only (EfficientNet image branch frozen) ---")
        # During head-only, the EfficientNet base is frozen. Augmentation is off.
        history_head = self._train_phase(
            X_train_inputs, y_train, X_val_inputs, y_val,
            epochs=EPOCHS_HEAD, learning_rate=LEARNING_RATE_HEAD,
            model_save_path=MODEL_PATH_HEAD_ONLY,
            trainable_base_layers=0, # EfficientNet Base frozen
            use_augmentation_for_phase=False
        )

        print("\n--- Stage 2: Fine-tuning (Unfreezing some EfficientNet layers, combined model) ---")
        # During fine-tuning, unfreeze some EfficientNet layers. Augmentation for image branch is on.
        history_finetune = self._train_phase(
            X_train_inputs, y_train, X_val_inputs, y_val,
            epochs=EPOCHS_FINETUNE, learning_rate=LEARNING_RATE_FINETUNE, # Lower LR for fine-tuning
            model_save_path=MODEL_PATH_FINE_TUNED,
            initial_weights_path=MODEL_PATH_HEAD_ONLY,
            trainable_base_layers=60, # Unfreeze last 60 layers of EfficientNet
            use_augmentation_for_phase=USE_AUGMENTATION 
        )
        if Path(MODEL_PATH_FINE_TUNED).exists():
            self.model.load_weights(MODEL_PATH_FINE_TUNED)
        return history_head, history_finetune

    def evaluate_model(self, X_test_inputs, y_test, model_path_to_load=None):
        # Logic to load the correct model state for evaluation
        load_path = model_path_to_load if model_path_to_load else MODEL_PATH_FINE_TUNED
        if Path(load_path).exists():
            print(f"💾 Loading weights for evaluation from: {load_path}")
            # Rebuild with potentially different trainable_base_layers if loading head-only vs fine-tuned
            # For simplicity, assume fine-tuned config (60 layers unfrozen) if not head_only explicitly
            trainable_layers_on_load = 0 if load_path == MODEL_PATH_HEAD_ONLY else 60
            if self.model is None or self.model.name != self._build_model(0.001, trainable_layers_on_load).name: # crude check
                 self.model = self._build_model(learning_rate=LEARNING_RATE_FINETUNE, trainable_base_layers=trainable_layers_on_load)
            self.model.load_weights(load_path)
        else:
            raise ValueError(f"Model not trained or weights not found at {load_path}")

        print("📊 Evaluating Multi-Input EfficientNet+Tabular Model...")
        predictions = self.model.predict(X_test_inputs)
        y_pred = (predictions > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Results (Model: {Path(load_path).name}):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Image Calib: {USE_CALIBRATION}, Image Aug (fine-tune): {USE_AUGMENTATION}")
        print(classification_report(y_test, y_pred, target_names=['No Jaundice', 'Jaundice']))
        return accuracy

    def predict_proba(self, X_processed_inputs):
        if self.model is None: raise ValueError("Model not trained/loaded.")
        return self.model.predict(X_processed_inputs)

def test_multi_input_efficientnet_tabular_model():
    print("🧠 Testing Multi-Input EfficientNet+Tabular Jaundice Detector")
    print("=" * 70)
    detector = MultiInputEfficientNetTabularDetector()
    
    [X_images, X_tabular], y = detector.load_and_preprocess_data(CSV_PATH, IMAGES_DIR)
    indices = np.arange(len(y))
    train_indices, temp_indices, y_train, y_temp = train_test_split(indices, y, test_size=0.3, random_state=42, stratify=y)
    val_indices, test_indices, y_val, y_test = train_test_split(temp_indices, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    X_train = [X_images[train_indices], X_tabular[train_indices]]
    X_val = [X_images[val_indices], X_tabular[val_indices]]
    X_test = [X_images[test_indices], X_tabular[test_indices]]

    print(f"\nDataset Summary: Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"X_train shapes: Image - {X_train[0].shape}, Tabular - {X_train[1].shape}")

    detector.train_two_stages(X_train, y_train, X_val, y_val)
    accuracy = detector.evaluate_model(X_test, y_test, model_path_to_load=MODEL_PATH_FINE_TUNED)
    print(f"\n🏆 Multi-Input EfficientNet+Tabular Final Accuracy: {accuracy:.4f}")
    
    # Optional: Evaluate head-only model too
    # accuracy_head = detector.evaluate_model(X_test, y_test, model_path_to_load=MODEL_PATH_HEAD_ONLY)
    # print(f"\n🏆 Multi-Input EfficientNet+Tabular Head-Only Accuracy: {accuracy_head:.4f}")
    return detector, accuracy

if __name__ == "__main__":
    test_multi_input_efficientnet_tabular_model() 
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

# Ensure utils is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils.preprocessing import create_cnn_processor, SimpleTabularProcessor

# === Configuration (Embedded) ===
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "jaundice_dataset/chd_jaundice_published_2.csv"
IMAGES_DIR = Path("D:/CS Project/ML pro/NeoJaundice/NeoJaundice/images") # Absolute path
MODEL_PATH = SCRIPT_DIR.parent / "best_cnn_tabular_model.keras"

# Image Config
IMG_SIZE_CNN = (128, 128)
USE_CALIBRATION = True
USE_AUGMENTATION = True # Augmentation for the image branch

# Tabular Config (using features provided by user)
CATEGORICAL_COLS = ['gender'] 
NUMERICAL_COLS = ['gestational_age', 'age(day)', 'weight', 'blood(mg/dL)']
TARGET_COL = 'jaundiced'

# Training Hyperparameters
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.0005

class MultiInputCNNTabularDetector:
    def __init__(self):
        self.image_processor = create_cnn_processor(use_calibration=USE_CALIBRATION, use_augmentation=USE_AUGMENTATION)
        self.tabular_processor = SimpleTabularProcessor()
        self.model = None
        self.tabular_input_shape = None

    def _build_model(self):
        if self.tabular_input_shape is None:
            raise ValueError("Tabular input shape must be set by loading data before building model.")

        # Image Branch (CNN - adapted from SimpleCNNJaundiceDetector)
        image_input = layers.Input(shape=(IMG_SIZE_CNN[0], IMG_SIZE_CNN[1], 3), name='image_input')
        x_img = layers.Conv2D(16, 3, padding='same', activation='relu')(image_input)
        x_img = layers.BatchNormalization()(x_img)
        x_img = layers.MaxPooling2D()(x_img)
        x_img = layers.Dropout(0.1)(x_img)
        x_img = layers.Conv2D(32, 3, padding='same', activation='relu')(x_img)
        x_img = layers.BatchNormalization()(x_img)
        x_img = layers.MaxPooling2D()(x_img)
        x_img = layers.Dropout(0.1)(x_img)
        x_img = layers.Conv2D(64, 3, padding='same', activation='relu')(x_img)
        x_img = layers.BatchNormalization()(x_img)
        x_img = layers.MaxPooling2D()(x_img)
        x_img = layers.Dropout(0.2)(x_img)
        x_img = layers.Conv2D(128, 3, padding='same', activation='relu')(x_img)
        x_img = layers.BatchNormalization()(x_img)
        x_img = layers.MaxPooling2D()(x_img)
        x_img = layers.Dropout(0.2)(x_img)
        x_img = layers.GlobalAveragePooling2D(name='image_features')(x_img)
        
        # Tabular Branch (MLP - adapted from TabularJaundiceDetector)
        tabular_input = layers.Input(shape=(self.tabular_input_shape,), name='tabular_input')
        x_tab = layers.Dense(64, activation='relu')(tabular_input) # Simplified MLP for tabular
        x_tab = layers.BatchNormalization()(x_tab)
        x_tab = layers.Dropout(0.3)(x_tab)
        x_tab = layers.Dense(32, activation='relu', name='tabular_features')(x_tab)

        # Concatenate features
        concatenated = layers.concatenate([x_img, x_tab], name='concatenated_features')
        
        # Combined Classification Head
        x = layers.Dropout(0.4)(concatenated)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output = layers.Dense(1, activation='sigmoid', name='output')(x)
        
        model = keras.Model(inputs=[image_input, tabular_input], outputs=output)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_and_preprocess_data(self, csv_path, images_dir):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip() # Clean column names

        # Verify all defined columns exist in the DataFrame
        all_tabular_cols = CATEGORICAL_COLS + NUMERICAL_COLS
        missing_tabular = [col for col in all_tabular_cols if col not in df.columns]
        if missing_tabular:
            raise ValueError(f"Missing tabular columns in CSV: {missing_tabular}. Available: {df.columns.tolist()}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target column '{TARGET_COL}' in CSV. Available: {df.columns.tolist()}")
        if 'image_idx' not in df.columns:
             raise ValueError(f"Missing 'image_idx' column in CSV for image paths. Available: {df.columns.tolist()}")

        images_processed, tabular_processed, labels = [], [], []
        print(f"🔍 Loading and preprocessing data for Multi-Input CNN+Tabular model...")

        # Fit tabular processor on the whole tabular dataset subset first
        X_tabular_raw_full = df[all_tabular_cols]
        self.tabular_processor.fit(X_tabular_raw_full, CATEGORICAL_COLS, NUMERICAL_COLS)
        self.tabular_input_shape = self.tabular_processor.transform(X_tabular_raw_full.iloc[[0]], CATEGORICAL_COLS, NUMERICAL_COLS).shape[1] # Get shape from one sample
        print(f"🛠️ Tabular processor fitted. Expected tabular input shape: ({self.tabular_input_shape},)")

        for idx, row in df.iterrows():
            if idx % 500 == 0: print(f"Processed {idx}/{len(df)} data points")
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                # Process image (no augmentation at loading time)
                image = self.image_processor.process_image(image_path, apply_augmentation=False)
                images_processed.append(image)
                
                # Process tabular data for this row
                tabular_row_raw = pd.DataFrame([row[all_tabular_cols]])
                tabular_data = self.tabular_processor.transform(tabular_row_raw, CATEGORICAL_COLS, NUMERICAL_COLS)
                tabular_processed.append(tabular_data.flatten()) # Flatten to 1D array
                
                labels.append(row[TARGET_COL])
        
        if not images_processed: # Check if any images were successfully processed
            raise ValueError("No images were processed. Check image_idx column and image paths.")
            
        print(f"✅ Dataset loaded. Images: {len(images_processed)}, Tabular rows: {len(tabular_processed)}, Labels: {len(labels)}")
        return [np.array(images_processed), np.array(tabular_processed)], np.array(labels)

    def train_model(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.model = self._build_model()

        class_weights_val = class_weight.compute_class_weight(
            class_weight='balanced', classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights_val))

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, min_delta=0.001),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=1e-7),
            keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', verbose=1, mode='max')
        ]

        print("🧠 Training Multi-Input CNN+Tabular Model...")
        
        # Handle data augmentation for the image part if enabled
        # X_train is a list: [images, tabular_data]
        X_train_images = X_train[0]
        X_train_tabular = X_train[1]

        if self.image_processor.use_augmentation:
            print("📈 Applying runtime data augmentation to image branch.")
            augmented_images_train, augmented_tabular_train, augmented_labels_train = [], [], []
            for i in range(len(X_train_images)):
                # Original sample
                augmented_images_train.append(X_train_images[i])
                augmented_tabular_train.append(X_train_tabular[i])
                augmented_labels_train.append(y_train[i])
                
                # Augmented image sample
                aug_img = self.image_processor.augmentation(tf.expand_dims(X_train_images[i], 0), training=True)[0].numpy()
                augmented_images_train.append(aug_img)
                augmented_tabular_train.append(X_train_tabular[i]) # Tabular data remains the same
                augmented_labels_train.append(y_train[i])

            X_train_feed = [np.array(augmented_images_train), np.array(augmented_tabular_train)]
            y_train_feed = np.array(augmented_labels_train)
            print(f"📊 Training set expanded (due to image augmentation): {len(X_train_images)} original pairs -> {len(y_train_feed)} training instances")
        else:
            X_train_feed = X_train
            y_train_feed = y_train
            
        history = self.model.fit(
            X_train_feed, y_train_feed,
            validation_data=(X_val, y_val),
            epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks,
            class_weight=class_weights_dict, verbose=1
        )
        return history

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        print("📊 Evaluating Multi-Input CNN+Tabular Model...")
        predictions = self.model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n🎯 Multi-Input CNN+Tabular Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Using image calibration: {USE_CALIBRATION}")
        print(f"Using image augmentation: {USE_AUGMENTATION}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Jaundice', 'Jaundice']))
        return accuracy

    def predict_proba(self, X_processed_inputs):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load weights first.")
        return self.model.predict(X_processed_inputs)

def test_multi_input_cnn_tabular_model():
    print("🧠 Testing Multi-Input CNN+Tabular Jaundice Detector")
    print("=" * 70)
    detector = MultiInputCNNTabularDetector()
    
    # Load data
    # Returns [images_array, tabular_array], labels_array
    [X_images, X_tabular], y = detector.load_and_preprocess_data(CSV_PATH, IMAGES_DIR)

    # Train/Val/Test Split for multi-input
    # Ensure consistent splitting for both input types
    indices = np.arange(len(y))
    train_indices, temp_indices, y_train, y_temp = train_test_split(
        indices, y, test_size=0.3, random_state=42, stratify=y
    )
    val_indices, test_indices, y_val, y_test = train_test_split(
        temp_indices, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    X_train = [X_images[train_indices], X_tabular[train_indices]]
    X_val = [X_images[val_indices], X_tabular[val_indices]]
    X_test = [X_images[test_indices], X_tabular[test_indices]]

    print(f"\nDataset Summary:")
    print(f"Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")
    print(f"X_train shapes: Image - {X_train[0].shape}, Tabular - {X_train[1].shape}")

    # Train model
    detector.train_model(X_train, y_train, X_val, y_val)
    
    # Load best weights for evaluation
    if Path(MODEL_PATH).exists():
        detector.model.load_weights(MODEL_PATH)
    else:
        print(f"Warning: Model file {MODEL_PATH} not found. Evaluation might use last epoch weights.")

    # Evaluate model
    accuracy = detector.evaluate_model(X_test, y_test)
    print(f"\n🏆 Multi-Input CNN+Tabular Model Accuracy: {accuracy:.4f}")
    return detector, accuracy

if __name__ == "__main__":
    test_multi_input_cnn_tabular_model()
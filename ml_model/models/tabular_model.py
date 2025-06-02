import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.preprocessing import SimpleTabularProcessor

# === Configuration (Embedded) ===
# Adjusted path relative to this model file (models/)
CSV_PATH = "ml_model/jaundice_dataset/chd_jaundice_published_2.csv" 
MODEL_PATH = "../best_tabular_model.keras" # Save in ml_model root

# Tabular features based on user input
CATEGORICAL_COLS = ['gender'] 
NUMERICAL_COLS = ['gestational_age', 'age(day)', 'weight', 'blood(mg/dL)']
TARGET_COL = 'jaundiced'

# Hyperparameters
EPOCHS_TABULAR = 50
BATCH_SIZE = 32
LEARNING_RATE_TABULAR = 1e-3

class TabularJaundiceDetector:
    def __init__(self):
        self.tabular_processor = SimpleTabularProcessor()
        self.model = None
        self.input_shape = None

    def _build_model(self):
        if self.input_shape is None:
            raise ValueError("Input shape must be set by loading data before building model.")
        
        model = keras.Sequential([
            layers.Input(shape=(self.input_shape,)),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(LEARNING_RATE_TABULAR),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        return model

    def load_and_preprocess_data(self, csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Verify all defined columns exist in the DataFrame
        all_feature_cols = CATEGORICAL_COLS + NUMERICAL_COLS
        missing_features = [col for col in all_feature_cols if col not in df.columns]
        if missing_features:
            raise ValueError(f"Missing feature columns in CSV: {missing_features}. Available columns: {df.columns.tolist()}")
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target column '{TARGET_COL}' in CSV. Available columns: {df.columns.tolist()}")

        X_tabular_raw = df[all_feature_cols]
        y = df[TARGET_COL].values

        # Do NOT fit/transform here; just return raw features and labels
        return X_tabular_raw, y

    def train_model(self, X_train, y_train, X_val, y_val):
        if self.model is None:
            self.model = self._build_model()
            
        class_weights = class_weight.compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train),
            y=y_train
        )
        class_weights_dict = dict(enumerate(class_weights))

        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, min_delta=0.001),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, factor=0.3, min_lr=1e-6),
            keras.callbacks.ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
        ]

        print("🧠 Training Tabular Model...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=EPOCHS_TABULAR,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            class_weight=class_weights_dict,
            verbose=1
        )
        return history

    def evaluate_model(self, X_test, y_test):
        if self.model is None:
            raise ValueError("Model not trained or loaded.")
        
        print("📊 Evaluating Tabular Model...")
        predictions = self.model.predict(X_test)
        y_pred = (predictions > 0.5).astype(int).flatten()
        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n🎯 Tabular Model Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Jaundice', 'Jaundice']))
        return accuracy

    def predict_proba(self, X_tabular_processed):
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load weights first.")
        return self.model.predict(X_tabular_processed)

def test_tabular_model():
    print("🧠 Testing Tabular Jaundice Detector")
    print("=" * 60)

    detector = TabularJaundiceDetector()
    X_tabular_raw, y = detector.load_and_preprocess_data(CSV_PATH)

    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X_tabular_raw, y, test_size=0.3, random_state=42, stratify=y)
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Fit processor only on training data, then transform all splits
    detector.tabular_processor.fit(X_train_raw, CATEGORICAL_COLS, NUMERICAL_COLS)
    X_train = detector.tabular_processor.transform(X_train_raw, CATEGORICAL_COLS, NUMERICAL_COLS)
    X_val = detector.tabular_processor.transform(X_val_raw, CATEGORICAL_COLS, NUMERICAL_COLS)
    X_test = detector.tabular_processor.transform(X_test_raw, CATEGORICAL_COLS, NUMERICAL_COLS)
    detector.input_shape = X_train.shape[1]

    print(f"\nDataset Summary:")
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Class distribution (train): {np.unique(y_train, return_counts=True)}")

    detector.train_model(X_train, y_train, X_val, y_val)
    detector.model.load_weights(MODEL_PATH) # Load best weights
    accuracy = detector.evaluate_model(X_test, y_test)
    print(f"\n🏆 Tabular Model Accuracy: {accuracy:.4f}")
    return detector, accuracy

if __name__ == "__main__":
    test_tabular_model()
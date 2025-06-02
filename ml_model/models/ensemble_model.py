import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import sys
import tensorflow as tf # For loading Keras models

# Ensure other models and utils are in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'models')) # if models are in subdir like models/tabular_model.py

from utils.preprocessing import create_cnn_processor, SimpleTabularProcessor # Or your chosen image processor
from cnn_models import SimpleCNNJaundiceDetector # Assuming this is the best image model
from tabular_model import TabularJaundiceDetector

# === Configuration (Embedded for Ensemble) ===
CSV_PATH = "../jaundice_dataset/chd_jaundice_published_2.csv"
IMAGES_DIR = Path("D:/NU_Courses/semester_6/MI/NeoJaundice/images") # Absolute path for image data

# Paths to pre-trained models (User needs to ensure these exist)
BEST_IMAGE_MODEL_PATH = "../best_cnn_calibrated.keras" # Example: Using the CNN model
BEST_TABULAR_MODEL_PATH = "../best_tabular_model.keras"

# Image processing config (must match the pre-trained image model)
IMG_SIZE_FOR_ENSEMBLE = (128, 128) # e.g., CNN
USE_CALIBRATION_FOR_ENSEMBLE = True

# Tabular features (must match the pre-trained tabular model)
CATEGORICAL_COLS_ENSEMBLE = ['gender']
NUMERICAL_COLS_ENSEMBLE = ['gestational_age', 'age(day)', 'weight', 'blood(mg/dL)']
TARGET_COL_ENSEMBLE = 'jaundiced'

class EnsembleJaundiceDetector:
    def __init__(self, image_model_path, tabular_model_path):
        # Image Model Branch
        print(f"Initializing Image Model branch for Ensemble from {image_model_path}...")
        self.image_detector = SimpleCNNJaundiceDetector(use_calibration=USE_CALIBRATION_FOR_ENSEMBLE, use_augmentation=False) # Augmentation not needed for prediction
        if Path(image_model_path).exists():
            self.image_detector.model.load_weights(image_model_path)
            print(f"✅ Image model weights loaded from {image_model_path}")
        else:
            raise FileNotFoundError(f"Image model weights not found at {image_model_path}. Please train and save it first.")
        self.image_processor = self.image_detector.image_processor # Use its processor

        # Tabular Model Branch
        print(f"Initializing Tabular Model branch for Ensemble from {tabular_model_path}...")
        self.tabular_detector = TabularJaundiceDetector()
        # Pre-load dummy data to set input_shape for tabular model before loading weights
        # This is a workaround if input_shape isn't saved with the model or known beforehand
        # A more robust way is to save/load input_shape with the tabular model itself
        num_dummy_features = len(CATEGORICAL_COLS_ENSEMBLE) + len(NUMERICAL_COLS_ENSEMBLE) 
        # Note: The actual number of features after preprocessing (e.g. one-hot encoding) is what's needed.
        # We will call load_and_preprocess_data for the tabular detector to set its input shape correctly before loading weights.
        # This is a bit of a hack for standalone loading; ideally tabular model saves its input_shape or it's derived from its processor config.
        
        # To properly set the tabular_detector.input_shape before loading weights:
        # We need to process a sample. This is a bit clunky here.
        # A better TabularJaundiceDetector would save its processor's state or input_shape.
        print("Temporarily processing a dummy sample for tabular model shape inference...")
        dummy_df_data = {col: [0] for col in NUMERICAL_COLS_ENSEMBLE}
        dummy_df_data.update({col: ['dummy'] for col in CATEGORICAL_COLS_ENSEMBLE})
        dummy_df_for_shape = pd.DataFrame(dummy_df_data)
        _ = self.tabular_detector.tabular_processor.fit_transform(dummy_df_for_shape, CATEGORICAL_COLS_ENSEMBLE, NUMERICAL_COLS_ENSEMBLE)
        self.tabular_detector.input_shape = _.shape[1] # Set the shape
        self.tabular_detector.model = self.tabular_detector._build_model() # Now build with correct shape

        if Path(tabular_model_path).exists():
            self.tabular_detector.model.load_weights(tabular_model_path)
            print(f"✅ Tabular model weights loaded from {tabular_model_path}")
        else:
            raise FileNotFoundError(f"Tabular model weights not found at {tabular_model_path}. Please train and save it first.")
        self.tabular_processor = self.tabular_detector.tabular_processor

    def load_and_preprocess_data_for_ensemble(self, csv_path, images_dir):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        # Validate columns
        all_tab_cols = CATEGORICAL_COLS_ENSEMBLE + NUMERICAL_COLS_ENSEMBLE
        if not all(c in df.columns for c in all_tab_cols + [TARGET_COL_ENSEMBLE, 'image_idx']):
            raise ValueError("Missing required columns in CSV for ensemble prediction.")

        images_input, tabular_input, labels_output = [], [], []
        print(f"🔍 Loading and preprocessing data for Ensemble model...")
        
        # Fit the tabular processor on the full dataset if not already done (though it should be by init)
        # This ensures transformations (like scaling, one-hot encoding) are consistent.
        # self.tabular_processor.fit(df[all_tab_cols], CATEGORICAL_COLS_ENSEMBLE, NUMERICAL_COLS_ENSEMBLE)

        for idx, row in df.iterrows():
            if idx % 500 == 0: print(f"Processed {idx}/{len(df)} data points for ensemble")
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                # Process image
                image = self.image_processor.process_image(image_path, apply_augmentation=False)
                images_input.append(image)
                
                # Process tabular data for this row
                tabular_row_raw = pd.DataFrame([row[all_tab_cols]])
                tabular_data = self.tabular_processor.transform(tabular_row_raw)
                tabular_input.append(tabular_data.flatten()) # Flatten to 1D array
                
                labels_output.append(row[TARGET_COL_ENSEMBLE])
        
        if not images_input: raise ValueError("No images were processed for ensemble.")
        print(f"✅ Ensemble data loaded. Images: {len(images_input)}, Tabular: {len(tabular_input)}, Labels: {len(labels_output)}")
        return [np.array(images_input), np.array(tabular_input)], np.array(labels_output)

    def predict_proba(self, X_inputs_list):
        """ Generates probability predictions using soft voting (averaging). """
        X_images = X_inputs_list[0]
        X_tabular = X_inputs_list[1]

        # Get probabilities from each model
        image_probs = self.image_detector.predict_proba(X_images)
        tabular_probs = self.tabular_detector.predict_proba(X_tabular)
        
        # Soft voting: Average the probabilities
        # Ensure they are 2D (N_samples, 1) before averaging if necessary
        avg_probs = (image_probs + tabular_probs) / 2.0
        return avg_probs

    def predict(self, X_inputs_list):
        """ Generates class predictions (0 or 1). """
        avg_probs = self.predict_proba(X_inputs_list)
        return (avg_probs > 0.5).astype(int).flatten()

    def evaluate_ensemble(self, X_test_list, y_test):
        print("📊 Evaluating Ensemble Model...")
        y_pred = self.predict(X_test_list)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n🎯 Ensemble Model Results (Soft Voting):")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Image Model: {self.image_detector.model.name} from {BEST_IMAGE_MODEL_PATH}")
        print(f"Tabular Model: {self.tabular_detector.model.name} from {BEST_TABULAR_MODEL_PATH}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Jaundice', 'Jaundice']))
        return accuracy

def test_ensemble_model():
    print("🧠 Testing Ensemble Jaundice Detector")
    print("=" * 70)
    
    if not Path(BEST_IMAGE_MODEL_PATH).exists() or not Path(BEST_TABULAR_MODEL_PATH).exists():
        print(f"🛑 Error: Pre-trained model weights not found.")
        print(f"Ensure '{BEST_IMAGE_MODEL_PATH}' and '{BEST_TABULAR_MODEL_PATH}' exist.")
        print("Please train the individual CNN and Tabular models first.")
        return

    ensemble_detector = EnsembleJaundiceDetector(
        image_model_path=BEST_IMAGE_MODEL_PATH, 
        tabular_model_path=BEST_TABULAR_MODEL_PATH
    )
    
    # Load data for ensemble
    [X_images, X_tabular], y = ensemble_detector.load_and_preprocess_data_for_ensemble(CSV_PATH, IMAGES_DIR)

    # Split for testing (no training for ensemble, just evaluation)
    # Using a simple split here as an example. Ensure data is consistent with how models were trained/tested.
    indices = np.arange(len(y))
    _, test_indices, _, y_test_ensemble = train_test_split(
        indices, y, test_size=0.2, random_state=42, stratify=y # Using 20% for test as an example
    )

    X_test_ensemble = [X_images[test_indices], X_tabular[test_indices]]

    print(f"\nEnsemble Test Set Summary: {len(y_test_ensemble)} samples")

    # Evaluate ensemble
    accuracy = ensemble_detector.evaluate_ensemble(X_test_ensemble, y_test_ensemble)
    print(f"\n🏆 Ensemble Model Accuracy: {accuracy:.4f}")
    return ensemble_detector, accuracy

if __name__ == "__main__":
    test_ensemble_model() 
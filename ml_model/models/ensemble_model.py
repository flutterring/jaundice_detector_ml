import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import sys
import tensorflow as tf

# Ensure other models and utils are in path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'models'))

from utils.preprocessing import create_cnn_processor, SimpleTabularProcessor
from cnn_model import SimpleCNNJaundiceDetector
from tabular_model import TabularJaundiceDetector

# === Configuration ===
SCRIPT_DIR = Path(__file__).resolve().parent
CSV_PATH = SCRIPT_DIR.parent / "jaundice_dataset/chd_jaundice_published_2.csv"
IMAGES_DIR = Path("D:/CS Project/ML pro/NeoJaundice/NeoJaundice/images")

# Model paths
BEST_IMAGE_MODEL_PATH = "D:/Flutter/jaundice_detector_ml/ml_model/best_cnn_calibrated.keras"
BEST_TABULAR_MODEL_PATH = "D:/Flutter/jaundice_detector_ml/ml_model/best_cnn_tabular_model.keras"

# Configuration
IMG_SIZE_FOR_ENSEMBLE = (128, 128)
USE_CALIBRATION_FOR_ENSEMBLE = True

CATEGORICAL_COLS_ENSEMBLE = ['gender']
NUMERICAL_COLS_ENSEMBLE = ['gestational_age', 'age(day)', 'weight', 'blood(mg/dL)']
TARGET_COL_ENSEMBLE = 'jaundiced'

class EnsembleJaundiceDetector:
    def __init__(self, image_model_path, tabular_model_path):
        print("🔧 Initializing Ensemble Jaundice Detector...")
        
        # Initialize Image Model
        print(f"Loading Image Model from {image_model_path}...")
        self.image_detector = SimpleCNNJaundiceDetector(
            use_calibration=USE_CALIBRATION_FOR_ENSEMBLE, 
            use_augmentation=False
        )
        
        if Path(image_model_path).exists():
            try:
                # Load the entire model instead of just weights
                self.image_detector.model = tf.keras.models.load_model(image_model_path)
                print(f"✅ Image model loaded successfully")
            except Exception as e:
                print(f"⚠️  Failed to load full image model, trying weights only: {e}")
                self.image_detector.model.load_weights(image_model_path)
                print(f"✅ Image model weights loaded")
        else:
            raise FileNotFoundError(f"Image model not found at {image_model_path}")
        
        self.image_processor = self.image_detector.image_processor

        # Initialize Tabular Model with proper preprocessing
        print(f"Setting up Tabular Model from {tabular_model_path}...")
        self._setup_tabular_model(tabular_model_path)

    def _setup_tabular_model(self, tabular_model_path):
        """Setup tabular model with proper preprocessing to match saved weights"""
        # Load and prepare data for preprocessing
        tabular_csv_df = pd.read_csv(CSV_PATH)
        tabular_csv_df.columns = tabular_csv_df.columns.str.strip()
        
        all_tab_cols = CATEGORICAL_COLS_ENSEMBLE + NUMERICAL_COLS_ENSEMBLE
        
        # Initialize tabular processor and fit it
        self.tabular_processor = SimpleTabularProcessor()
        self.tabular_processor.fit(
            tabular_csv_df[all_tab_cols], 
            CATEGORICAL_COLS_ENSEMBLE, 
            NUMERICAL_COLS_ENSEMBLE
        )
        
        # Get the correct input shape after preprocessing
        sample_processed = self.tabular_processor.transform(
            tabular_csv_df[all_tab_cols].iloc[[0]], 
            CATEGORICAL_COLS_ENSEMBLE, 
            NUMERICAL_COLS_ENSEMBLE
        )
        input_shape = sample_processed.shape[1]
        print(f"📐 Tabular input shape determined: {input_shape}")
        
        # Try to load the saved model first
        if Path(tabular_model_path).exists():
            try:
                # Option 1: Load the entire model
                print("🔄 Attempting to load complete tabular model...")
                self.tabular_model = tf.keras.models.load_model(tabular_model_path)
                print(f"✅ Complete tabular model loaded successfully")
                return
            except Exception as e:
                print(f"⚠️  Failed to load complete model: {e}")
                print("🔄 Attempting to create model and load weights...")
        
        # Option 2: Create model and load weights
        try:
            # Create a new tabular detector
            self.tabular_detector = TabularJaundiceDetector()
            self.tabular_detector.input_shape = input_shape
            self.tabular_detector.tabular_processor = self.tabular_processor
            
            # Build the model
            self.tabular_detector.model = self.tabular_detector._build_model()
            
            # Try to load weights
            self.tabular_detector.model.load_weights(tabular_model_path)
            self.tabular_model = self.tabular_detector.model
            print(f"✅ Tabular model weights loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading tabular model: {e}")
            print("🔧 Creating new model with detected input shape...")
            
            # Create a simple model as fallback
            self.tabular_model = self._create_fallback_tabular_model(input_shape)
            print("⚠️  Using fallback model - you may need to retrain")

    def _create_fallback_tabular_model(self, input_shape):
        """Create a simple fallback tabular model"""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_shape,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def load_and_preprocess_data_for_ensemble(self, csv_path, images_dir):
        """Load and preprocess data for ensemble prediction"""
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        
        all_tab_cols = CATEGORICAL_COLS_ENSEMBLE + NUMERICAL_COLS_ENSEMBLE
        required_cols = all_tab_cols + [TARGET_COL_ENSEMBLE, 'image_idx']
        
        if not all(c in df.columns for c in required_cols):
            missing = [c for c in required_cols if c not in df.columns]
            raise ValueError(f"Missing required columns: {missing}")

        images_input, tabular_input, labels_output = [], [], []
        print(f"🔍 Loading and preprocessing data for Ensemble model...")
        
        for idx, row in df.iterrows():
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(df)} data points")
            
            image_path = images_dir / row['image_idx']
            if image_path.exists():
                try:
                    # Process image
                    image = self.image_processor.process_image(image_path, apply_augmentation=False)
                    images_input.append(image)
                    
                    # Process tabular data
                    tabular_row_raw = pd.DataFrame([row[all_tab_cols]])
                    tabular_data = self.tabular_processor.transform(
                        tabular_row_raw, 
                        CATEGORICAL_COLS_ENSEMBLE, 
                        NUMERICAL_COLS_ENSEMBLE
                    )
                    tabular_input.append(tabular_data.flatten())
                    
                    labels_output.append(row[TARGET_COL_ENSEMBLE])
                    
                except Exception as e:
                    print(f"⚠️  Error processing row {idx}: {e}")
                    continue
        
        if not images_input:
            raise ValueError("No valid data was processed for ensemble")
        
        print(f"✅ Ensemble data loaded - Samples: {len(images_input)}")
        return [np.array(images_input), np.array(tabular_input)], np.array(labels_output)

    def predict_proba(self, X_inputs_list):
        """Generate probability predictions using soft voting"""
        X_images, X_tabular = X_inputs_list
        
        # Get probabilities from each model
        try:
            image_probs = self.image_detector.predict_proba(X_images)
            if hasattr(self, 'tabular_detector'):
                tabular_probs = self.tabular_detector.predict_proba(X_tabular)
            else:
                tabular_probs = self.tabular_model.predict(X_tabular, verbose=0)
            
            # Ensure probabilities are in the right shape
            if image_probs.ndim == 1:
                image_probs = image_probs.reshape(-1, 1)
            if tabular_probs.ndim == 1:
                tabular_probs = tabular_probs.reshape(-1, 1)
            
            # Soft voting: average the probabilities
            avg_probs = (image_probs + tabular_probs) / 2.0
            return avg_probs.flatten()
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            # Fallback to image model only
            return self.image_detector.predict_proba(X_images)

    def predict(self, X_inputs_list):
        """Generate class predictions"""
        probs = self.predict_proba(X_inputs_list)
        return (probs > 0.5).astype(int)

    def evaluate_ensemble(self, X_test_list, y_test):
        """Evaluate the ensemble model"""
        print("📊 Evaluating Ensemble Model...")
        
        try:
            y_pred = self.predict(X_test_list)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"\n🎯 Ensemble Model Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Test samples: {len(y_test)}")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Jaundice', 'Jaundice']))
            
            return accuracy
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            return 0.0

def test_ensemble_model():
    """Test the ensemble model"""
    print("🧠 Testing Ensemble Jaundice Detector")
    print("=" * 70)
    
    # Check if model files exist
    if not Path(BEST_IMAGE_MODEL_PATH).exists():
        print(f"❌ Image model not found: {BEST_IMAGE_MODEL_PATH}")
        return None
    
    if not Path(BEST_TABULAR_MODEL_PATH).exists():
        print(f"❌ Tabular model not found: {BEST_TABULAR_MODEL_PATH}")
        return None
    
    try:
        # Initialize ensemble
        ensemble_detector = EnsembleJaundiceDetector(
            image_model_path=BEST_IMAGE_MODEL_PATH,
            tabular_model_path=BEST_TABULAR_MODEL_PATH
        )
        
        # Load and preprocess data
        print("\n🔄 Loading data for ensemble evaluation...")
        [X_images, X_tabular], y = ensemble_detector.load_and_preprocess_data_for_ensemble(
            CSV_PATH, IMAGES_DIR
        )
        
        # Split data for testing
        indices = np.arange(len(y))
        _, test_indices, _, y_test = train_test_split(
            indices, y, test_size=0.2, random_state=42, stratify=y
        )
        
        X_test = [X_images[test_indices], X_tabular[test_indices]]
        
        print(f"\n📈 Test set: {len(y_test)} samples")
        print(f"Class distribution: {np.bincount(y_test)}")
        
        # Evaluate ensemble
        accuracy = ensemble_detector.evaluate_ensemble(X_test, y_test)
        
        print(f"\n🏆 Final Ensemble Accuracy: {accuracy:.4f}")
        return ensemble_detector, accuracy
        
    except Exception as e:
        print(f"❌ Error in ensemble testing: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = test_ensemble_model()
    if result:
        print("✅ Ensemble testing completed successfully!")
    else:
        print("❌ Ensemble testing failed!")
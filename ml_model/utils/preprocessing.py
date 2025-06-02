"""
Enhanced Preprocessing with Color Calibration and Augmentation for Jaundice Detection
"""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import joblib
import tensorflow as tf
from tensorflow.keras import layers

class ColorCalibratedImageProcessor:
    """Enhanced image processor with color calibration and augmentation"""
    
    def __init__(self, target_size=(128, 128), use_calibration=True, use_augmentation=True):
        self.target_size = target_size
        self.use_calibration = use_calibration
        self.use_augmentation = use_augmentation
        
        # Define reference colors for calibration card (typical values)
        self.reference_colors = {
            'white': [255, 255, 255],
            'black': [0, 0, 0],
            'red': [255, 0, 0],
            'green': [0, 255, 0],
            'blue': [0, 0, 255],
            'gray': [128, 128, 128]
        }
        
        # Data augmentation pipeline - reduced intensity for medical imaging
        if self.use_augmentation:
            self.augmentation = tf.keras.Sequential([
                layers.RandomFlip("horizontal"),
                layers.RandomRotation(0.05),      # Reduced from 0.1
                layers.RandomZoom(0.05),          # Reduced from 0.1  
                layers.RandomBrightness(0.05),    # Reduced from 0.1
                layers.RandomContrast(0.05),      # Reduced from 0.1
            ])
    
    def detect_calibration_card(self, image):
        """Detect and extract calibration card patches"""
        try:
            # Convert to grayscale for contour detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Apply adaptive threshold
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                         cv2.THRESH_BINARY, 11, 2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by area and aspect ratio
            calibration_patches = []
            min_area = 500  # Minimum area for calibration patch
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > min_area:
                    # Approximate contour to rectangle
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    
                    if len(approx) == 4:  # Rectangle
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Check if it's roughly square (calibration patches are usually square)
                        if 0.8 <= aspect_ratio <= 1.2:
                            # Extract patch
                            patch = image[y:y+h, x:x+w]
                            avg_color = np.mean(patch.reshape(-1, 3), axis=0)
                            calibration_patches.append({
                                'bbox': (x, y, w, h),
                                'color': avg_color,
                                'patch': patch
                            })
            
            return calibration_patches
        
        except Exception as e:
            print(f"Error detecting calibration card: {e}")
            return []

    def detect_skin_color_region(self, image):
        """Detect skin region using color-based segmentation"""
        try:
            # Convert to HSV for better skin detection
            hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            
            # Define skin color range in HSV (works for various skin tones)
            # Lower bound: [0, 20, 70] - covers lighter skin
            # Upper bound: [20, 255, 255] - covers darker skin
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            # Create mask for skin colors
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Additional skin detection in YCrCb color space (more robust)
            ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            # Skin detection in YCrCb: Cr=[133,173], Cb=[77,127]
            lower_skin_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            upper_skin_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            skin_mask_ycrcb = cv2.inRange(ycrcb, lower_skin_ycrcb, upper_skin_ycrcb)
            
            # Combine both masks
            combined_skin_mask = cv2.bitwise_or(skin_mask, skin_mask_ycrcb)
            
            # Apply morphological operations to clean up the mask
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            combined_skin_mask = cv2.morphologyEx(combined_skin_mask, cv2.MORPH_OPEN, kernel)
            combined_skin_mask = cv2.morphologyEx(combined_skin_mask, cv2.MORPH_CLOSE, kernel)
            
            # Fill holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
            combined_skin_mask = cv2.morphologyEx(combined_skin_mask, cv2.MORPH_CLOSE, kernel)
            
            return combined_skin_mask
            
        except Exception as e:
            print(f"Error in skin color detection: {e}")
            return None

    def find_skin_region(self, image, calibration_patches):
        """Find the skin region avoiding calibration patches"""
        h, w = image.shape[:2]
        
        # Method 1: Use skin color detection
        skin_mask = self.detect_skin_color_region(image)
        
        if skin_mask is not None:
            # Create exclusion mask for calibration patches
            exclusion_mask = np.ones((h, w), dtype=np.uint8) * 255
            
            for patch in calibration_patches:
                x, y, pw, ph = patch['bbox']
                # Create exclusion zone around each patch
                expand_factor = 0.5  # 50% expansion
                expand_x = int(pw * expand_factor)
                expand_y = int(ph * expand_factor)
                
                x_start = max(0, x - expand_x)
                y_start = max(0, y - expand_y)
                x_end = min(w, x + pw + expand_x)
                y_end = min(h, y + ph + expand_y)
                
                exclusion_mask[y_start:y_end, x_start:x_end] = 0
            
            # Combine skin mask with exclusion mask
            final_mask = cv2.bitwise_and(skin_mask, exclusion_mask)
            
            # Find contours in the final mask
            contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest skin region
                largest_skin_contour = max(contours, key=cv2.contourArea)
                skin_area = cv2.contourArea(largest_skin_contour)
                
                # Only use if the skin area is substantial
                if skin_area > (h * w * 0.03):  # At least 3% of image
                    x, y, w_rect, h_rect = cv2.boundingRect(largest_skin_contour)
                    
                    # Add some padding but keep it reasonable
                    padding = max(5, min(w_rect, h_rect) // 15)
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w_rect = min(image.shape[1] - x, w_rect + 2*padding)
                    h_rect = min(image.shape[0] - y, h_rect + 2*padding)
                    
                    skin_region = image[y:y+h_rect, x:x+w_rect]
                    
                    # Ensure reasonable size
                    min_size = min(h, w) // 8
                    if w_rect >= min_size and h_rect >= min_size:
                        return self.make_square_crop(skin_region)
        
        # Fallback: Simple center crop
        center_y, center_x = h // 2, w // 2
        crop_size = min(h, w) // 4
        
        y1 = max(0, center_y - crop_size)
        y2 = min(h, center_y + crop_size)
        x1 = max(0, center_x - crop_size)
        x2 = min(w, center_x + crop_size)
        
        return image[y1:y2, x1:x2]
    
    def make_square_crop(self, region):
        """Make the region square and clean"""
        h, w = region.shape[:2]
        
        if abs(w - h) < min(w, h) * 0.1:
            # Already roughly square
            return region
        
        # Make it square by cropping to smaller dimension
        size = min(w, h)
        center_x = w // 2
        center_y = h // 2
        
        x_start = max(0, center_x - size // 2)
        y_start = max(0, center_y - size // 2)
        x_end = min(w, x_start + size)
        y_end = min(h, y_start + size)
        
        return region[y_start:y_end, x_start:x_end]
    
    def color_correct_image(self, image, calibration_patches):
        """Apply color correction based on calibration patches"""
        try:
            if not calibration_patches:
                return image
            
            # Find white and black patches for white balance
            white_patch = None
            black_patch = None
            
            for patch in calibration_patches:
                avg_color = patch['color']
                brightness = np.mean(avg_color)
                
                if brightness > 200:  # Likely white patch
                    white_patch = avg_color
                elif brightness < 50:  # Likely black patch
                    black_patch = avg_color
            
            # Apply white balance correction
            corrected = image.astype(np.float32)
            
            if white_patch is not None:
                # Normalize based on white patch
                white_target = np.array([255, 255, 255])
                white_current = white_patch
                
                # Calculate correction factors
                correction_factors = white_target / (white_current + 1e-8)
                
                # Apply correction
                for i in range(3):
                    corrected[:, :, i] *= correction_factors[i]
            
            # Apply gamma correction for better contrast
            corrected = np.power(corrected / 255.0, 0.8) * 255.0
            
            # Clip values to valid range
            corrected = np.clip(corrected, 0, 255).astype(np.uint8)
            
            return corrected
        
        except Exception as e:
            print(f"Error in color correction: {e}")
            return image
    
    def process_image(self, image_path, apply_augmentation=False):
        """
        Complete image processing pipeline with calibration and augmentation
        """
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            if self.use_calibration:
                # Detect calibration patches
                calibration_patches = self.detect_calibration_card(image)
                
                # Apply color correction
                image = self.color_correct_image(image, calibration_patches)
                
                # Extract skin region using advanced detection methods
                skin_region = self.find_skin_region(image, calibration_patches)
            else:
                # Fallback: simple center crop
                h, w = image.shape[:2]
                center_y, center_x = h // 2, w // 2
                crop_size = min(h, w) // 3
                
                y1 = max(0, center_y - crop_size)
                y2 = min(h, center_y + crop_size)
                x1 = max(0, center_x - crop_size)
                x2 = min(w, center_x + crop_size)
                
                skin_region = image[y1:y2, x1:x2]
            
            # Resize to target size
            processed_image = cv2.resize(skin_region, self.target_size)
            
            # Normalize to [0, 1]
            processed_image = processed_image.astype(np.float32) / 255.0
            
            # Apply augmentation if requested and available
            if apply_augmentation and self.use_augmentation:
                # Convert to tensor for augmentation
                tensor_image = tf.convert_to_tensor(processed_image)
                tensor_image = tf.expand_dims(tensor_image, 0)  # Add batch dimension
                
                # Apply augmentation
                augmented = self.augmentation(tensor_image, training=True)
                processed_image = tf.squeeze(augmented, 0).numpy()
            
            return processed_image
            
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
    
    def visualize_calibration(self, image_path):
        """Visualize the calibration process"""
        # Load original image
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detect calibration patches
        calibration_patches = self.detect_calibration_card(image)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Image with detected patches highlighted
        image_with_patches = image.copy()
        for patch in calibration_patches:
            x, y, w, h = patch['bbox']
            cv2.rectangle(image_with_patches, (x, y), (x+w, y+h), (255, 0, 0), 3)
        
        axes[0, 1].imshow(image_with_patches)
        axes[0, 1].set_title('Detected Calibration Patches')
        axes[0, 1].axis('off')
        
        # Color corrected image
        corrected = self.color_correct_image(image, calibration_patches)
        axes[1, 0].imshow(corrected)
        axes[1, 0].set_title('Color Corrected')
        axes[1, 0].axis('off')
        
        # Final processed skin region
        skin_region = self.find_skin_region(corrected, calibration_patches)
        final = cv2.resize(skin_region, self.target_size)
        axes[1, 1].imshow(final)
        axes[1, 1].set_title('Final Processed Skin Region')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig

# Keep the simple processor for backward compatibility
class SimpleImageProcessor:
    """Simplified image processing without calibration"""
    
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size
        
    def process_image(self, image_path):
        """Simple processing pipeline focusing on central skin region"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            h, w = image.shape[:2]
            center_y, center_x = h // 2, w // 2
            crop_size = min(h, w) // 3
            
            y1 = max(0, center_y - crop_size)
            y2 = min(h, center_y + crop_size)
            x1 = max(0, center_x - crop_size)
            x2 = min(w, center_x + crop_size)
            
            skin_region = image[y1:y2, x1:x2]
            image = cv2.resize(skin_region, self.target_size)
            image = image.astype(np.float32) / 255.0
            
            return image
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            return np.zeros((self.target_size[0], self.target_size[1], 3), dtype=np.float32)

class SimpleTabularProcessor:
    """Simplified tabular data preprocessing"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
    
    def fit(self, df, categorical_cols, numerical_cols):
        """Fit preprocessors on training data"""
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                le.fit(df[col].astype(str))
                self.label_encoders[col] = le
        
        numerical_data = df[numerical_cols].copy()
        for col in numerical_cols:
            if col in numerical_data.columns:
                median_val = numerical_data[col].median()
                numerical_data[col].fillna(median_val, inplace=True)
        
        self.scaler.fit(numerical_data)
        self.is_fitted = True
        return self
    
    def transform(self, df, categorical_cols, numerical_cols):
        """Transform data using fitted preprocessors"""
        if not self.is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        processed_data = []
        
        for col in categorical_cols:
            if col in df.columns and col in self.label_encoders:
                encoded = self.label_encoders[col].transform(df[col].astype(str))
                processed_data.append(encoded.reshape(-1, 1))
        
        numerical_data = df[numerical_cols].copy()
        for col in numerical_cols:
            if col in numerical_data.columns:
                median_val = numerical_data[col].median()
                numerical_data[col].fillna(median_val, inplace=True)
        
        scaled_numerical = self.scaler.transform(numerical_data)
        processed_data.append(scaled_numerical)
        
        if processed_data:
            return np.concatenate(processed_data, axis=1)
        else:
            return np.array([]).reshape(len(df), 0)
    
    def fit_transform(self, df, categorical_cols, numerical_cols):
        """Fit and transform in one step"""
        return self.fit(df, categorical_cols, numerical_cols).transform(df, categorical_cols, numerical_cols)

def create_train_val_splits(df, n_folds=5, random_state=42):
    """Create stratified k-fold splits for cross-validation"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    splits = []
    for train_idx, val_idx in skf.split(df.index, df['jaundiced']):
        splits.append((train_idx, val_idx))
    
    return splits

def test_calibration_on_sample(csv_path, images_dir, sample_size=5):
    """Test calibration on a sample of images"""
    print("🔍 Testing Color Calibration on Sample Images")
    print("=" * 50)
    
    # Load dataset
    df = pd.read_csv(csv_path)
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # Create processors
    calibrated_processor = ColorCalibratedImageProcessor(use_calibration=True, use_augmentation=False)
    simple_processor = SimpleImageProcessor()
    
    results = []
    
    for idx, row in sample_df.iterrows():
        image_path = images_dir / row['image_idx']
        
        if image_path.exists():
            print(f"Processing: {row['image_idx']}")
            
            # Process with both methods
            calibrated_result = calibrated_processor.process_image(image_path)
            simple_result = simple_processor.process_image(image_path)
            
            results.append({
                'image_path': image_path,
                'calibrated': calibrated_result,
                'simple': simple_result,
                'jaundiced': row['jaundiced']
            })
            
            # Show calibration visualization for first image
            if len(results) == 1:
                fig = calibrated_processor.visualize_calibration(image_path)
                if fig:
                    plt.show()
    
    print(f"✅ Processed {len(results)} sample images")
    return results

def create_cnn_processor(use_calibration=True, use_augmentation=True):
    """Create image processor optimized for CNN model (128x128)"""
    return ColorCalibratedImageProcessor(
        target_size=(128, 128),
        use_calibration=use_calibration,
        use_augmentation=use_augmentation
    )

def create_efficientnet_processor(use_calibration=True, use_augmentation=True):
    """Create image processor optimized for EfficientNet model (224x224)"""
    return ColorCalibratedImageProcessor(
        target_size=(224, 224),
        use_calibration=use_calibration,
        use_augmentation=use_augmentation
    )

if __name__ == "__main__":
    # Test the calibration functionality
    csv_path = "jaundice_dataset/chd_jaundice_published_2.csv"
    images_dir = Path("D:/NU_Courses/semester_6/MI/NeoJaundice/images")
    
    test_calibration_on_sample(csv_path, images_dir) 
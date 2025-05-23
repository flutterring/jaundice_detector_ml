import pandas as pd
import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split

def augment_image(image):
    """
    Apply data augmentation to an image
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    return image

def load_and_preprocess_data(csv_path, images_dir):
    """
    Load and preprocess the jaundice dataset
    
    Args:
        csv_path (str): Path to the CSV file containing metadata
        images_dir (str): Path to the directory containing images
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Drop unnecessary columns
    df = df.drop(['gender', 'age(day)', 'weight'], axis=1, errors='ignore')
    
    # Get image names
    image_names = df['image_idx'].tolist()
    
    # Process images
    processed_images = []
    
    def preprocess_image_tf(image_id):
        filename = f"{image_id}"
        full_path = os.path.join(images_dir, filename)
        
        img = tf.io.read_file(full_path)
        img = tf.image.decode_image(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = tf.cast(img, tf.float32) / 255.0
        return img
    
    for image_id in image_names:
        try:
            processed_img = preprocess_image_tf(image_id)
            processed_images.append(processed_img.numpy())
        except Exception as e:
            print(f"❌ Error processing {image_id}: {e}")
    
    processed_images = np.array(processed_images)
    print("✅ Final shape:", processed_images.shape)
    
    # Prepare labels
    X = processed_images
    y = df["Treatment"]
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42)
    
    print("Shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def create_tf_datasets(X_train, y_train, X_val, y_val, batch_size=32):
    """
    Create TensorFlow datasets for training and validation
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels
        batch_size (int): Batch size for the datasets
        
    Returns:
        tuple: (train_ds, val_ds)
    """
    # Create training dataset with augmentation
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    train_ds = train_ds.map(lambda x, y: (augment_image(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Create validation dataset without augmentation
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds

 
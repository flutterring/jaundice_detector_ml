import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras import layers, models
import os

def create_model(input_shape=(224, 224, 3)):
    """
    Create and compile the EfficientNet model for jaundice detection
    
    Args:
        input_shape (tuple): Input shape of the images
        
    Returns:
        model: Compiled Keras model
    """
    # Remove corrupted EfficientNetB0 weights if they exist
    keras_cache_dir = os.path.expanduser("~/.keras/models")
    for fname in os.listdir(keras_cache_dir):
        if "efficientnetb0" in fname.lower():
            os.remove(os.path.join(keras_cache_dir, fname))
    
    # Create base model
    base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False  # Freeze for transfer learning
    
    # Create model
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

def fine_tune_model(model, train_ds, val_ds, epochs=5, learning_rate=1e-5):
    """
    Fine-tune the model by unfreezing some layers
    
    Args:
        model: The model to fine-tune
        train_ds, val_ds: Training and validation datasets
        epochs (int): Number of epochs for fine-tuning
        learning_rate (float): Learning rate for fine-tuning
        
    Returns:
        history: Training history
    """
    # Unfreeze base model
    base_model = model.layers[0]
    base_model.trainable = True
    
    # Freeze first few layers to avoid catastrophic forgetting
    for layer in base_model.layers[:100]:
        layer.trainable = False
    
    # Recompile model with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Fine-tune the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )
    
    return history

def save_model(model, filepath):
    """
    Save the model to disk
    
    Args:
        model: The model to save
        filepath (str): Path where to save the model
    """
    try:
        model.save(filepath)
        print(f"✅ Model saved successfully to {filepath}")
    except Exception as e:
        print(f"❌ Error saving model: {e}")

if __name__ == "__main__":
    # Example usage
    from preprocessing import load_and_preprocess_data, create_tf_datasets
    
    # Load and preprocess data
    csv_path = "../jaundice_dataset/chd_jaundice_published_2.csv"
    images_dir = "D:/NU_Courses/semester_6/MI/NeoJaundice/images"
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(csv_path, images_dir)
    train_ds, val_ds = create_tf_datasets(X_train, y_train, X_val, y_val)
    
    # Create and train model
    model = create_model()
    history = model.fit(train_ds, validation_data=val_ds, epochs=10)
    
    # Fine-tune model
    history_fine = fine_tune_model(model, train_ds, val_ds)
    
    # Save model
    save_model(model, "efficientnet_jaundice_model.h5") 
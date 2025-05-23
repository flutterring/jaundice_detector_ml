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
    
    # Create model with improved architecture
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    # Compile model with improved optimizer settings
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def fine_tune_model(model, train_ds, val_ds, epochs=10, learning_rate=1e-5):
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
    
    # Add callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Recompile model with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    # Fine-tune the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
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
        # Save full model
        model.save(filepath)
        print(f"✅ Full model saved successfully to {filepath}")
        
        # Convert and save TFLite model
        tflite_path = filepath.replace('.h5', '.tflite')
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        print(f"✅ TFLite model saved successfully to {tflite_path}")
        
    except Exception as e:
        print(f"❌ Error saving model: {e}")

if __name__ == "__main__":
    # Example usage
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from ml_model.utils.preprocessing import load_and_preprocess_data, create_tf_datasets
    
    # Load and preprocess data
    csv_path = "ml_model/jaundice_dataset/chd_jaundice_published_2.csv"
    images_dir = "D:/NU_Courses/semester_6/MI/NeoJaundice/images"
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_data(csv_path, images_dir)
    train_ds, val_ds = create_tf_datasets(X_train, y_train, X_val, y_val)
    
    # Create and train model
    model = create_model()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=3,
                min_lr=1e-6
            )
        ]
    )
    
    # Fine-tune model
    history_fine = fine_tune_model(model, train_ds, val_ds)
    
    # Save model
    save_model(model, "ml_model/saved_models/efficientnet_jaundice_model.h5") 
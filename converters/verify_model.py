import tensorflow as tf
from tensorflow.keras.models import load_model
import os

model_path = 'frutas_cnn.h5'

if os.path.exists(model_path):
    print(f"Attempting to load the full model from: {model_path}")
    try:
        # Load the model including architecture and weights
        loaded_model = load_model(model_path)
        print("Model loaded successfully!")
        loaded_model.summary()
    except Exception as e:
        print(f"Failed to load the full model. The file might be corrupted or is weights-only.")
        print(f"Error details: {e}")
else:
    print(f"Error: Model file not found at {model_path}")


import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Define the paths
h5_model_path = '/Users/serbarale/Documents/GitHub/clasificador_frutas/frutas_cnn.h5'
savedmodel_path = '/Users/serbarale/Documents/GitHub/clasificador_frutas/frutas_cnn_savedmodel'

# 1. Load the full model from the H5 file
print("Loading model from H5 file...")
try:
    model = load_model(h5_model_path)
    print("Model loaded successfully!")

    # 2. Save the loaded model in the SavedModel format
    print(f"Saving model to SavedModel format at {savedmodel_path}...")
    tf.saved_model.save(model, savedmodel_path)
    print("Model successfully saved as SavedModel.")
    
except Exception as e:
    print(f"An error occurred: {e}")
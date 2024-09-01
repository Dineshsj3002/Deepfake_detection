# deepfake_detector.py

from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import tensorflow as tf
model = tf.keras.models.load_model('C:/Users/sjdin/OneDrive/Documents/deepfake_detection/deepfake_model.h5')

# Load the pre-trained model


def detect_deepfake(image_path):
    """
    Function to detect deepfake.

    :param image_path: Path to the image or video file
    :return: Detection result
    """
    # Load and preprocess the image
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Make prediction
    prediction = model.predict(image)
    return prediction

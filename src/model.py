"""
CNN Model Architecture for Ocean Objects Classification

This module contains the definition of the Convolutional Neural Network
used for classifying underwater objects and marine life.
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import Callback
import numpy as np


class EarlyStoppingCallback(Callback):
    """
    Custom callback to stop training when accuracy reaches 99%
    """
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        if logs.get('acc') is not None and logs.get('acc') > 0.99:
            print("\nReached 99% accuracy so cancelling training!")
            self.model.stop_training = True


class OceanObjectsCNN:
    """
    Convolutional Neural Network for Ocean Objects Classification
    
    This class implements a CNN architecture optimized for classifying
    10 different categories of ocean objects and marine life.
    """
    
    def __init__(self, input_shape=(150, 150, 3), num_classes=10):
        """
        Initialize the CNN model
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels)
            num_classes (int): Number of object classes to classify
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        self.classes = [
            "coral_reef", "diver", "fish", "jellyfish", "plant",
            "robot", "ruins", "shark", "starfish", "turtle"
        ]
        
    def build_model(self):
        """
        Build the CNN architecture
        
        Returns:
            tensorflow.keras.Model: Compiled CNN model
        """
        self.model = Sequential([
            # First convolutional block
            Conv2D(16, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D(2, 2),
            
            # Second convolutional block
            Conv2D(32, (3, 3), activation='relu'),
            Conv2D(32, (3, 3), activation='relu'),
            
            # Third convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Fourth convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Fifth convolutional block
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(2, 2),
            
            # Flatten and dense layers
            Flatten(),
            Dense(128, activation='relu'),
            Dense(self.num_classes, activation='softmax')
        ])
        
        return self.model
    
    def compile_model(self, learning_rate=0.001):
        """
        Compile the model with optimizer, loss function, and metrics
        
        Args:
            learning_rate (float): Learning rate for the optimizer
        """
        if self.model is None:
            self.build_model()
            
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=RMSprop(learning_rate=learning_rate),
            metrics=['acc']
        )
    
    def get_model_summary(self):
        """
        Get a summary of the model architecture
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            self.build_model()
        return self.model.summary()
    
    def get_callbacks(self):
        """
        Get training callbacks
        
        Returns:
            list: List of Keras callbacks
        """
        return [EarlyStoppingCallback()]
    
    def save_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save. Please build and train the model first.")
    
    def load_model(self, filepath):
        """
        Load a pre-trained model
        
        Args:
            filepath (str): Path to the saved model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def predict_class(self, image):
        """
        Predict the class of a single image
        
        Args:
            image (numpy.ndarray): Preprocessed image array
            
        Returns:
            tuple: (predicted_class_name, confidence_score)
        """
        if self.model is None:
            raise ValueError("Model not built or loaded. Please build/load model first.")
        
        # Ensure image has batch dimension
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        predictions = self.model.predict(image)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        return self.classes[predicted_class_idx], confidence 
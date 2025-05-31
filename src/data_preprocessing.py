"""
Data Preprocessing Utilities for Ocean Objects Classification

This module provides utilities for loading, preprocessing, and augmenting
image data for the CNN model training.
"""

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image
import matplotlib.pyplot as plt


class DataPreprocessor:
    """
    Class for handling data preprocessing operations
    """
    
    def __init__(self, target_size=(150, 150), batch_size=64):
        """
        Initialize the data preprocessor
        
        Args:
            target_size (tuple): Target size for resizing images
            batch_size (int): Batch size for data generators
        """
        self.target_size = target_size
        self.batch_size = batch_size
        self.classes = [
            "coral_reef", "diver", "fish", "jellyfish", "plant",
            "robot", "ruins", "shark", "starfish", "turtle"
        ]
    
    def create_train_generator(self, train_dir, rescale=1./255, shuffle=True, 
                             class_mode='categorical', augment=False):
        """
        Create a data generator for training data
        
        Args:
            train_dir (str): Path to training data directory
            rescale (float): Rescaling factor for pixel values
            shuffle (bool): Whether to shuffle the data
            class_mode (str): Type of class labels to return
            augment (bool): Whether to apply data augmentation
            
        Returns:
            tensorflow.keras.preprocessing.image.DirectoryIterator: Data generator
        """
        if augment:
            train_datagen = ImageDataGenerator(
                rescale=rescale,
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=rescale)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=shuffle
        )
        
        return train_generator
    
    def create_validation_generator(self, val_dir, rescale=1./255, 
                                  class_mode='categorical'):
        """
        Create a data generator for validation data
        
        Args:
            val_dir (str): Path to validation data directory
            rescale (float): Rescaling factor for pixel values
            class_mode (str): Type of class labels to return
            
        Returns:
            tensorflow.keras.preprocessing.image.DirectoryIterator: Data generator
        """
        val_datagen = ImageDataGenerator(rescale=rescale)
        
        val_generator = val_datagen.flow_from_directory(
            val_dir,
            target_size=self.target_size,
            batch_size=self.batch_size,
            class_mode=class_mode,
            shuffle=False
        )
        
        return val_generator
    
    def preprocess_single_image(self, image_path, rescale=True):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path (str): Path to the image file
            rescale (bool): Whether to rescale pixel values
            
        Returns:
            numpy.ndarray: Preprocessed image array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        image = cv2.resize(image, self.target_size)
        
        # Rescale if requested
        if rescale:
            image = image.astype(np.float32) / 255.0
        
        return image
    
    def extract_rar_files(self, rar_dir, extract_dir):
        """
        Extract RAR files from a directory
        
        Args:
            rar_dir (str): Directory containing RAR files
            extract_dir (str): Directory to extract files to
        """
        import subprocess
        
        rar_files = [f for f in os.listdir(rar_dir) if f.endswith('.rar')]
        
        for rar_file in rar_files:
            rar_path = os.path.join(rar_dir, rar_file)
            print(f"Extracting {rar_file}...")
            
            try:
                subprocess.run(['unrar', 'x', rar_path, extract_dir], 
                             check=True, capture_output=True)
                print(f"Successfully extracted {rar_file}")
            except subprocess.CalledProcessError as e:
                print(f"Error extracting {rar_file}: {e}")
            except FileNotFoundError:
                print("unrar command not found. Please install unrar utility.")
                break
    
    def visualize_data_distribution(self, data_dir):
        """
        Visualize the distribution of classes in the dataset
        
        Args:
            data_dir (str): Path to data directory
        """
        class_counts = {}
        
        for class_name in os.listdir(data_dir):
            class_path = os.path.join(data_dir, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                class_counts[class_name] = count
        
        # Create bar plot
        plt.figure(figsize=(12, 6))
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        plt.bar(classes, counts)
        plt.title('Distribution of Classes in Dataset')
        plt.xlabel('Classes')
        plt.ylabel('Number of Images')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        return class_counts
    
    def display_sample_images(self, data_dir, samples_per_class=3):
        """
        Display sample images from each class
        
        Args:
            data_dir (str): Path to data directory
            samples_per_class (int): Number of sample images per class
        """
        classes = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d))]
        
        fig, axes = plt.subplots(len(classes), samples_per_class, 
                                figsize=(15, 3*len(classes)))
        
        for i, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for j in range(min(samples_per_class, len(images))):
                img_path = os.path.join(class_path, images[j])
                img = Image.open(img_path)
                
                if len(classes) == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                
                ax.imshow(img)
                ax.set_title(f'{class_name}')
                ax.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def get_class_weights(self, data_dir):
        """
        Calculate class weights for handling imbalanced datasets
        
        Args:
            data_dir (str): Path to data directory
            
        Returns:
            dict: Dictionary mapping class indices to weights
        """
        class_counts = self.visualize_data_distribution(data_dir)
        total_samples = sum(class_counts.values())
        
        class_weights = {}
        for i, (class_name, count) in enumerate(class_counts.items()):
            class_weights[i] = total_samples / (len(class_counts) * count)
        
        return class_weights 
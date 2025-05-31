"""
Prediction Module for Ocean Objects Classification

This module provides utilities for making predictions on new images
using the trained CNN model.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from .model import OceanObjectsCNN
from .data_preprocessing import DataPreprocessor


class OceanObjectsPredictor:
    """
    Class for handling predictions on ocean object images
    """
    
    def __init__(self, model_path=None, model=None):
        """
        Initialize the predictor
        
        Args:
            model_path (str): Path to saved model file
            model (OceanObjectsCNN): Pre-loaded model instance
        """
        self.model = model if model else OceanObjectsCNN()
        self.data_preprocessor = DataPreprocessor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        elif model is None:
            print("Warning: No model loaded. Please load a model before making predictions.")
    
    def load_model(self, model_path):
        """
        Load a pre-trained model
        
        Args:
            model_path (str): Path to the saved model
        """
        self.model.load_model(model_path)
        print("Model loaded successfully!")
    
    def predict_single_image(self, image_path, show_image=True, show_confidence=True):
        """
        Predict the class of a single image
        
        Args:
            image_path (str): Path to the image file
            show_image (bool): Whether to display the image
            show_confidence (bool): Whether to show confidence scores
            
        Returns:
            tuple: (predicted_class, confidence, all_predictions)
        """
        if self.model.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Preprocess the image
        try:
            image = self.data_preprocessor.preprocess_single_image(image_path)
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None, None, None
        
        # Make prediction
        predicted_class, confidence = self.model.predict_class(image)
        
        # Get all class predictions for detailed analysis
        image_batch = np.expand_dims(image, axis=0)
        all_predictions = self.model.model.predict(image_batch)[0]
        
        # Display results
        if show_image:
            self._display_prediction_result(image_path, predicted_class, 
                                          confidence, all_predictions, 
                                          show_confidence)
        
        return predicted_class, confidence, all_predictions
    
    def predict_batch_images(self, image_paths, show_results=True):
        """
        Predict classes for multiple images
        
        Args:
            image_paths (list): List of image file paths
            show_results (bool): Whether to display results
            
        Returns:
            list: List of prediction results
        """
        if self.model.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        results = []
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {os.path.basename(image_path)}")
            
            try:
                predicted_class, confidence, all_predictions = self.predict_single_image(
                    image_path, show_image=False, show_confidence=False
                )
                
                results.append({
                    'image_path': image_path,
                    'predicted_class': predicted_class,
                    'confidence': confidence,
                    'all_predictions': all_predictions
                })
                
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        if show_results:
            self._display_batch_results(results)
        
        return results
    
    def predict_from_directory(self, directory_path, max_images=None, show_results=True):
        """
        Predict classes for all images in a directory
        
        Args:
            directory_path (str): Path to directory containing images
            max_images (int): Maximum number of images to process
            show_results (bool): Whether to display results
            
        Returns:
            list: List of prediction results
        """
        # Get all image files from directory
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = []
        
        for file in os.listdir(directory_path):
            if file.lower().endswith(image_extensions):
                image_files.append(os.path.join(directory_path, file))
        
        if not image_files:
            print(f"No image files found in {directory_path}")
            return []
        
        # Limit number of images if specified
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Found {len(image_files)} images to process")
        
        return self.predict_batch_images(image_files, show_results)
    
    def _display_prediction_result(self, image_path, predicted_class, confidence, 
                                 all_predictions, show_confidence=True):
        """
        Display prediction result with image and confidence scores
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Display image
        image = Image.open(image_path)
        ax1.imshow(image)
        ax1.set_title(f'Predicted: {predicted_class}\nConfidence: {confidence:.3f}')
        ax1.axis('off')
        
        # Display confidence scores if requested
        if show_confidence:
            classes = self.model.classes
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))
            
            y_pos = np.arange(len(classes))
            bars = ax2.barh(y_pos, all_predictions, color=colors)
            
            # Highlight the predicted class
            max_idx = np.argmax(all_predictions)
            bars[max_idx].set_color('red')
            
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(classes)
            ax2.set_xlabel('Confidence Score')
            ax2.set_title('Class Confidence Scores')
            ax2.set_xlim(0, 1)
            
            # Add value labels on bars
            for i, v in enumerate(all_predictions):
                ax2.text(v + 0.01, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.show()
    
    def _display_batch_results(self, results):
        """
        Display summary of batch prediction results
        """
        print("\n" + "="*60)
        print("BATCH PREDICTION RESULTS")
        print("="*60)
        
        successful_predictions = [r for r in results if 'predicted_class' in r]
        errors = [r for r in results if 'error' in r]
        
        print(f"Total images processed: {len(results)}")
        print(f"Successful predictions: {len(successful_predictions)}")
        print(f"Errors: {len(errors)}")
        
        if successful_predictions:
            print("\nPrediction Summary:")
            print("-" * 40)
            
            # Count predictions by class
            class_counts = {}
            total_confidence = 0
            
            for result in successful_predictions:
                pred_class = result['predicted_class']
                confidence = result['confidence']
                
                if pred_class in class_counts:
                    class_counts[pred_class]['count'] += 1
                    class_counts[pred_class]['total_confidence'] += confidence
                else:
                    class_counts[pred_class] = {'count': 1, 'total_confidence': confidence}
                
                total_confidence += confidence
            
            # Display class distribution
            for class_name, data in class_counts.items():
                avg_confidence = data['total_confidence'] / data['count']
                print(f"{class_name}: {data['count']} images (avg confidence: {avg_confidence:.3f})")
            
            avg_overall_confidence = total_confidence / len(successful_predictions)
            print(f"\nOverall average confidence: {avg_overall_confidence:.3f}")
        
        if errors:
            print(f"\nErrors encountered:")
            print("-" * 20)
            for result in errors:
                print(f"File: {os.path.basename(result['image_path'])}")
                print(f"Error: {result['error']}\n")
        
        print("="*60)
    
    def create_prediction_report(self, results, save_path=None):
        """
        Create a detailed prediction report
        
        Args:
            results (list): List of prediction results
            save_path (str): Path to save the report
            
        Returns:
            str: Report content
        """
        successful_predictions = [r for r in results if 'predicted_class' in r]
        
        report_lines = [
            "Ocean Objects Classification - Prediction Report",
            "=" * 50,
            f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total images processed: {len(results)}",
            f"Successful predictions: {len(successful_predictions)}",
            f"Failed predictions: {len(results) - len(successful_predictions)}",
            "",
            "Detailed Results:",
            "-" * 30
        ]
        
        for i, result in enumerate(successful_predictions, 1):
            report_lines.extend([
                f"{i}. File: {os.path.basename(result['image_path'])}",
                f"   Predicted Class: {result['predicted_class']}",
                f"   Confidence: {result['confidence']:.4f}",
                ""
            ])
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_content)
            print(f"Prediction report saved to: {save_path}")
        
        return report_content 
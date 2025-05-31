"""
Training Module for Ocean Objects Classification

This module provides utilities for training the CNN model with proper
configuration and monitoring.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from .model import OceanObjectsCNN
from .data_preprocessing import DataPreprocessor


class ModelTrainer:
    """
    Class for handling model training operations
    """
    
    def __init__(self, model=None, data_preprocessor=None):
        """
        Initialize the model trainer
        
        Args:
            model (OceanObjectsCNN): CNN model instance
            data_preprocessor (DataPreprocessor): Data preprocessing instance
        """
        self.model = model if model else OceanObjectsCNN()
        self.data_preprocessor = data_preprocessor if data_preprocessor else DataPreprocessor()
        self.history = None
        
    def prepare_data(self, train_dir, val_dir=None, augment_data=True):
        """
        Prepare training and validation data generators
        
        Args:
            train_dir (str): Path to training data directory
            val_dir (str): Path to validation data directory (optional)
            augment_data (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        print("Preparing training data...")
        train_generator = self.data_preprocessor.create_train_generator(
            train_dir, augment=augment_data
        )
        
        val_generator = None
        if val_dir:
            print("Preparing validation data...")
            val_generator = self.data_preprocessor.create_validation_generator(val_dir)
        
        return train_generator, val_generator
    
    def train_model(self, train_generator, val_generator=None, epochs=100, 
                   steps_per_epoch=None, validation_steps=None, 
                   use_early_stopping=True, save_model_path=None):
        """
        Train the CNN model
        
        Args:
            train_generator: Training data generator
            val_generator: Validation data generator (optional)
            epochs (int): Maximum number of epochs
            steps_per_epoch (int): Steps per epoch (optional)
            validation_steps (int): Validation steps (optional)
            use_early_stopping (bool): Whether to use early stopping callback
            save_model_path (str): Path to save the trained model
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Build and compile model if not already done
        if self.model.model is None:
            print("Building and compiling model...")
            self.model.build_model()
            self.model.compile_model()
        
        # Display model summary
        print("\nModel Architecture:")
        self.model.get_model_summary()
        
        # Prepare callbacks
        callbacks = []
        if use_early_stopping:
            callbacks.extend(self.model.get_callbacks())
        
        # Calculate steps if not provided
        if steps_per_epoch is None:
            steps_per_epoch = len(train_generator)
        
        if val_generator and validation_steps is None:
            validation_steps = len(val_generator)
        
        print(f"\nStarting training...")
        print(f"Epochs: {epochs}")
        print(f"Steps per epoch: {steps_per_epoch}")
        if val_generator:
            print(f"Validation steps: {validation_steps}")
        
        # Train the model
        self.history = self.model.model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\nTraining completed!")
        
        # Save model if path provided
        if save_model_path:
            self.save_trained_model(save_model_path)
        
        return self.history
    
    def save_trained_model(self, filepath):
        """
        Save the trained model
        
        Args:
            filepath (str): Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_model(filepath)
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and accuracy)
        
        Args:
            save_path (str): Path to save the plot (optional)
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        history = self.history.history
        epochs = range(1, len(history['loss']) + 1)
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training & validation loss
        ax1.plot(epochs, history['loss'], 'b-', label='Training Loss')
        if 'val_loss' in history:
            ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot training & validation accuracy
        ax2.plot(epochs, history['acc'], 'b-', label='Training Accuracy')
        if 'val_acc' in history:
            ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")
        
        plt.show()
    
    def evaluate_model(self, test_generator):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            tuple: (loss, accuracy)
        """
        if self.model.model is None:
            raise ValueError("Model not trained. Please train the model first.")
        
        print("Evaluating model on test data...")
        loss, accuracy = self.model.model.evaluate(test_generator, verbose=1)
        
        print(f"\nTest Results:")
        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        return loss, accuracy
    
    def get_training_summary(self):
        """
        Get a summary of the training process
        
        Returns:
            dict: Training summary statistics
        """
        if self.history is None:
            return None
        
        history = self.history.history
        
        summary = {
            'epochs_trained': len(history['loss']),
            'final_train_loss': history['loss'][-1],
            'final_train_acc': history['acc'][-1],
            'best_train_acc': max(history['acc']),
            'best_train_acc_epoch': np.argmax(history['acc']) + 1
        }
        
        if 'val_loss' in history:
            summary.update({
                'final_val_loss': history['val_loss'][-1],
                'final_val_acc': history['val_acc'][-1],
                'best_val_acc': max(history['val_acc']),
                'best_val_acc_epoch': np.argmax(history['val_acc']) + 1
            })
        
        return summary
    
    def print_training_summary(self):
        """
        Print a formatted training summary
        """
        summary = self.get_training_summary()
        
        if summary is None:
            print("No training summary available.")
            return
        
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Epochs Trained: {summary['epochs_trained']}")
        print(f"Final Training Loss: {summary['final_train_loss']:.4f}")
        print(f"Final Training Accuracy: {summary['final_train_acc']:.4f}")
        print(f"Best Training Accuracy: {summary['best_train_acc']:.4f} (Epoch {summary['best_train_acc_epoch']})")
        
        if 'final_val_loss' in summary:
            print(f"Final Validation Loss: {summary['final_val_loss']:.4f}")
            print(f"Final Validation Accuracy: {summary['final_val_acc']:.4f}")
            print(f"Best Validation Accuracy: {summary['best_val_acc']:.4f} (Epoch {summary['best_val_acc_epoch']})")
        
        print("="*50) 
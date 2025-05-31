"""
Ocean Objects Classification - Example Usage

This script demonstrates how to use the modularized Ocean Objects Classification system.
"""

import sys
import os

# Add src directory to path
sys.path.append('../src')

from model import OceanObjectsCNN
from data_preprocessing import DataPreprocessor
from train import ModelTrainer
from predict import OceanObjectsPredictor


def main():
    """
    Main function demonstrating the usage of the Ocean Objects Classification system
    """
    print("Ocean Objects Classification - Example Usage")
    print("=" * 50)
    
    # 1. Initialize components
    print("\n1. Initializing components...")
    model = OceanObjectsCNN(input_shape=(150, 150, 3), num_classes=10)
    preprocessor = DataPreprocessor(target_size=(150, 150), batch_size=64)
    trainer = ModelTrainer(model=model, data_preprocessor=preprocessor)
    
    # 2. Build and compile model
    print("\n2. Building and compiling model...")
    model.build_model()
    model.compile_model(learning_rate=0.001)
    
    # Display model summary
    print("\n3. Model Architecture:")
    model.get_model_summary()
    
    # 3. Data preparation (example paths)
    print("\n4. Data preparation example:")
    train_data_path = '../data/train/'
    test_data_path = '../data/test/'
    
    print(f"Training data path: {train_data_path}")
    print(f"Test data path: {test_data_path}")
    print("Note: Extract RAR files before training!")
    
    # Example of how to extract RAR files
    print("\nTo extract RAR files, use:")
    print("preprocessor.extract_rar_files('../data/train/', '../data/train_extracted/')")
    
    # 4. Training example (commented out as data needs to be extracted first)
    print("\n5. Training example (requires extracted data):")
    print("""
    # Prepare data generators
    train_generator, val_generator = trainer.prepare_data(
        '../data/train_extracted/', 
        augment_data=True
    )
    
    # Train the model
    history = trainer.train_model(
        train_generator,
        epochs=100,
        steps_per_epoch=8,
        save_model_path='../models/ocean_objects_model.h5'
    )
    
    # Plot training history
    trainer.plot_training_history(save_path='../docs/training_history.png')
    trainer.print_training_summary()
    """)
    
    # 5. Prediction example
    print("\n6. Prediction example (requires trained model):")
    print("""
    # Initialize predictor
    predictor = OceanObjectsPredictor(model_path='../models/ocean_objects_model.h5')
    
    # Predict single image
    predicted_class, confidence, all_predictions = predictor.predict_single_image(
        'path/to/test/image.jpg',
        show_image=True,
        show_confidence=True
    )
    
    # Batch predictions
    results = predictor.predict_from_directory(
        '../data/test_extracted/some_class/',
        max_images=10,
        show_results=True
    )
    """)
    
    # 6. Replicating original notebook workflow
    print("\n7. Replicating original AUV_FP.ipynb workflow:")
    
    # Create model (same as original)
    ocean_model = OceanObjectsCNN()
    ocean_model.build_model()
    ocean_model.compile_model()
    
    # Get callbacks (same as original)
    callbacks = ocean_model.get_callbacks()
    
    print("Original workflow components ready!")
    print("Classes:", ocean_model.classes)
    
    print("\n" + "=" * 50)
    print("Setup complete! Ready to train with extracted data.")
    print("Follow the README.md for detailed instructions.")


if __name__ == "__main__":
    main()

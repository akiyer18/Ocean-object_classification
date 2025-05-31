# API Reference

## Ocean Objects Classification API Documentation

This document provides detailed API reference for all modules in the Ocean Objects Classification system.

## Table of Contents

1. [OceanObjectsCNN](#oceanobjectscnn)
2. [DataPreprocessor](#datapreprocessor)
3. [ModelTrainer](#modeltrainer)
4. [OceanObjectsPredictor](#oceanobjectspredictor)

---

## OceanObjectsCNN

Main CNN model class for ocean objects classification.

### Constructor

```python
OceanObjectsCNN(input_shape=(150, 150, 3), num_classes=10)
```

**Parameters:**
- `input_shape` (tuple): Shape of input images (height, width, channels)
- `num_classes` (int): Number of object classes to classify

### Methods

#### `build_model()`
Builds the CNN architecture.

**Returns:** `tensorflow.keras.Model`

#### `compile_model(learning_rate=0.001)`
Compiles the model with optimizer, loss function, and metrics.

**Parameters:**
- `learning_rate` (float): Learning rate for the optimizer

#### `get_model_summary()`
Returns a summary of the model architecture.

#### `get_callbacks()`
Returns training callbacks including early stopping.

**Returns:** `list` of Keras callbacks

#### `save_model(filepath)`
Saves the trained model.

**Parameters:**
- `filepath` (str): Path to save the model

#### `load_model(filepath)`
Loads a pre-trained model.

**Parameters:**
- `filepath` (str): Path to the saved model

#### `predict_class(image)`
Predicts the class of a single image.

**Parameters:**
- `image` (numpy.ndarray): Preprocessed image array

**Returns:** `tuple` (predicted_class_name, confidence_score)

---

## DataPreprocessor

Class for handling data preprocessing operations.

### Constructor

```python
DataPreprocessor(target_size=(150, 150), batch_size=64)
```

**Parameters:**
- `target_size` (tuple): Target size for resizing images
- `batch_size` (int): Batch size for data generators

### Methods

#### `create_train_generator(train_dir, rescale=1./255, shuffle=True, class_mode='categorical', augment=False)`
Creates a data generator for training data.

**Parameters:**
- `train_dir` (str): Path to training data directory
- `rescale` (float): Rescaling factor for pixel values
- `shuffle` (bool): Whether to shuffle the data
- `class_mode` (str): Type of class labels to return
- `augment` (bool): Whether to apply data augmentation

**Returns:** `tensorflow.keras.preprocessing.image.DirectoryIterator`

#### `create_validation_generator(val_dir, rescale=1./255, class_mode='categorical')`
Creates a data generator for validation data.

**Parameters:**
- `val_dir` (str): Path to validation data directory
- `rescale` (float): Rescaling factor for pixel values
- `class_mode` (str): Type of class labels to return

**Returns:** `tensorflow.keras.preprocessing.image.DirectoryIterator`

#### `preprocess_single_image(image_path, rescale=True)`
Preprocesses a single image for prediction.

**Parameters:**
- `image_path` (str): Path to the image file
- `rescale` (bool): Whether to rescale pixel values

**Returns:** `numpy.ndarray` - Preprocessed image array

#### `extract_rar_files(rar_dir, extract_dir)`
Extracts RAR files from a directory.

**Parameters:**
- `rar_dir` (str): Directory containing RAR files
- `extract_dir` (str): Directory to extract files to

#### `visualize_data_distribution(data_dir)`
Visualizes the distribution of classes in the dataset.

**Parameters:**
- `data_dir` (str): Path to data directory

#### `display_sample_images(data_dir, samples_per_class=3)`
Displays sample images from each class.

**Parameters:**
- `data_dir` (str): Path to data directory
- `samples_per_class` (int): Number of sample images per class

#### `get_class_weights(data_dir)`
Calculates class weights for handling imbalanced datasets.

**Parameters:**
- `data_dir` (str): Path to data directory

**Returns:** `dict` - Dictionary mapping class indices to weights

---

## ModelTrainer

Class for handling model training operations.

### Constructor

```python
ModelTrainer(model=None, data_preprocessor=None)
```

**Parameters:**
- `model` (OceanObjectsCNN): CNN model instance
- `data_preprocessor` (DataPreprocessor): Data preprocessing instance

### Methods

#### `prepare_data(train_dir, val_dir=None, augment_data=True)`
Prepares training and validation data generators.

**Parameters:**
- `train_dir` (str): Path to training data directory
- `val_dir` (str): Path to validation data directory (optional)
- `augment_data` (bool): Whether to apply data augmentation

**Returns:** `tuple` (train_generator, val_generator)

#### `train_model(train_generator, val_generator=None, epochs=100, steps_per_epoch=None, validation_steps=None, use_early_stopping=True, save_model_path=None)`
Trains the CNN model.

**Parameters:**
- `train_generator`: Training data generator
- `val_generator`: Validation data generator (optional)
- `epochs` (int): Maximum number of epochs
- `steps_per_epoch` (int): Steps per epoch (optional)
- `validation_steps` (int): Validation steps (optional)
- `use_early_stopping` (bool): Whether to use early stopping callback
- `save_model_path` (str): Path to save the trained model

**Returns:** `tensorflow.keras.callbacks.History`

#### `save_trained_model(filepath)`
Saves the trained model.

**Parameters:**
- `filepath` (str): Path to save the model

#### `plot_training_history(save_path=None)`
Plots training history (loss and accuracy).

**Parameters:**
- `save_path` (str): Path to save the plot (optional)

#### `evaluate_model(test_generator)`
Evaluates the model on test data.

**Parameters:**
- `test_generator`: Test data generator

**Returns:** `tuple` (loss, accuracy)

#### `get_training_summary()`
Gets a summary of the training process.

**Returns:** `dict` - Training summary statistics

#### `print_training_summary()`
Prints a formatted training summary.

---

## OceanObjectsPredictor

Class for handling predictions on ocean object images.

### Constructor

```python
OceanObjectsPredictor(model_path=None, model=None)
```

**Parameters:**
- `model_path` (str): Path to saved model file
- `model` (OceanObjectsCNN): Pre-loaded model instance

### Methods

#### `load_model(model_path)`
Loads a pre-trained model.

**Parameters:**
- `model_path` (str): Path to the saved model

#### `predict_single_image(image_path, show_image=True, show_confidence=True)`
Predicts the class of a single image.

**Parameters:**
- `image_path` (str): Path to the image file
- `show_image` (bool): Whether to display the image
- `show_confidence` (bool): Whether to show confidence scores

**Returns:** `tuple` (predicted_class, confidence, all_predictions)

#### `predict_batch_images(image_paths, show_results=True)`
Predicts classes for multiple images.

**Parameters:**
- `image_paths` (list): List of image file paths
- `show_results` (bool): Whether to display results

**Returns:** `list` - List of prediction results

#### `predict_from_directory(directory_path, max_images=None, show_results=True)`
Predicts classes for all images in a directory.

**Parameters:**
- `directory_path` (str): Path to directory containing images
- `max_images` (int): Maximum number of images to process
- `show_results` (bool): Whether to display results

**Returns:** `list` - List of prediction results

#### `create_prediction_report(results, save_path=None)`
Creates a detailed prediction report.

**Parameters:**
- `results` (list): List of prediction results
- `save_path` (str): Path to save the report

**Returns:** `str` - Report content

---

## Classes

The system recognizes 10 classes of ocean objects:

1. **coral_reef** - Various coral formations and reef structures
2. **diver** - Human divers underwater
3. **fish** - Various fish species
4. **jellyfish** - Different jellyfish types
5. **plant** - Marine vegetation and seaweed
6. **robot** - Underwater robots and equipment
7. **ruins** - Underwater archaeological structures
8. **shark** - Shark species
9. **starfish** - Various starfish species
10. **turtle** - Sea turtles

## Example Usage

```python
# Basic usage example
from src.model import OceanObjectsCNN
from src.train import ModelTrainer
from src.predict import OceanObjectsPredictor

# Create and train model
model = OceanObjectsCNN()
trainer = ModelTrainer(model=model)

# Train (after data preparation)
# history = trainer.train_model(train_generator, epochs=100)

# Make predictions
predictor = OceanObjectsPredictor(model=model)
predicted_class, confidence, _ = predictor.predict_single_image('image.jpg')
``` 
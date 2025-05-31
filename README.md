# Ocean Objects Classification - Deep Learning for Marine Robotics

![Ocean AI Banner](https://img.shields.io/badge/Project-Ocean%20Objects%20Classification-blue?style=for-the-badge)
![Deep Learning](https://img.shields.io/badge/Technology-Deep%20Learning-orange?style=for-the-badge)
![AUV Applications](https://img.shields.io/badge/Application-Autonomous%20Underwater%20Vehicles-green?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow%2FKeras-red?style=for-the-badge)

## Overview

**Ocean Objects Classification** is a state-of-the-art deep learning system designed for **Autonomous Underwater Vehicle (AUV) applications** that can identify and classify 10 different categories of marine objects and creatures. This project implements a custom Convolutional Neural Network (CNN) architecture optimized for underwater image classification, enabling real-time marine environment understanding for robotic systems.

### Project Context

- **Domain**: Marine Robotics & Computer Vision
- **Application**: Autonomous Underwater Vehicle Navigation
- **Technology Stack**: TensorFlow/Keras, Python, Deep Learning
- **Target Use Cases**: Underwater exploration, marine research, robotic navigation
- **Classification Categories**: 10 distinct marine object types

## Key Features

### Advanced CNN Architecture

- **Custom Deep Network**: Multi-layer convolutional neural network with ~509K parameters
- **Optimized Performance**: Achieves up to 99% training accuracy with early stopping
- **GPU Acceleration**: Full support for CUDA-enabled training and inference
- **Input Specifications**: 150×150×3 RGB image processing
- **Output Classes**: 10 marine object categories with confidence scores

### Comprehensive Marine Object Recognition

- **🐠 Marine Life**: Fish, jellyfish, sharks, starfish, sea turtles
- **🌊 Ocean Environment**: Coral reefs, marine plants, underwater ruins
- **🤖 Human Activity**: Divers, underwater robots and equipment
- **🧠 Intelligent Classification**: Softmax-based probability distribution
- **📊 Confidence Scoring**: Real-time prediction confidence metrics

### Professional Software Architecture

- **Modular Design**: Clean separation of concerns across multiple modules
- **Object-Oriented**: Well-structured classes for model, training, and prediction
- **API-Ready**: Easy integration with existing robotics systems
- **Extensible**: Simple to add new classes or modify architecture
- **Production-Ready**: Comprehensive error handling and logging

### Advanced Data Processing Pipeline

- **Smart Preprocessing**: Automated image resizing, normalization, and augmentation
- **Data Augmentation**: Rotation, shifting, zooming, and flipping for robustness
- **RAR Archive Support**: Built-in extraction utilities for compressed datasets
- **Batch Processing**: Efficient handling of large image datasets
- **Visualization Tools**: Data distribution analysis and sample image display

### Comprehensive Training System

- **Early Stopping**: Intelligent training termination at optimal performance
- **Progress Monitoring**: Real-time training visualization and metrics
- **Model Persistence**: Save/load functionality for trained models
- **History Tracking**: Complete training history with loss and accuracy plots
- **Hyperparameter Control**: Configurable learning rates and batch sizes

### Flexible Prediction Engine

- **Single Image Prediction**: Real-time classification with confidence scores
- **Batch Processing**: Efficient multiple image classification
- **Directory Processing**: Automatic processing of image folders
- **Visual Results**: Integrated matplotlib visualization
- **Report Generation**: Detailed prediction reports and analytics

## Technical Architecture

### Project Structure

```
Ocean-object_classification/
├── README.md                   # Comprehensive project documentation
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore patterns
├── requirements.txt           # Python dependencies
├── CHANGELOG.md              # Version history and updates
├── data/                      # Dataset organization
│   ├── train/                 # Training datasets (RAR archives)
│   │   ├── coral_reef.rar    # Coral formation images
│   │   ├── divers.rar        # Human diver images
│   │   ├── fish.rar          # Various fish species
│   │   ├── jellyfish.rar     # Jellyfish varieties
│   │   ├── plant.rar         # Marine vegetation
│   │   ├── robots.rar        # Underwater equipment
│   │   ├── ruins.rar         # Archaeological structures
│   │   ├── shark.rar         # Shark species
│   │   ├── starfish.rar      # Starfish varieties
│   │   └── turtle.rar        # Sea turtle species
│   ├── test/                  # Test datasets
│   └── test-colour/           # Color-specific test data
├── src/                       # Core source code modules
│   ├── __init__.py           # Package initialization
│   ├── model.py              # CNN architecture definition
│   ├── data_preprocessing.py  # Data handling utilities
│   ├── train.py              # Training pipeline
│   └── predict.py            # Prediction engine
├── models/                    # Saved model storage
├── notebooks/                 # Jupyter notebooks and examples
│   ├── AUV_FP.ipynb          # Original research notebook
│   └── example_usage.py      # Usage demonstration
└── docs/                      # Documentation
    └── API_REFERENCE.md      # Complete API documentation
```

### Core Components

#### CNN Model Architecture (`src/model.py`)

```python
class OceanObjectsCNN:
    """
    Custom CNN for marine object classification
    
    Architecture:
    - Input Layer: 150×150×3 RGB images
    - Convolutional Blocks: Progressive feature extraction
    - Pooling Layers: Spatial dimension reduction
    - Dense Layers: Classification with 128 neurons
    - Output: 10-class softmax classification
    """
```

#### Data Processing Pipeline (`src/data_preprocessing.py`)

```python
class DataPreprocessor:
    """
    Comprehensive data handling system
    
    Features:
    - RAR archive extraction
    - Image augmentation
    - Batch generation
    - Visualization tools
    - Class weight balancing
    """
```

#### Training System (`src/train.py`)

```python
class ModelTrainer:
    """
    Advanced training pipeline
    
    Capabilities:
    - Early stopping callbacks
    - Training history visualization
    - Model evaluation
    - Progress monitoring
    - Automated model saving
    """
```

#### Prediction Engine (`src/predict.py`)

```python
class OceanObjectsPredictor:
    """
    Flexible prediction system
    
    Modes:
    - Single image classification
    - Batch processing
    - Directory processing
    - Report generation
    - Confidence visualization
    """
```

## Installation & Setup

### Prerequisites

- **Python**: 3.7 or higher
- **TensorFlow**: 2.10+ (GPU support recommended)
- **Hardware**: CUDA-compatible GPU (optional but recommended)
- **Memory**: 8GB RAM minimum, 16GB recommended
- **Storage**: 5GB free space for datasets and models

### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/akiyer18/Ocean-object_classification.git
cd Ocean-object_classification
```

### 2. Environment Setup

```bash
# Create virtual environment
python -m venv ocean_classifier_env

# Activate environment
# On macOS/Linux:
source ocean_classifier_env/bin/activate
# On Windows:
ocean_classifier_env\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Check GPU availability (optional)
python -c "import tensorflow as tf; print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)"
```

### 4. Dataset Preparation

```bash
# Extract training data
cd data/train
for file in *.rar; do unrar x "$file"; done

# Extract test data
cd ../test
for file in *.rar; do unrar x "$file"; done

# Extract color test data
cd ../test-colour
unrar x test_col.rar
cd ../../
```

### 5. Quick Start

```python
# Basic usage example
from src.model import OceanObjectsCNN
from src.train import ModelTrainer
from src.predict import OceanObjectsPredictor

# Initialize and train model
model = OceanObjectsCNN()
trainer = ModelTrainer(model=model)

# Train with your extracted data
# trainer.train_model(train_generator, epochs=100)

# Make predictions
predictor = OceanObjectsPredictor(model=model)
predicted_class, confidence, _ = predictor.predict_single_image('your_image.jpg')
print(f"Predicted: {predicted_class} (Confidence: {confidence:.3f})")
```

## Marine Object Classes

### Classification Categories

| Class | Description | Applications |
|-------|-------------|-------------|
| **🪸 Coral Reef** | Various coral formations and reef structures | Navigation, ecosystem mapping |
| **🤿 Diver** | Human divers underwater | Safety monitoring, rescue operations |
| **🐠 Fish** | Various fish species | Marine life research, population studies |
| **🎐 Jellyfish** | Different jellyfish types | Hazard detection, marine biology |
| **🌿 Plant** | Marine vegetation and seaweed | Ecosystem analysis, navigation |
| **🤖 Robot** | Underwater robots and equipment | Equipment recognition, maintenance |
| **🏛️ Ruins** | Underwater archaeological structures | Exploration, cultural preservation |
| **🦈 Shark** | Shark species | Safety protocols, research |
| **⭐ Starfish** | Various starfish species | Biodiversity monitoring |
| **🐢 Turtle** | Sea turtles | Conservation, tracking |

## Model Performance & Specifications

### Technical Specifications

- **Architecture**: Custom CNN with 5 convolutional blocks
- **Parameters**: ~509,514 trainable parameters
- **Input Size**: 150×150×3 RGB images
- **Batch Size**: 64 images (configurable)
- **Optimizer**: RMSprop with 0.001 learning rate
- **Loss Function**: Categorical Crossentropy
- **Training Data**: 4,109 images across 10 classes

### Performance Metrics

- **Training Accuracy**: Up to 99% (with early stopping)
- **Model Size**: ~2MB (compressed)
- **Inference Speed**: ~50ms per image (GPU)
- **Memory Usage**: ~1GB during training
- **Convergence**: Typically 20-40 epochs

### Training Configuration

```python
# Model configuration
INPUT_SHAPE = (150, 150, 3)
NUM_CLASSES = 10
BATCH_SIZE = 64
LEARNING_RATE = 0.001
MAX_EPOCHS = 100
EARLY_STOPPING_THRESHOLD = 0.99
```

## Advanced Usage Examples

### 1. Custom Training Pipeline

```python
from src.model import OceanObjectsCNN
from src.train import ModelTrainer
from src.data_preprocessing import DataPreprocessor

# Initialize components
model = OceanObjectsCNN(input_shape=(150, 150, 3))
preprocessor = DataPreprocessor(target_size=(150, 150), batch_size=64)
trainer = ModelTrainer(model=model, data_preprocessor=preprocessor)

# Prepare data with augmentation
train_gen, val_gen = trainer.prepare_data(
    'data/train_extracted/',
    'data/test_extracted/',
    augment_data=True
)

# Train with monitoring
history = trainer.train_model(
    train_gen,
    val_generator=val_gen,
    epochs=100,
    save_model_path='models/ocean_classifier_v1.h5'
)

# Visualize results
trainer.plot_training_history(save_path='docs/training_results.png')
trainer.print_training_summary()
```

### 2. Batch Prediction System

```python
from src.predict import OceanObjectsPredictor

# Load trained model
predictor = OceanObjectsPredictor(model_path='models/ocean_classifier_v1.h5')

# Process entire directory
results = predictor.predict_from_directory(
    'path/to/new/images/',
    max_images=100,
    show_results=True
)

# Generate detailed report
report = predictor.create_prediction_report(
    results,
    save_path='reports/classification_results.txt'
)
```

### 3. Real-time AUV Integration

```python
import cv2
from src.predict import OceanObjectsPredictor

# Initialize predictor for real-time use
predictor = OceanObjectsPredictor(model_path='models/ocean_classifier_v1.h5')

# Process video stream (example)
def process_auv_feed(video_source):
    cap = cv2.VideoCapture(video_source)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Save frame temporarily
        cv2.imwrite('temp_frame.jpg', frame)
        
        # Classify
        class_name, confidence, _ = predictor.predict_single_image(
            'temp_frame.jpg',
            show_image=False
        )
        
        # Display results
        cv2.putText(frame, f'{class_name}: {confidence:.2f}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('AUV Classification', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

## Contributing & Development

### Development Setup

```bash
# Clone for development
git clone https://github.com/akiyer18/Ocean-object_classification.git
cd Ocean-object_classification

# Create development environment
python -m venv dev_env
source dev_env/bin/activate  # or dev_env\Scripts\activate on Windows

# Install development dependencies
pip install -r requirements.txt
pip install jupyter pytest black flake8  # Additional dev tools
```

### Contributing Guidelines

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Follow** PEP 8 coding standards
4. **Add** comprehensive tests for new functionality
5. **Update** documentation as needed
6. **Commit** with clear, descriptive messages
7. **Push** to your branch (`git push origin feature/amazing-feature`)
8. **Open** a Pull Request with detailed description

### Code Quality Standards

- **Documentation**: All functions must have docstrings
- **Testing**: Minimum 80% code coverage
- **Formatting**: Use Black for code formatting
- **Linting**: Pass Flake8 checks
- **Type Hints**: Use type annotations where applicable

## Future Enhancements & Roadmap

### Short-term Goals (v1.1)

- [ ] **Transfer Learning**: Integration with ResNet, VGG, and EfficientNet
- [ ] **Model Optimization**: TensorRT optimization for edge deployment
- [ ] **Data Augmentation**: Advanced augmentation strategies
- [ ] **API Endpoints**: RESTful API for web integration
- [ ] **Docker Support**: Containerized deployment

### Medium-term Goals (v2.0)

- [ ] **Real-time Video**: Live video stream classification
- [ ] **3D Object Detection**: Integration with depth sensors
- [ ] **Multi-modal Input**: Sonar and camera fusion
- [ ] **Edge Deployment**: Raspberry Pi and Jetson Nano support
- [ ] **Web Interface**: Browser-based classification tool

### Long-term Vision (v3.0)

- [ ] **Reinforcement Learning**: Autonomous navigation integration
- [ ] **Federated Learning**: Distributed model training
- [ ] **Mobile Applications**: iOS/Android apps
- [ ] **Cloud Integration**: AWS/Azure deployment
- [ ] **Commercial Licensing**: Industry-ready solutions

## Research Applications

### Academic Research

- **Marine Biology**: Automated species identification and counting
- **Oceanography**: Seafloor mapping and characterization
- **Archaeology**: Underwater cultural heritage documentation
- **Environmental Science**: Pollution monitoring and ecosystem health

### Industrial Applications

- **Offshore Energy**: Pipeline inspection and maintenance
- **Aquaculture**: Fish farming monitoring and optimization
- **Search and Rescue**: Underwater object and person detection
- **Defense**: Naval reconnaissance and surveillance

## License & Legal

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.

### Open Source Commitment

- ✅ **Free for Academic Use**: Research and educational purposes
- ✅ **Commercial Friendly**: Can be used in commercial projects
- ✅ **Modification Rights**: Fork, modify, and distribute
- ✅ **Attribution Required**: Please credit original authors

## Credits & Acknowledgments

### Development Team

- **Lead Developer**: [Akshaye Iyer](https://github.com/akiyer18)
- **Email**: akshaye.iyer@outlook.com
- **Project Type**: Individual Research & Development
- **Domain**: Marine Robotics & Computer Vision

### Technical Acknowledgments

- **TensorFlow Team**: For the exceptional deep learning framework
- **Keras Community**: For high-level neural network APIs
- **OpenCV Contributors**: For computer vision utilities
- **Python Community**: For the amazing ecosystem
- **Marine Research Community**: For domain knowledge and inspiration

### Dataset Acknowledgments

- **Marine Biology Researchers**: For contributing to open marine datasets
- **Underwater Photography Community**: For high-quality training images
- **Ocean Conservation Organizations**: For supporting marine AI research
- **Academic Institutions**: For promoting open science initiatives

## Contact & Support

### Getting Help

- **📧 Email**: [akshaye.iyer@outlook.com](mailto:akshaye.iyer@outlook.com)
- **🐙 GitHub**: [akiyer18](https://github.com/akiyer18)
- **🐛 Issues**: [Report bugs and request features](https://github.com/akiyer18/Ocean-object_classification/issues)
- **💬 Discussions**: Use GitHub Discussions for questions

### Project Links

- **📦 Repository**: [Ocean Objects Classification](https://github.com/akiyer18/Ocean-object_classification)
- **📚 Documentation**: [Complete API Reference](docs/API_REFERENCE.md)
- **📈 Releases**: [Version History](https://github.com/akiyer18/Ocean-object_classification/releases)
- **🔄 Changelog**: [View Changes](CHANGELOG.md)

---

*🌊 Advancing Marine Robotics through Deep Learning - Built with passion for ocean exploration and AI innovation*

**[⭐ Star this project](https://github.com/akiyer18/Ocean-object_classification) if you find it useful!**
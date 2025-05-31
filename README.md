# Ocean Objects Classification

A deep learning project for classifying underwater objects using Convolutional Neural Networks (CNN) with TensorFlow/Keras. This project is designed for Autonomous Underwater Vehicle (AUV) applications to identify and classify various marine objects and creatures.

## 🌊 Project Overview

This project implements a CNN-based image classification system capable of identifying 10 different categories of ocean objects:

- **Coral Reef** - Various coral formations and reef structures
- **Diver** - Human divers underwater  
- **Fish** - Various fish species
- **Jellyfish** - Different jellyfish types
- **Plant** - Marine vegetation and seaweed
- **Robot** - Underwater robots and equipment
- **Ruins** - Underwater archaeological structures
- **Shark** - Shark species
- **Starfish** - Various starfish species
- **Turtle** - Sea turtles

## 🏗️ Project Structure

```
Ocean-object_classification/
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
├── .gitignore                 # Git ignore rules
├── AUV_FP.ipynb              # Main Jupyter notebook with CNN implementation
├── requirements.txt           # Python dependencies (to be created)
├── data/                      # Data directory
│   ├── train/                 # Training datasets (RAR files)
│   │   ├── coral_reef.rar
│   │   ├── divers.rar
│   │   ├── fish.rar
│   │   ├── jellyfish.rar
│   │   ├── plant.rar
│   │   ├── robots.rar
│   │   ├── ruins.rar
│   │   ├── shark.rar
│   │   ├── starfish.rar
│   │   ├── turtle.rar
│   │   └── readme.md
│   ├── test/                  # Test datasets (RAR files)
│   │   ├── coral_reef.rar
│   │   ├── diver.rar
│   │   ├── fish.rar
│   │   ├── jellyfish.rar
│   │   ├── plant.rar
│   │   ├── robots.rar
│   │   ├── ruins.rar
│   │   ├── shark.rar
│   │   ├── starfish.rar
│   │   ├── turtle.rar
│   │   └── readMe.md
│   └── test-colour/           # Color-specific test data
│       ├── test_col.rar
│       └── readMe.md
├── src/                       # Source code (to be created)
│   ├── __init__.py
│   ├── model.py              # CNN model definition
│   ├── data_preprocessing.py  # Data preprocessing utilities
│   ├── train.py              # Training script
│   └── predict.py            # Prediction utilities
├── models/                    # Saved models directory (to be created)
├── notebooks/                 # Additional notebooks (to be created)
└── docs/                      # Documentation (to be created)
```

## 🚀 Features

- **Deep CNN Architecture**: Multi-layer convolutional neural network optimized for underwater image classification
- **10-Class Classification**: Comprehensive marine object recognition
- **Data Augmentation**: Image preprocessing and augmentation for better model generalization
- **Early Stopping**: Automatic training termination at 99% accuracy
- **GPU Support**: Optimized for GPU acceleration
- **Modular Design**: Well-structured codebase for easy maintenance and extension

## 📋 Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- Jupyter Notebook
- OpenCV (optional, for advanced image processing)

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/akiyer18/Ocean-object_classification.git
   cd Ocean-object_classification
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv ocean_classifier_env
   source ocean_classifier_env/bin/activate  # On Windows: ocean_classifier_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Extract dataset:**
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
   ```

## 🎯 Usage

### Training the Model

1. **Open the Jupyter notebook:**
   ```bash
   jupyter notebook AUV_FP.ipynb
   ```

2. **Run all cells** to train the model from scratch

3. **Monitor training progress** - the model will automatically stop when reaching 99% accuracy

### Model Architecture

The CNN architecture includes:
- **Input Layer**: 150x150x3 (RGB images)
- **Convolutional Layers**: Multiple Conv2D layers with ReLU activation
- **Pooling Layers**: MaxPooling2D for spatial dimension reduction
- **Dense Layers**: Fully connected layers for classification
- **Output Layer**: 10 neurons with softmax activation for multi-class classification

### Key Hyperparameters

- **Image Size**: 150x150 pixels
- **Batch Size**: 64
- **Optimizer**: RMSprop (learning rate: 0.001)
- **Loss Function**: Categorical Crossentropy
- **Epochs**: Up to 100 (with early stopping)

## 📊 Dataset Information

- **Total Training Images**: 4,109 images across 10 classes
- **Image Format**: Various formats (extracted from RAR archives)
- **Class Distribution**: Balanced across marine object categories
- **Data Preprocessing**: 
  - Rescaling to [0,1] range
  - Resizing to 150x150 pixels
  - Data shuffling for training

## 🎯 Model Performance

The model achieves:
- **Training Accuracy**: Up to 99% (with early stopping)
- **Architecture**: Custom CNN with ~509K parameters
- **Training Time**: Varies based on hardware (GPU recommended)

## 🔮 Future Enhancements

- [ ] Transfer learning with pre-trained models (ResNet, VGG, etc.)
- [ ] Real-time video classification
- [ ] Data augmentation strategies
- [ ] Model deployment for edge devices
- [ ] Web interface for easy interaction
- [ ] API endpoints for integration
- [ ] Mobile app development

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Authors

- **Akshaye Iyer** - *Initial work and major reorganization* - [akiyer18](https://github.com/akiyer18)

## 🙏 Acknowledgments

- TensorFlow and Keras communities
- Marine biology datasets contributors
- Underwater robotics research community
- Google Colab for providing computational resources

## 📞 Contact

For questions and suggestions, please contact:
- **Email**: akshaye.iyer@outlook.com
- **GitHub**: [akiyer18](https://github.com/akiyer18)
- **Issues**: Open an issue on this repository

---

**Note**: This project is designed for research and educational purposes in marine robotics and computer vision applications.
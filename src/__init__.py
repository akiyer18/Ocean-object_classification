"""
Ocean Objects Classification Package

A deep learning package for classifying underwater objects using CNNs.
"""

__version__ = "1.0.0"
__author__ = "akshaye iyer"
__email__ = "akshaye.iyer@outlook.com"

from .model import OceanObjectsCNN
from .data_preprocessing import DataPreprocessor
from .train import ModelTrainer
from .predict import OceanObjectsPredictor

__all__ = [
    'OceanObjectsCNN',
    'DataPreprocessor', 
    'ModelTrainer',
    'OceanObjectsPredictor'
] 
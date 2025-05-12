"""
Emotion Data Collection Module

This module handles collecting various types of physiological and user-input 
data for emotion analysis through wearable devices and mobile sensors.
"""

from .sensor_manager import SensorManager
from .bioimpedance import BioimpedanceProcessor
from .data_integrator import DataIntegrator
from .user_input import UserInputCollector

__all__ = [
    'SensorManager',
    'BioimpedanceProcessor',
    'DataIntegrator',
    'UserInputCollector',
]

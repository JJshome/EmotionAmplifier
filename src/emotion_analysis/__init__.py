"""
Emotion Analysis Module

This module handles the analysis of emotional data collected from various sources,
using deep learning models to identify emotional states, intensity, and context.
"""

from .emotion_classifier import EmotionClassifier
from .emotion_model import EmotionModel
from .multimodal_processor import MultimodalProcessor
from .personal_model import PersonalEmotionModel

__all__ = [
    'EmotionClassifier',
    'EmotionModel',
    'MultimodalProcessor',
    'PersonalEmotionModel',
]

"""
Emotion Model Module

This module provides a base class for emotion models with standardized interfaces
for processing, analyzing, and representing emotional states.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
import time
from abc import ABC, abstractmethod
from .emotion_classifier import EmotionClassifier, EmotionModel as EmModelType

logger = logging.getLogger(__name__)

class EmotionModel(ABC):
    """
    Abstract base class for emotion models.
    
    Defines a standardized interface for emotion processing and analysis
    to be implemented by specific emotion model classes.
    """
    
    def __init__(self, model_name: str, model_version: str = "1.0.0"):
        """
        Initialize the emotion model.
        
        Args:
            model_name: Name of the model
            model_version: Version string
        """
        self.model_name = model_name
        self.model_version = model_version
        self.last_update_time = time.time()
        self.classifier = EmotionClassifier()
        
        # Statistics tracking
        self.processed_samples = 0
        self.model_stats = {}
        
        logger.info(f"Initialized {model_name} v{model_version}")
    
    @abstractmethod
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data to extract emotion-related features.
        
        Args:
            input_data: Input data from sensors and user input
            
        Returns:
            Dict containing processed features
        """
        pass
    
    @abstractmethod
    def analyze_emotion(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze processed data to determine emotional state.
        
        Args:
            processed_data: Processed input data
            
        Returns:
            Dict containing emotion analysis results
        """
        pass
    
    def get_emotion_representation(self, 
                                   emotion_results: Dict[str, Any],
                                   representation_type: EmModelType = EmModelType.PLUTCHIK) -> Dict[str, Any]:
        """
        Get standardized representation of emotion results.
        
        Args:
            emotion_results: Results from emotion analysis
            representation_type: Desired emotion representation type
            
        Returns:
            Dict containing standardized emotion representation
        """
        # Extract emotion values from results
        emotion_values = {}
        
        if "emotions" in emotion_results:
            emotion_values.update(emotion_results["emotions"])
        
        if "dimensions" in emotion_results:
            emotion_values.update(emotion_results["dimensions"])
        
        # Use classifier to convert to desired representation
        self.classifier.primary_model = representation_type
        representation = self.classifier.classify_emotion(emotion_values)
        
        # Add timestamp
        if "timestamp" in emotion_results:
            representation["timestamp"] = emotion_results["timestamp"]
        
        return representation
    
    def get_emotion_coordinates(self, emotion_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get dimensional coordinates for the emotion results.
        
        Args:
            emotion_results: Results from emotion analysis
            
        Returns:
            Dict containing dimensional coordinates
        """
        # Extract emotion values
        emotion_values = {}
        
        if "emotions" in emotion_results:
            emotion_values.update(emotion_results["emotions"])
        
        if "dimensions" in emotion_results:
            emotion_values.update(emotion_results["dimensions"])
        
        # Get coordinates using classifier
        coordinates = self.classifier.get_emotion_coordinates(emotion_values)
        
        # Add timestamp
        if "timestamp" in emotion_results:
            coordinates["timestamp"] = emotion_results["timestamp"]
        
        return coordinates
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dict containing model information
        """
        return {
            "name": self.model_name,
            "version": self.model_version,
            "last_update": self.last_update_time,
            "processed_samples": self.processed_samples,
            "stats": self.model_stats
        }
    
    def process_and_analyze(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data and perform emotion analysis in one step.
        
        Args:
            input_data: Input data from sensors and user input
            
        Returns:
            Dict containing emotion analysis results
        """
        # Process input data
        processed_data = self.process_input(input_data)
        
        # Analyze emotions
        emotion_results = self.analyze_emotion(processed_data)
        
        # Update statistics
        self.processed_samples += 1
        
        # Add timestamp if not present
        if "timestamp" not in emotion_results:
            emotion_results["timestamp"] = time.time()
        
        return emotion_results

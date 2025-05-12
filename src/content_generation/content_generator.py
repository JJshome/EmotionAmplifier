"""
Content Generator Module

This module provides the base class for content generators that create
personalized emotion-amplifying content based on emotional states.
"""

import numpy as np
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import time
import os
import json
from abc import ABC, abstractmethod
import uuid
from .content_types import (
    ContentType, ContentFormat, ModalityType, 
    get_recommended_formats, get_format_by_name, get_type_for_format
)

logger = logging.getLogger(__name__)

class ContentParameters:
    """Parameters for content generation"""
    
    def __init__(self, 
                 content_format: ContentFormat,
                 emotion_data: Dict[str, Any],
                 user_preferences: Optional[Dict[str, Any]] = None,
                 content_settings: Optional[Dict[str, Any]] = None):
        """
        Initialize content parameters.
        
        Args:
            content_format: Desired content format
            emotion_data: Emotion analysis data
            user_preferences: Optional user preferences
            content_settings: Optional additional content settings
        """
        self.content_format = content_format
        self.content_type = get_type_for_format(content_format)
        self.emotion_data = emotion_data
        self.user_preferences = user_preferences or {}
        self.content_settings = content_settings or {}
        
        # Extract key emotional parameters for easy access
        self.extract_emotion_parameters()
    
    def extract_emotion_parameters(self) -> None:
        """Extract key emotional parameters from emotion data."""
        # Initialize with default neutral values
        self.valence = 0.0
        self.arousal = 0.0
        self.dominant_emotion = "neutral"
        self.emotion_intensity = 0.5
        self.emotional_quadrant = "neutral"
        
        # Extract dimensional values if available
        if "dimensions" in self.emotion_data:
            dimensions = self.emotion_data["dimensions"]
            if "valence" in dimensions:
                self.valence = dimensions["valence"]
            if "arousal" in dimensions:
                self.arousal = dimensions["arousal"]
        
        # Extract from coordinates if available
        if "dimensional" in self.emotion_data:
            dimensional = self.emotion_data["dimensional"]
            if "valence" in dimensional:
                self.valence = dimensional["valence"]
            if "arousal" in dimensional:
                self.arousal = dimensional["arousal"]
        
        # Extract dominant emotion if available
        if "dominant_emotion" in self.emotion_data:
            self.dominant_emotion = self.emotion_data["dominant_emotion"]
        
        # Extract intensity if available
        if "intensity" in self.emotion_data:
            self.emotion_intensity = self.emotion_data["intensity"]
        elif "confidence" in self.emotion_data:
            self.emotion_intensity = self.emotion_data["confidence"]
        
        # Extract or determine quadrant
        if "quadrant" in self.emotion_data:
            self.emotional_quadrant = self.emotion_data["quadrant"]
        else:
            # Determine quadrant from valence and arousal
            if self.valence >= 0 and self.arousal >= 0:
                self.emotional_quadrant = "happy-excited"
            elif self.valence >= 0 and self.arousal < 0:
                self.emotional_quadrant = "relaxed-content"
            elif self.valence < 0 and self.arousal >= 0:
                self.emotional_quadrant = "angry-stressed"
            else:
                self.emotional_quadrant = "sad-depressed"
    
    def get_parameter(self, 
                      key: str, 
                      default: Any = None) -> Any:
        """
        Get a parameter value from the available data.
        
        Checks user preferences first, then content settings, then returns default.
        
        Args:
            key: Parameter key
            default: Default value if not found
            
        Returns:
            Parameter value
        """
        # Check user preferences first
        if key in self.user_preferences:
            return self.user_preferences[key]
        
        # Then check content settings
        if key in self.content_settings:
            return self.content_settings[key]
        
        # Finally return default
        return default
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert parameters to a dictionary.
        
        Returns:
            Dict representation of parameters
        """
        return {
            "content_format": self.content_format.value,
            "content_type": self.content_type.value,
            "valence": self.valence,
            "arousal": self.arousal,
            "dominant_emotion": self.dominant_emotion,
            "emotion_intensity": self.emotion_intensity,
            "emotional_quadrant": self.emotional_quadrant,
            "user_preferences": self.user_preferences,
            "content_settings": self.content_settings
        }


class ContentGenerator(ABC):
    """
    Abstract base class for content generators.
    
    Defines the interface for generating personalized emotion-amplifying
    content based on emotional states and user preferences.
    """
    
    def __init__(self, 
                 generator_name: str,
                 supported_formats: List[ContentFormat],
                 model_path: Optional[str] = None,
                 use_gpu: bool = False):
        """
        Initialize the content generator.
        
        Args:
            generator_name: Name of the generator
            supported_formats: List of supported content formats
            model_path: Optional path to model files
            use_gpu: Whether to use GPU for generation
        """
        self.generator_name = generator_name
        self.supported_formats = supported_formats
        self.model_path = model_path
        self.use_gpu = use_gpu
        
        # Generation history
        self.generation_history = []
        self.generation_count = 0
        
        # Load models
        self.models_loaded = False
        if model_path:
            self.load_models()
        
        logger.info(f"Initialized {generator_name} content generator")
    
    @abstractmethod
    def load_models(self) -> bool:
        """
        Load generative models.
        
        Returns:
            bool: True if models were loaded successfully
        """
        pass
    
    @abstractmethod
    def generate_content(self, 
                        params: ContentParameters) -> Dict[str, Any]:
        """
        Generate content based on parameters.
        
        Args:
            params: Content generation parameters
            
        Returns:
            Dict containing generated content
        """
        pass
    
    def supports_format(self, 
                       content_format: Union[str, ContentFormat]) -> bool:
        """
        Check if the generator supports a content format.
        
        Args:
            content_format: Content format to check
            
        Returns:
            bool: True if format is supported
        """
        if isinstance(content_format, str):
            format_enum = get_format_by_name(content_format)
            if not format_enum:
                return False
            return format_enum in self.supported_formats
        
        return content_format in self.supported_formats
    
    def recommend_content(self, 
                         emotion_data: Dict[str, Any],
                         user_preferences: Optional[Dict[str, Any]] = None) -> List[ContentFormat]:
        """
        Recommend suitable content formats based on emotion data.
        
        Args:
            emotion_data: Emotion analysis data
            user_preferences: Optional user preferences
            
        Returns:
            List of recommended content formats
        """
        # Get all recommended formats
        all_formats = get_recommended_formats(emotion_data)
        
        # Filter to only supported formats
        supported_recommended = [fmt for fmt in all_formats if fmt in self.supported_formats]
        
        # If there are user preferences, prioritize those formats
        if user_preferences and "preferred_formats" in user_preferences:
            preferred = user_preferences["preferred_formats"]
            
            # Convert string preferences to enum values
            preferred_enums = []
            for pref in preferred:
                if isinstance(pref, str):
                    format_enum = get_format_by_name(pref)
                    if format_enum:
                        preferred_enums.append(format_enum)
                else:
                    preferred_enums.append(pref)
            
            # Filter to supported and preferred formats
            preferred_supported = [fmt for fmt in preferred_enums if fmt in self.supported_formats]
            
            # If we have any preferred formats, return those
            if preferred_supported:
                return preferred_supported
        
        # If no preferences or no supported preferences, return recommended formats
        return supported_recommended if supported_recommended else [self.supported_formats[0]]
    
    def create_content_object(self, 
                            content_data: Any,
                            content_format: ContentFormat,
                            params: ContentParameters,
                            metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a standardized content object.
        
        Args:
            content_data: The generated content data
            content_format: Format of the content
            params: Parameters used for generation
            metadata: Optional additional metadata
            
        Returns:
            Dict containing structured content object
        """
        # Create a unique ID for the content
        content_id = str(uuid.uuid4())
        
        # Content object structure
        content_object = {
            "content_id": content_id,
            "content_format": content_format.value,
            "content_type": get_type_for_format(content_format).value,
            "content_data": content_data,
            "timestamp": time.time(),
            "generator": self.generator_name,
            "emotion_context": {
                "valence": params.valence,
                "arousal": params.arousal,
                "dominant_emotion": params.dominant_emotion,
                "quadrant": params.emotional_quadrant,
                "intensity": params.emotion_intensity
            }
        }
        
        # Add metadata if provided
        if metadata:
            content_object["metadata"] = metadata
        
        # Add to generation history
        self.generation_history.append({
            "content_id": content_id,
            "content_format": content_format.value,
            "timestamp": content_object["timestamp"],
            "emotion_context": content_object["emotion_context"]
        })
        
        self.generation_count += 1
        
        return content_object
    
    def generate_for_emotion(self, 
                            emotion_data: Dict[str, Any],
                            content_format: Optional[Union[str, ContentFormat]] = None,
                            user_preferences: Optional[Dict[str, Any]] = None,
                            content_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate content for a given emotional state.
        
        This is the main entry point for content generation.
        
        Args:
            emotion_data: Emotion analysis data
            content_format: Optional specific content format to generate
            user_preferences: Optional user preferences
            content_settings: Optional additional content settings
            
        Returns:
            Dict containing generated content
        """
        # If content format is specified, convert string to enum if needed
        if content_format:
            if isinstance(content_format, str):
                format_enum = get_format_by_name(content_format)
                if not format_enum:
                    logger.error(f"Unknown content format: {content_format}")
                    return {"error": f"Unknown content format: {content_format}"}
                if not self.supports_format(format_enum):
                    logger.error(f"Unsupported content format: {format_enum.value}")
                    return {"error": f"Unsupported content format: {format_enum.value}"}
                content_format = format_enum
            elif not self.supports_format(content_format):
                logger.error(f"Unsupported content format: {content_format.value}")
                return {"error": f"Unsupported content format: {content_format.value}"}
        else:
            # Recommend content format based on emotion
            recommended = self.recommend_content(emotion_data, user_preferences)
            if not recommended:
                logger.error("No suitable content format could be recommended")
                return {"error": "No suitable content format could be recommended"}
            content_format = recommended[0]
        
        # Create content parameters
        params = ContentParameters(
            content_format=content_format,
            emotion_data=emotion_data,
            user_preferences=user_preferences,
            content_settings=content_settings
        )
        
        # Generate content
        try:
            logger.info(f"Generating {content_format.value} content for emotion: {params.dominant_emotion}")
            content = self.generate_content(params)
            return content
        except Exception as e:
            logger.error(f"Error generating content: {str(e)}")
            return {"error": f"Content generation failed: {str(e)}"}
    
    def save_generation_history(self, filepath: str) -> bool:
        """
        Save generation history to a file.
        
        Args:
            filepath: Path to save the history
            
        Returns:
            bool: True if history was saved successfully
        """
        try:
            with open(filepath, 'w') as f:
                json.dump(self.generation_history, f, indent=2)
            logger.info(f"Saved generation history to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving generation history: {str(e)}")
            return False
    
    def get_generator_info(self) -> Dict[str, Any]:
        """
        Get information about the generator.
        
        Returns:
            Dict containing generator information
        """
        return {
            "name": self.generator_name,
            "supported_formats": [fmt.value for fmt in self.supported_formats],
            "models_loaded": self.models_loaded,
            "generation_count": self.generation_count,
            "uses_gpu": self.use_gpu
        }

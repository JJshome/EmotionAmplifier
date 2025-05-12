"""
Emotion Classifier Module

This module provides a standardized interface for emotion classification
based on various emotional models and taxonomies.
"""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionModel(Enum):
    """Enum for supported emotion models"""
    BASIC = "basic"            # Basic emotions (e.g., Ekman's six)
    PLUTCHIK = "plutchik"      # Plutchik's wheel of emotions
    PAD = "pad"                # Pleasure-Arousal-Dominance model
    DISCRETE = "discrete"      # Discrete emotions model
    DIMENSIONAL = "dimensional"  # Dimensional model (valence-arousal)
    APPRAISAL = "appraisal"    # Appraisal model


class EmotionClassifier:
    """
    Classifies emotions using various emotion models and taxonomies.
    
    Provides a standardized way to categorize emotional states and
    convert between different emotion representation systems.
    """
    
    def __init__(self, 
                primary_model: EmotionModel = EmotionModel.PLUTCHIK,
                secondary_model: Optional[EmotionModel] = EmotionModel.PAD):
        """
        Initialize the emotion classifier.
        
        Args:
            primary_model: Primary emotion model to use
            secondary_model: Optional secondary model for dual representation
        """
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        
        # Define emotion categories for each model
        self.model_categories = {
            EmotionModel.BASIC: ["happiness", "sadness", "anger", "fear", "disgust", "surprise"],
            EmotionModel.PLUTCHIK: [
                "joy", "trust", "fear", "surprise", 
                "sadness", "disgust", "anger", "anticipation"
            ],
            EmotionModel.DISCRETE: [
                "joy", "love", "surprise", "anger", "sadness", "fear",
                "shame", "guilt", "envy", "jealousy", "disgust", "contempt",
                "pride", "gratitude", "awe", "interest", "amusement", "compassion"
            ]
        }
        
        # Define emotion dimension axes
        self.pad_dimensions = ["pleasure", "arousal", "dominance"]
        self.dimensional_axes = ["valence", "arousal"]
        
        # Define emotion mappings between models
        self._init_emotion_mappings()
        
        logger.info(f"Initialized emotion classifier with {primary_model.value} model")
    
    def _init_emotion_mappings(self) -> None:
        """Initialize mappings between different emotion models."""
        # Mapping from Plutchik to Basic emotions
        self.plutchik_to_basic = {
            "joy": "happiness",
            "trust": None,
            "fear": "fear",
            "surprise": "surprise",
            "sadness": "sadness",
            "disgust": "disgust",
            "anger": "anger",
            "anticipation": None
        }
        
        # Mapping from Basic to PAD space (approximate values)
        self.basic_to_pad = {
            "happiness": (0.8, 0.5, 0.6),    # High pleasure, moderate arousal, moderate dominance
            "sadness": (-0.7, -0.5, -0.5),   # Low pleasure, low arousal, low dominance
            "anger": (-0.6, 0.8, 0.7),       # Low pleasure, high arousal, high dominance
            "fear": (-0.8, 0.8, -0.7),       # Low pleasure, high arousal, low dominance
            "disgust": (-0.6, 0.2, 0.0),     # Low pleasure, moderate arousal, neutral dominance
            "surprise": (0.1, 0.8, -0.2)     # Neutral pleasure, high arousal, slight low dominance
        }
        
        # Mapping from Plutchik to PAD space (approximate values)
        self.plutchik_to_pad = {
            "joy": (0.8, 0.5, 0.6),
            "trust": (0.6, -0.2, 0.5),
            "fear": (-0.8, 0.8, -0.7),
            "surprise": (0.1, 0.8, -0.2),
            "sadness": (-0.7, -0.5, -0.5),
            "disgust": (-0.6, 0.2, 0.0),
            "anger": (-0.6, 0.8, 0.7),
            "anticipation": (0.5, 0.3, 0.3)
        }
        
        # Mapping from PAD to dimension coordinates (valence-arousal)
        # This is a simplified mapping that excludes dominance
        self.pad_to_dimensional = lambda p, a, d: (p, a)
    
    def classify_emotion(self, 
                        emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify emotions using the primary model.
        
        Args:
            emotion_values: Dictionary of emotion categories and their intensities
            
        Returns:
            Dict containing classified emotions
        """
        if self.primary_model == EmotionModel.BASIC:
            return self._classify_basic(emotion_values)
        elif self.primary_model == EmotionModel.PLUTCHIK:
            return self._classify_plutchik(emotion_values)
        elif self.primary_model == EmotionModel.DISCRETE:
            return self._classify_discrete(emotion_values)
        elif self.primary_model == EmotionModel.PAD:
            return self._classify_pad(emotion_values)
        elif self.primary_model == EmotionModel.DIMENSIONAL:
            return self._classify_dimensional(emotion_values)
        else:
            logger.warning(f"Unsupported primary model: {self.primary_model}")
            return {"error": f"Unsupported model: {self.primary_model}"}
    
    def get_emotion_coordinates(self, 
                              emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Get emotion coordinates in dimensional space.
        
        Args:
            emotion_values: Dictionary of emotion categories and their intensities
            
        Returns:
            Dict containing emotion coordinates in various dimensional spaces
        """
        result = {}
        
        # Convert to PAD coordinates
        pad_coordinates = self._map_to_pad(emotion_values)
        if pad_coordinates:
            result["pad"] = {
                "pleasure": pad_coordinates[0],
                "arousal": pad_coordinates[1],
                "dominance": pad_coordinates[2]
            }
        
        # Convert to valence-arousal coordinates
        if pad_coordinates:
            valence, arousal = self.pad_to_dimensional(*pad_coordinates)
            result["dimensional"] = {
                "valence": valence,
                "arousal": arousal
            }
            
            # Add quadrant information
            result["quadrant"] = self._get_emotion_quadrant(valence, arousal)
        
        # Add intensity (magnitude in the space)
        if "pad" in result:
            pad = result["pad"]
            intensity = np.sqrt(pad["pleasure"]**2 + pad["arousal"]**2 + pad["dominance"]**2) / np.sqrt(3)
            result["intensity"] = intensity
        
        return result
    
    def _map_to_pad(self, 
                   emotion_values: Dict[str, float]) -> Optional[Tuple[float, float, float]]:
        """
        Map emotion values to PAD coordinates.
        
        Args:
            emotion_values: Dictionary of emotion categories and their intensities
            
        Returns:
            Tuple of (pleasure, arousal, dominance) coordinates or None if mapping failed
        """
        # Direct PAD values if provided
        if "pleasure" in emotion_values and "arousal" in emotion_values and "dominance" in emotion_values:
            return (
                emotion_values["pleasure"],
                emotion_values["arousal"],
                emotion_values["dominance"]
            )
        
        # Valence-arousal if provided
        if "valence" in emotion_values and "arousal" in emotion_values:
            # Set a neutral dominance since it's not specified
            return (
                emotion_values["valence"],
                emotion_values["arousal"],
                0.0
            )
        
        # Map from categorical emotions
        pad_coords = [0.0, 0.0, 0.0]
        total_weight = 0.0
        
        # Try mapping from basic emotions
        for emotion, intensity in emotion_values.items():
            if emotion in self.basic_to_pad:
                p, a, d = self.basic_to_pad[emotion]
                pad_coords[0] += p * intensity
                pad_coords[1] += a * intensity
                pad_coords[2] += d * intensity
                total_weight += intensity
            elif emotion in self.plutchik_to_pad:
                p, a, d = self.plutchik_to_pad[emotion]
                pad_coords[0] += p * intensity
                pad_coords[1] += a * intensity
                pad_coords[2] += d * intensity
                total_weight += intensity
        
        # If we have valid mappings, normalize by total weight
        if total_weight > 0:
            pad_coords = [c / total_weight for c in pad_coords]
            return tuple(pad_coords)
        
        return None
    
    def _classify_basic(self, 
                       emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify using the basic emotions model.
        
        Args:
            emotion_values: Dictionary of emotion values
            
        Returns:
            Dict containing classified emotions in basic model
        """
        # Initialize all basic emotions with zero intensity
        basic_emotions = {emotion: 0.0 for emotion in self.model_categories[EmotionModel.BASIC]}
        
        # Map input emotions to basic emotions
        for emotion, intensity in emotion_values.items():
            # Direct mapping if emotion is in basic emotions
            if emotion in basic_emotions:
                basic_emotions[emotion] = intensity
            # Map from Plutchik if possible
            elif emotion in self.plutchik_to_basic and self.plutchik_to_basic[emotion] is not None:
                basic_emotion = self.plutchik_to_basic[emotion]
                basic_emotions[basic_emotion] = max(basic_emotions[basic_emotion], intensity)
        
        # Find the dominant emotion
        dominant_emotion = max(basic_emotions, key=basic_emotions.get)
        
        return {
            "model": EmotionModel.BASIC.value,
            "emotions": basic_emotions,
            "dominant_emotion": dominant_emotion
        }
    
    def _classify_plutchik(self, 
                          emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify using Plutchik's wheel of emotions.
        
        Args:
            emotion_values: Dictionary of emotion values
            
        Returns:
            Dict containing classified emotions in Plutchik model
        """
        # Initialize all Plutchik emotions with zero intensity
        plutchik_emotions = {emotion: 0.0 for emotion in self.model_categories[EmotionModel.PLUTCHIK]}
        
        # Map input emotions to Plutchik emotions
        for emotion, intensity in emotion_values.items():
            # Direct mapping if emotion is in Plutchik emotions
            if emotion in plutchik_emotions:
                plutchik_emotions[emotion] = intensity
            # Map from basic if needed (would need a reverse mapping)
        
        # Find the dominant emotion
        dominant_emotion = max(plutchik_emotions, key=plutchik_emotions.get)
        
        # Identify emotion pairs (opposites in Plutchik's wheel)
        emotion_pairs = {
            "joy": "sadness",
            "trust": "disgust",
            "fear": "anger",
            "surprise": "anticipation"
        }
        
        # Calculate emotional dyads (primary combinations)
        dyads = {}
        for emotion1, emotion2 in [
            ("joy", "trust"),     # love
            ("trust", "fear"),    # submission
            ("fear", "surprise"), # awe
            ("surprise", "sadness"), # disappointment
            ("sadness", "disgust"), # remorse
            ("disgust", "anger"), # contempt
            ("anger", "anticipation"), # aggressiveness
            ("anticipation", "joy") # optimism
        ]:
            # The dyad intensity is the minimum of the two emotions
            dyad_intensity = min(plutchik_emotions[emotion1], plutchik_emotions[emotion2])
            dyad_name = f"{emotion1}-{emotion2}"
            dyads[dyad_name] = dyad_intensity
        
        return {
            "model": EmotionModel.PLUTCHIK.value,
            "emotions": plutchik_emotions,
            "dominant_emotion": dominant_emotion,
            "dyads": dyads
        }
    
    def _classify_discrete(self, 
                          emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify using a discrete emotions model with many categories.
        
        Args:
            emotion_values: Dictionary of emotion values
            
        Returns:
            Dict containing classified discrete emotions
        """
        # Initialize all discrete emotions with zero intensity
        discrete_emotions = {emotion: 0.0 for emotion in self.model_categories[EmotionModel.DISCRETE]}
        
        # Map input emotions to discrete emotions
        for emotion, intensity in emotion_values.items():
            # Direct mapping if emotion is in discrete emotions
            if emotion in discrete_emotions:
                discrete_emotions[emotion] = intensity
        
        # Find the dominant emotion
        dominant_emotion = max(discrete_emotions, key=discrete_emotions.get)
        
        return {
            "model": EmotionModel.DISCRETE.value,
            "emotions": discrete_emotions,
            "dominant_emotion": dominant_emotion
        }
    
    def _classify_pad(self, 
                     emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify using the PAD (Pleasure-Arousal-Dominance) model.
        
        Args:
            emotion_values: Dictionary of emotion values
            
        Returns:
            Dict containing PAD classification
        """
        # Get PAD coordinates
        pad_coords = self._map_to_pad(emotion_values)
        
        if pad_coords is None:
            logger.warning("Could not map emotions to PAD space")
            return {"error": "Could not map emotions to PAD space"}
        
        pleasure, arousal, dominance = pad_coords
        
        # Classify emotional regions in PAD space
        regions = []
        
        if pleasure > 0.3:
            if arousal > 0.3:
                if dominance > 0.3:
                    regions.append("exuberant")
                elif dominance < -0.3:
                    regions.append("dependent")
                else:
                    regions.append("pleased")
            elif arousal < -0.3:
                if dominance > 0.3:
                    regions.append("relaxed")
                elif dominance < -0.3:
                    regions.append("docile")
                else:
                    regions.append("peaceful")
            else:
                regions.append("pleasant")
        elif pleasure < -0.3:
            if arousal > 0.3:
                if dominance > 0.3:
                    regions.append("hostile")
                elif dominance < -0.3:
                    regions.append("anxious")
                else:
                    regions.append("distressed")
            elif arousal < -0.3:
                if dominance > 0.3:
                    regions.append("disdainful")
                elif dominance < -0.3:
                    regions.append("bored")
                else:
                    regions.append("gloomy")
            else:
                regions.append("unpleasant")
        
        return {
            "model": EmotionModel.PAD.value,
            "pleasure": pleasure,
            "arousal": arousal,
            "dominance": dominance,
            "regions": regions
        }
    
    def _classify_dimensional(self, 
                             emotion_values: Dict[str, float]) -> Dict[str, Any]:
        """
        Classify using the dimensional (valence-arousal) model.
        
        Args:
            emotion_values: Dictionary of emotion values
            
        Returns:
            Dict containing dimensional classification
        """
        # Try to get valence and arousal directly
        if "valence" in emotion_values and "arousal" in emotion_values:
            valence = emotion_values["valence"]
            arousal = emotion_values["arousal"]
        else:
            # Map to PAD and then to valence-arousal
            pad_coords = self._map_to_pad(emotion_values)
            
            if pad_coords is None:
                logger.warning("Could not map emotions to dimensional space")
                return {"error": "Could not map emotions to dimensional space"}
            
            valence, arousal = self.pad_to_dimensional(*pad_coords)
        
        # Determine the emotion quadrant
        quadrant = self._get_emotion_quadrant(valence, arousal)
        
        # Calculate intensity (distance from origin)
        intensity = np.sqrt(valence**2 + arousal**2) / np.sqrt(2)
        
        return {
            "model": EmotionModel.DIMENSIONAL.value,
            "valence": valence,
            "arousal": arousal,
            "quadrant": quadrant,
            "intensity": intensity
        }
    
    def _get_emotion_quadrant(self, valence: float, arousal: float) -> str:
        """
        Determine the emotion quadrant based on valence and arousal.
        
        Args:
            valence: Valence value (-1 to 1)
            arousal: Arousal value (-1 to 1)
            
        Returns:
            String describing the emotion quadrant
        """
        if valence >= 0 and arousal >= 0:
            return "happy-excited"
        elif valence >= 0 and arousal < 0:
            return "relaxed-content"
        elif valence < 0 and arousal >= 0:
            return "angry-stressed"
        else:
            return "sad-depressed"
    
    def get_model_categories(self, model: Optional[EmotionModel] = None) -> List[str]:
        """
        Get the list of emotion categories for a given model.
        
        Args:
            model: Emotion model to get categories for (default: primary model)
            
        Returns:
            List of emotion category names
        """
        if model is None:
            model = self.primary_model
        
        if model in self.model_categories:
            return self.model_categories[model]
        
        return []

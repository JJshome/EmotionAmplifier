"""
User Input Collection Module

This module handles the collection of user-provided data about their emotional state,
preferences, and goals to complement the physiological sensor data.
"""

import time
from typing import Dict, List, Optional, Any, Union
import logging
import json
from enum import Enum

logger = logging.getLogger(__name__)

class EmotionScale(Enum):
    """Enum for different emotion rating scales"""
    BINARY = "binary"  # Yes/No
    LIKERT_5 = "likert_5"  # 1-5 scale
    LIKERT_7 = "likert_7"  # 1-7 scale
    PERCENTAGE = "percentage"  # 0-100%
    PLUTCHIK = "plutchik"  # Plutchik's wheel of emotions
    PAD = "pad"  # Pleasure, Arousal, Dominance model
    CUSTOM = "custom"  # Custom scale


class UserInputCollector:
    """
    Collects and processes user input about emotional states and preferences.
    
    Handles various forms of user-provided data, including:
    - Self-reported emotional states
    - Emotional goals and preferences
    - Content preferences for emotion amplification
    - Feedback on generated content
    """
    
    def __init__(self, storage_enabled: bool = True):
        """
        Initialize the user input collector.
        
        Args:
            storage_enabled: Whether to store input history
        """
        self.storage_enabled = storage_enabled
        self.input_history = [] if storage_enabled else None
        self.current_emotion_ratings = {}
        self.emotion_scale = EmotionScale.PERCENTAGE
        self.preferences = {}
        self.emotional_goals = {}
        self.content_feedback = {}
        
        # Default emotion set
        self.emotion_set = [
            "joy", "trust", "fear", "surprise", 
            "sadness", "disgust", "anger", "anticipation"
        ]
        
        logger.info("User input collector initialized")
    
    def set_emotion_scale(self, scale: EmotionScale) -> None:
        """
        Set the emotion rating scale.
        
        Args:
            scale: Emotion rating scale
        """
        self.emotion_scale = scale
        logger.info(f"Set emotion scale to {scale.value}")
    
    def set_emotion_set(self, emotion_set: List[str]) -> None:
        """
        Set the list of emotions to track.
        
        Args:
            emotion_set: List of emotion names
        """
        self.emotion_set = emotion_set
        self.current_emotion_ratings = {emotion: 0 for emotion in emotion_set}
        logger.info(f"Set emotion set to {emotion_set}")
    
    def record_emotion_rating(self, 
                             emotion: str, 
                             rating: Union[float, int, bool], 
                             timestamp: Optional[float] = None) -> bool:
        """
        Record a user-provided emotion rating.
        
        Args:
            emotion: Name of the emotion
            rating: Rating value
            timestamp: Optional timestamp (default is current time)
            
        Returns:
            bool: True if rating was recorded successfully
        """
        if emotion not in self.emotion_set:
            logger.warning(f"Unknown emotion: {emotion}")
            return False
        
        if timestamp is None:
            timestamp = time.time()
        
        # Validate rating based on scale
        validated_rating = self._validate_rating(rating, self.emotion_scale)
        if validated_rating is None:
            logger.warning(f"Invalid rating: {rating} for scale {self.emotion_scale.value}")
            return False
        
        # Record the rating
        self.current_emotion_ratings[emotion] = validated_rating
        
        # Add to history if storage is enabled
        if self.storage_enabled and self.input_history is not None:
            entry = {
                "timestamp": timestamp,
                "type": "emotion_rating",
                "emotion": emotion,
                "rating": validated_rating,
                "scale": self.emotion_scale.value
            }
            self.input_history.append(entry)
        
        logger.debug(f"Recorded {emotion} rating: {validated_rating}")
        return True
    
    def record_pad_values(self, 
                         pleasure: float, 
                         arousal: float, 
                         dominance: float,
                         timestamp: Optional[float] = None) -> bool:
        """
        Record PAD (Pleasure-Arousal-Dominance) values.
        
        Args:
            pleasure: Pleasure value (-1.0 to 1.0)
            arousal: Arousal value (-1.0 to 1.0)
            dominance: Dominance value (-1.0 to 1.0)
            timestamp: Optional timestamp (default is current time)
            
        Returns:
            bool: True if values were recorded successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Validate PAD values
        pleasure = max(-1.0, min(1.0, pleasure))
        arousal = max(-1.0, min(1.0, arousal))
        dominance = max(-1.0, min(1.0, dominance))
        
        # Record the values
        pad_values = {
            "pleasure": pleasure,
            "arousal": arousal,
            "dominance": dominance
        }
        
        # Add to history if storage is enabled
        if self.storage_enabled and self.input_history is not None:
            entry = {
                "timestamp": timestamp,
                "type": "pad_values",
                "values": pad_values
            }
            self.input_history.append(entry)
        
        logger.debug(f"Recorded PAD values: P={pleasure}, A={arousal}, D={dominance}")
        return True
    
    def record_preference(self, 
                         category: str, 
                         preference_values: Dict[str, Any],
                         timestamp: Optional[float] = None) -> bool:
        """
        Record user preferences for content generation.
        
        Args:
            category: Preference category (e.g., "music", "visuals", "text")
            preference_values: Dictionary of preference values
            timestamp: Optional timestamp (default is current time)
            
        Returns:
            bool: True if preferences were recorded successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Record the preferences
        if category not in self.preferences:
            self.preferences[category] = {}
        
        self.preferences[category].update(preference_values)
        
        # Add to history if storage is enabled
        if self.storage_enabled and self.input_history is not None:
            entry = {
                "timestamp": timestamp,
                "type": "preference",
                "category": category,
                "values": preference_values
            }
            self.input_history.append(entry)
        
        logger.debug(f"Recorded {category} preferences: {preference_values}")
        return True
    
    def record_emotional_goal(self, 
                             goal_type: str, 
                             target_emotions: Dict[str, float],
                             timestamp: Optional[float] = None) -> bool:
        """
        Record user's emotional goals.
        
        Args:
            goal_type: Type of goal (e.g., "enhance", "reduce", "maintain")
            target_emotions: Dictionary mapping emotions to target intensities
            timestamp: Optional timestamp (default is current time)
            
        Returns:
            bool: True if goal was recorded successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Validate target emotions
        valid_targets = {}
        for emotion, intensity in target_emotions.items():
            if emotion in self.emotion_set:
                valid_targets[emotion] = max(0.0, min(1.0, intensity))
            else:
                logger.warning(f"Unknown emotion in goal: {emotion}")
        
        if not valid_targets:
            logger.warning("No valid target emotions specified")
            return False
        
        # Record the goal
        self.emotional_goals[goal_type] = valid_targets
        
        # Add to history if storage is enabled
        if self.storage_enabled and self.input_history is not None:
            entry = {
                "timestamp": timestamp,
                "type": "emotional_goal",
                "goal_type": goal_type,
                "target_emotions": valid_targets
            }
            self.input_history.append(entry)
        
        logger.debug(f"Recorded emotional goal of type {goal_type}: {valid_targets}")
        return True
    
    def record_content_feedback(self, 
                               content_id: str, 
                               rating: float, 
                               feedback_text: Optional[str] = None,
                               timestamp: Optional[float] = None) -> bool:
        """
        Record user feedback on generated content.
        
        Args:
            content_id: Identifier for the content
            rating: Rating value (0.0 to 1.0)
            feedback_text: Optional text feedback
            timestamp: Optional timestamp (default is current time)
            
        Returns:
            bool: True if feedback was recorded successfully
        """
        if timestamp is None:
            timestamp = time.time()
        
        # Validate rating
        rating = max(0.0, min(1.0, rating))
        
        # Record the feedback
        self.content_feedback[content_id] = {
            "rating": rating,
            "feedback_text": feedback_text,
            "timestamp": timestamp
        }
        
        # Add to history if storage is enabled
        if self.storage_enabled and self.input_history is not None:
            entry = {
                "timestamp": timestamp,
                "type": "content_feedback",
                "content_id": content_id,
                "rating": rating,
                "feedback_text": feedback_text
            }
            self.input_history.append(entry)
        
        logger.debug(f"Recorded feedback for content {content_id}: {rating}")
        return True
    
    def get_current_emotion_state(self) -> Dict[str, float]:
        """
        Get the current emotional state based on user ratings.
        
        Returns:
            Dict mapping emotions to intensity values (0.0 to 1.0)
        """
        # For most scales, values are already normalized to 0.0-1.0
        # For others, we need to convert
        normalized_ratings = {}
        
        for emotion, rating in self.current_emotion_ratings.items():
            if self.emotion_scale == EmotionScale.LIKERT_5:
                normalized_ratings[emotion] = rating / 5.0
            elif self.emotion_scale == EmotionScale.LIKERT_7:
                normalized_ratings[emotion] = rating / 7.0
            elif self.emotion_scale == EmotionScale.BINARY:
                normalized_ratings[emotion] = 1.0 if rating else 0.0
            else:
                normalized_ratings[emotion] = rating
        
        return normalized_ratings
    
    def get_emotional_goals(self) -> Dict[str, Dict[str, float]]:
        """
        Get the user's emotional goals.
        
        Returns:
            Dict mapping goal types to target emotions
        """
        return self.emotional_goals
    
    def get_preferences(self, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Get user preferences.
        
        Args:
            category: Optional category to filter preferences
            
        Returns:
            Dict of preferences
        """
        if category is not None:
            return self.preferences.get(category, {})
        return self.preferences
    
    def get_content_effectiveness(self, content_id: str) -> Optional[float]:
        """
        Get the effectiveness rating for a piece of content.
        
        Args:
            content_id: Identifier for the content
            
        Returns:
            Effectiveness rating or None if not available
        """
        if content_id in self.content_feedback:
            return self.content_feedback[content_id]["rating"]
        return None
    
    def get_input_history(self, 
                         input_type: Optional[str] = None, 
                         start_time: Optional[float] = None,
                         end_time: Optional[float] = None) -> List[Dict[str, Any]]:
        """
        Get the history of user inputs.
        
        Args:
            input_type: Optional type of input to filter
            start_time: Optional start time for filtering
            end_time: Optional end time for filtering
            
        Returns:
            List of input history entries
        """
        if not self.storage_enabled or not self.input_history:
            return []
        
        filtered_history = self.input_history
        
        # Filter by type if specified
        if input_type is not None:
            filtered_history = [entry for entry in filtered_history if entry["type"] == input_type]
        
        # Filter by start time if specified
        if start_time is not None:
            filtered_history = [entry for entry in filtered_history if entry["timestamp"] >= start_time]
        
        # Filter by end time if specified
        if end_time is not None:
            filtered_history = [entry for entry in filtered_history if entry["timestamp"] <= end_time]
        
        # Sort by timestamp
        filtered_history.sort(key=lambda x: x["timestamp"])
        
        return filtered_history
    
    def save_input_history(self, filepath: str) -> bool:
        """
        Save the input history to a JSON file.
        
        Args:
            filepath: Path to save the JSON file
            
        Returns:
            bool: True if file was saved successfully
        """
        if not self.storage_enabled or not self.input_history:
            logger.warning("No input history to save")
            return False
        
        try:
            with open(filepath, 'w') as f:
                json.dump(self.input_history, f, indent=2)
            logger.info(f"Saved input history to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving input history to {filepath}: {str(e)}")
            return False
    
    def load_input_history(self, filepath: str) -> bool:
        """
        Load input history from a JSON file.
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            bool: True if file was loaded successfully
        """
        if not self.storage_enabled:
            logger.warning("Cannot load history when storage is disabled")
            return False
        
        try:
            with open(filepath, 'r') as f:
                history = json.load(f)
            
            if not isinstance(history, list):
                logger.error(f"Invalid input history format in {filepath}")
                return False
            
            self.input_history = history
            
            # Update current state based on most recent entries
            self._update_current_state_from_history()
            
            logger.info(f"Loaded input history from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading input history from {filepath}: {str(e)}")
            return False
    
    def _update_current_state_from_history(self) -> None:
        """
        Update the current state based on loaded history.
        """
        if not self.input_history:
            return
        
        # Sort by timestamp to process in chronological order
        sorted_history = sorted(self.input_history, key=lambda x: x["timestamp"])
        
        # Process each entry to update current state
        for entry in sorted_history:
            entry_type = entry.get("type")
            
            if entry_type == "emotion_rating":
                emotion = entry.get("emotion")
                rating = entry.get("rating")
                if emotion in self.emotion_set and rating is not None:
                    self.current_emotion_ratings[emotion] = rating
            
            elif entry_type == "preference":
                category = entry.get("category")
                values = entry.get("values")
                if category and values:
                    if category not in self.preferences:
                        self.preferences[category] = {}
                    self.preferences[category].update(values)
            
            elif entry_type == "emotional_goal":
                goal_type = entry.get("goal_type")
                target_emotions = entry.get("target_emotions")
                if goal_type and target_emotions:
                    self.emotional_goals[goal_type] = target_emotions
            
            elif entry_type == "content_feedback":
                content_id = entry.get("content_id")
                rating = entry.get("rating")
                feedback_text = entry.get("feedback_text")
                timestamp = entry.get("timestamp")
                if content_id and rating is not None:
                    self.content_feedback[content_id] = {
                        "rating": rating,
                        "feedback_text": feedback_text,
                        "timestamp": timestamp
                    }
    
    def _validate_rating(self, 
                        rating: Union[float, int, bool], 
                        scale: EmotionScale) -> Optional[float]:
        """
        Validate and normalize rating based on the scale.
        
        Args:
            rating: Rating value
            scale: Emotion rating scale
            
        Returns:
            Normalized rating value or None if invalid
        """
        if scale == EmotionScale.BINARY:
            if isinstance(rating, bool):
                return 1.0 if rating else 0.0
            elif isinstance(rating, (int, float)):
                return 1.0 if rating > 0 else 0.0
        
        elif scale == EmotionScale.LIKERT_5:
            if isinstance(rating, (int, float)):
                rating_int = int(round(rating))
                if 1 <= rating_int <= 5:
                    return rating_int
        
        elif scale == EmotionScale.LIKERT_7:
            if isinstance(rating, (int, float)):
                rating_int = int(round(rating))
                if 1 <= rating_int <= 7:
                    return rating_int
        
        elif scale == EmotionScale.PERCENTAGE:
            if isinstance(rating, (int, float)):
                return max(0.0, min(1.0, float(rating)))
        
        elif scale == EmotionScale.PAD:
            if isinstance(rating, (int, float)):
                return max(-1.0, min(1.0, float(rating)))
        
        elif scale == EmotionScale.PLUTCHIK:
            if isinstance(rating, (int, float)):
                return max(0.0, min(1.0, float(rating)))
        
        elif scale == EmotionScale.CUSTOM:
            # For custom scale, we just pass through the rating
            if isinstance(rating, (int, float)):
                return float(rating)
        
        # If we get here, the rating is invalid for the given scale
        return None

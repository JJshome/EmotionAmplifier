"""
Personal Emotion Model Module

This module implements personalized emotion models that adapt to individual users
by learning their unique emotional patterns and responses over time.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Dict, List, Optional, Any, Tuple
import os
import json
import time
from .multimodal_processor import MultimodalProcessor, ModalityType

logger = logging.getLogger(__name__)

class PersonalEmotionModel:
    """
    Personalized emotion model that adapts to individual users.
    
    Uses transfer learning from a base model and continues learning from
    user data to create a personalized model that better recognizes
    individual emotional patterns and responses.
    """
    
    def __init__(self, 
                 user_id: str,
                 base_model: Optional[MultimodalProcessor] = None,
                 model_dir: str = 'models/personal',
                 learning_rate: float = 0.0005,
                 personal_weight: float = 0.7):
        """
        Initialize the personal emotion model.
        
        Args:
            user_id: Unique identifier for the user
            base_model: Optional base MultimodalProcessor to adapt from
            model_dir: Directory to store model files
            learning_rate: Learning rate for fine-tuning
            personal_weight: Weight for personal model vs base model
        """
        self.user_id = user_id
        self.model_dir = model_dir
        self.learning_rate = learning_rate
        self.personal_weight = personal_weight
        
        # History of personal data
        self.personal_data = []
        self.personal_labels = []
        
        # Track model training
        self.training_iterations = 0
        self.last_training_time = 0
        
        # Create model file paths
        os.makedirs(model_dir, exist_ok=True)
        self.model_path = os.path.join(model_dir, f"personal_model_{user_id}")
        
        # Initialize or load the model
        self.model = self._initialize_model(base_model)
        
        logger.info(f"Initialized personal emotion model for user {user_id}")
    
    def _initialize_model(self, 
                         base_model: Optional[MultimodalProcessor]) -> MultimodalProcessor:
        """
        Initialize the personal model from base model or load from disk.
        
        Args:
            base_model: Optional base model to adapt from
            
        Returns:
            Initialized MultimodalProcessor
        """
        # Check if personal model exists
        if os.path.exists(f"{self.model_path}.config.json"):
            logger.info(f"Loading existing personal model for user {self.user_id}")
            try:
                return MultimodalProcessor.load_model(self.model_path)
            except Exception as e:
                logger.error(f"Error loading personal model: {str(e)}")
                logger.info("Falling back to base model")
        
        # If no personal model, start with base model
        if base_model is not None:
            logger.info(f"Creating personal model from base model for user {self.user_id}")
            
            # Create a copy of the base model with adjusted learning rate
            model_copy = MultimodalProcessor(
                input_window=base_model.input_window,
                use_attention=base_model.use_attention,
                modality_weights=base_model.modality_weights
            )
            
            # Copy weights from base model
            for layer_base, layer_personal in zip(base_model.model.layers, model_copy.model.layers):
                layer_personal.set_weights(layer_base.get_weights())
            
            # Adjust learning rate for fine-tuning
            model_copy.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
                loss={
                    'emotion_output': 'binary_crossentropy',
                    'dimension_output': 'mse'
                },
                metrics={
                    'emotion_output': ['accuracy'],
                    'dimension_output': ['mae']
                }
            )
            
            return model_copy
        
        # If no base model, create a new model
        logger.info(f"Creating new personal model for user {self.user_id}")
        return MultimodalProcessor()
    
    def add_personal_data(self, 
                         input_data: Dict[str, Any], 
                         emotion_labels: Optional[Dict[str, float]] = None,
                         dimension_labels: Optional[Dict[str, float]] = None) -> None:
        """
        Add personal data for training.
        
        Args:
            input_data: Sensor and user input data
            emotion_labels: Optional emotion category labels (if user provided feedback)
            dimension_labels: Optional valence/arousal labels (if user provided feedback)
        """
        # Add timestamp if not present
        if "timestamp" not in input_data:
            input_data["timestamp"] = time.time()
        
        # Add to personal data collection
        self.personal_data.append(input_data)
        
        # Create label entry (can be partially labeled)
        label_entry = {
            "emotion_labels": emotion_labels,
            "dimension_labels": dimension_labels,
            "timestamp": input_data["timestamp"]
        }
        self.personal_labels.append(label_entry)
        
        logger.debug(f"Added personal data point for user {self.user_id}")
    
    def analyze_emotions(self, 
                        input_data: Dict[str, Any],
                        use_personalization: bool = True) -> Dict[str, Any]:
        """
        Analyze emotions using the personal model.
        
        Args:
            input_data: Input data from sensors and user input
            use_personalization: Whether to apply personal adjustments
            
        Returns:
            Dict containing emotion analysis results
        """
        # Get base analysis
        results = self.model.analyze_emotions(input_data)
        
        if use_personalization:
            # Apply personalized adjustments based on learned patterns
            self._apply_personalization(results)
        
        # Add to personal data for potential future training
        # Note: We don't add labels here since we don't know ground truth
        self.add_personal_data(input_data)
        
        return results
    
    def _apply_personalization(self, results: Dict[str, Any]) -> None:
        """
        Apply personalized adjustments to emotion results.
        
        This method modifies the results in place based on learned patterns.
        
        Args:
            results: Emotion analysis results to adjust
        """
        # Skip if no training has occurred yet
        if self.training_iterations == 0:
            return
        
        # Apply personalization based on user's emotional patterns
        # This could include:
        # - Adjusting emotion thresholds based on user baseline
        # - Emphasizing certain emotions based on user sensitivity
        # - Modifying valence/arousal based on user's typical range
        
        # For simplicity, we'll just scale the confidence based on personal weight
        if "confidence" in results:
            results["confidence"] = results["confidence"] * self.personal_weight
    
    def train_personal_model(self, 
                            min_samples: int = 10,
                            max_samples: int = 1000,
                            epochs: int = 5) -> Dict[str, Any]:
        """
        Train the personal model using collected data.
        
        Args:
            min_samples: Minimum number of labeled samples required
            max_samples: Maximum number of samples to use
            epochs: Number of training epochs
            
        Returns:
            Dict containing training results
        """
        # Check if we have enough labeled data
        labeled_samples = self._get_labeled_samples()
        
        if len(labeled_samples) < min_samples:
            logger.info(f"Not enough labeled samples for training: {len(labeled_samples)}/{min_samples}")
            return {"status": "not_enough_data", "labeled_samples": len(labeled_samples)}
        
        # Prepare training data
        training_data = []
        emotion_labels = []
        dimension_labels = []
        
        # Use the most recent samples up to max_samples
        samples_to_use = labeled_samples[-max_samples:]
        
        for sample_idx in samples_to_use:
            data = self.personal_data[sample_idx]
            label = self.personal_labels[sample_idx]
            
            # Only use samples with at least one type of label
            if label["emotion_labels"] is not None or label["dimension_labels"] is not None:
                training_data.append(data)
                
                # Create emotion labels array (defaults to 0.5 if not provided)
                emotions = np.array([
                    label["emotion_labels"].get(emotion, 0.5) 
                    if label["emotion_labels"] is not None else 0.5
                    for emotion in self.model.emotion_categories
                ], dtype=np.float32).reshape(1, -1)
                emotion_labels.append(emotions)
                
                # Create dimension labels array (defaults to 0.0 if not provided)
                if label["dimension_labels"] is not None:
                    valence = label["dimension_labels"].get("valence", 0.0)
                    arousal = label["dimension_labels"].get("arousal", 0.0)
                else:
                    valence, arousal = 0.0, 0.0
                
                dimensions = np.array([valence, arousal], dtype=np.float32).reshape(1, -1)
                dimension_labels.append(dimensions)
        
        # Stack all labels
        all_emotion_labels = np.vstack(emotion_labels)
        all_dimension_labels = np.vstack(dimension_labels)
        
        # Train the model
        training_result = self.model.train(
            training_data=training_data,
            labels={
                'emotions': all_emotion_labels,
                'dimensions': all_dimension_labels
            },
            validation_split=0.2,
            epochs=epochs,
            batch_size=min(32, len(training_data))
        )
        
        # Save the updated model
        self.model.save_model(self.model_path)
        
        # Update training tracking
        self.training_iterations += 1
        self.last_training_time = time.time()
        
        logger.info(f"Trained personal model for user {self.user_id} with {len(training_data)} samples")
        
        return {
            "status": "success",
            "samples_used": len(training_data),
            "training_history": training_result,
            "iterations": self.training_iterations
        }
    
    def _get_labeled_samples(self) -> List[int]:
        """
        Get indices of samples with at least partial labels.
        
        Returns:
            List of indices of labeled samples
        """
        labeled_indices = []
        
        for i, label in enumerate(self.personal_labels):
            if label["emotion_labels"] is not None or label["dimension_labels"] is not None:
                labeled_indices.append(i)
        
        return labeled_indices
    
    def add_feedback(self, 
                    timestamp: float, 
                    emotion_feedback: Optional[Dict[str, float]] = None,
                    dimension_feedback: Optional[Dict[str, float]] = None) -> bool:
        """
        Add user feedback for a specific data point.
        
        Args:
            timestamp: Timestamp of the data point to update
            emotion_feedback: User feedback on emotion categories
            dimension_feedback: User feedback on valence/arousal
            
        Returns:
            bool: True if feedback was added successfully
        """
        # Find the closest data point by timestamp
        closest_idx = -1
        min_time_diff = float('inf')
        
        for i, data in enumerate(self.personal_data):
            time_diff = abs(data.get("timestamp", 0) - timestamp)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_idx = i
        
        # Check if we found a reasonably close data point (within 10 seconds)
        if closest_idx >= 0 and min_time_diff <= 10.0:
            # Update the labels
            if emotion_feedback is not None:
                if self.personal_labels[closest_idx]["emotion_labels"] is None:
                    self.personal_labels[closest_idx]["emotion_labels"] = {}
                
                self.personal_labels[closest_idx]["emotion_labels"].update(emotion_feedback)
            
            if dimension_feedback is not None:
                if self.personal_labels[closest_idx]["dimension_labels"] is None:
                    self.personal_labels[closest_idx]["dimension_labels"] = {}
                
                self.personal_labels[closest_idx]["dimension_labels"].update(dimension_feedback)
            
            logger.info(f"Added feedback for user {self.user_id} at timestamp {timestamp}")
            return True
        
        logger.warning(f"No matching data point found for feedback at timestamp {timestamp}")
        return False
    
    def reset_model(self, keep_data: bool = True) -> None:
        """
        Reset the personal model to the base model.
        
        Args:
            keep_data: Whether to keep collected personal data
        """
        # Initialize a new model
        self.model = MultimodalProcessor()
        
        # Clear data if requested
        if not keep_data:
            self.personal_data = []
            self.personal_labels = []
        
        # Reset training tracking
        self.training_iterations = 0
        self.last_training_time = 0
        
        logger.info(f"Reset personal model for user {self.user_id}")
    
    def get_personalization_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the personalization.
        
        Returns:
            Dict containing personalization statistics
        """
        total_samples = len(self.personal_data)
        labeled_samples = len(self._get_labeled_samples())
        
        # Calculate personalization score (0-100)
        if self.training_iterations == 0:
            personalization_score = 0
        else:
            # Score based on iterations and labeled data ratio
            base_score = min(50, self.training_iterations * 10)
            data_score = min(50, (labeled_samples / max(1, total_samples)) * 100)
            personalization_score = int(base_score + data_score)
        
        return {
            "user_id": self.user_id,
            "total_samples": total_samples,
            "labeled_samples": labeled_samples,
            "training_iterations": self.training_iterations,
            "last_training_time": self.last_training_time,
            "personalization_score": personalization_score
        }
    
    def save_personal_data(self, filepath: str) -> bool:
        """
        Save collected personal data to a file.
        
        Args:
            filepath: Path to save the data
            
        Returns:
            bool: True if data was saved successfully
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            # Convert data to serializable format
            serializable_data = []
            
            for i, data in enumerate(self.personal_data):
                # Create a copy of the data without numpy arrays
                data_copy = {}
                for key, value in data.items():
                    if isinstance(value, dict):
                        # Convert nested dicts with numpy values
                        data_copy[key] = {
                            k: v.tolist() if isinstance(v, np.ndarray) else v
                            for k, v in value.items()
                        }
                    elif isinstance(value, np.ndarray):
                        data_copy[key] = value.tolist()
                    else:
                        data_copy[key] = value
                
                # Add label information
                entry = {
                    "data": data_copy,
                    "labels": self.personal_labels[i]
                }
                
                serializable_data.append(entry)
            
            # Save to file
            with open(filepath, 'w') as f:
                json.dump(serializable_data, f, indent=2)
            
            logger.info(f"Saved personal data for user {self.user_id} to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving personal data: {str(e)}")
            return False
    
    def load_personal_data(self, filepath: str) -> bool:
        """
        Load personal data from a file.
        
        Args:
            filepath: Path to the data file
            
        Returns:
            bool: True if data was loaded successfully
        """
        if not os.path.exists(filepath):
            logger.error(f"Personal data file not found: {filepath}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                serializable_data = json.load(f)
            
            # Process loaded data
            self.personal_data = []
            self.personal_labels = []
            
            for entry in serializable_data:
                self.personal_data.append(entry["data"])
                self.personal_labels.append(entry["labels"])
            
            logger.info(f"Loaded personal data for user {self.user_id} from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error loading personal data: {str(e)}")
            return False

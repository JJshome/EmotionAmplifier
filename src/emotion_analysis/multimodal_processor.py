"""
Multimodal Processor Module

This module processes data from multiple modalities (sensors and user input)
using a multimodal LSTM neural network architecture to analyze emotional states.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import os
import json

logger = logging.getLogger(__name__)

class ModalityType(Enum):
    """Enum for supported input modality types"""
    HEART_RATE = "heart_rate"
    EEG = "eeg"
    SKIN_CONDUCTANCE = "skin_conductance"
    BIOIMPEDANCE = "bioimpedance"
    FACIAL_EXPRESSION = "facial_expression"
    VOICE = "voice"
    USER_INPUT = "user_input"


class MultimodalProcessor:
    """
    Processes multimodal emotion data using deep learning techniques.
    
    Implements a multimodal LSTM architecture to process and integrate
    data from different sensor modalities and user input for accurate
    emotion classification.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 input_window: int = 30,
                 use_attention: bool = True,
                 modality_weights: Optional[Dict[ModalityType, float]] = None):
        """
        Initialize the multimodal processor.
        
        Args:
            model_path: Path to pre-trained model weights (if None, initializes untrained model)
            input_window: Number of time steps to consider for time series data
            use_attention: Whether to use attention mechanism for modality fusion
            modality_weights: Optional weights for different modalities
        """
        self.input_window = input_window
        self.use_attention = use_attention
        
        # Default weights if none provided
        if modality_weights is None:
            modality_weights = {
                ModalityType.HEART_RATE: 1.0,
                ModalityType.EEG: 1.0,
                ModalityType.SKIN_CONDUCTANCE: 1.0,
                ModalityType.BIOIMPEDANCE: 1.0,
                ModalityType.FACIAL_EXPRESSION: 1.0,
                ModalityType.VOICE: 1.0,
                ModalityType.USER_INPUT: 1.0
            }
        self.modality_weights = modality_weights
        
        # Emotion categories
        self.emotion_categories = [
            "joy", "trust", "fear", "surprise", 
            "sadness", "disgust", "anger", "anticipation"
        ]
        
        # Feature dimensions for each modality
        self.feature_dims = {
            ModalityType.HEART_RATE: 5,         # HR, HRV features
            ModalityType.EEG: 7,                # EEG band powers and metrics
            ModalityType.SKIN_CONDUCTANCE: 3,   # GSR features
            ModalityType.BIOIMPEDANCE: 6,       # Bioimpedance features
            ModalityType.FACIAL_EXPRESSION: 8,  # One per emotion category
            ModalityType.VOICE: 5,              # Voice features
            ModalityType.USER_INPUT: 10         # User reported emotions and preferences
        }
        
        # Initialize the model
        self.model = self._build_model()
        
        # Load pre-trained weights if specified
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_weights(model_path)
                logger.info(f"Loaded model weights from {model_path}")
            except Exception as e:
                logger.error(f"Error loading model weights: {str(e)}")
        else:
            logger.info("Initialized new model (untrained)")

    def _build_model(self) -> tf.keras.Model:
        """
        Build the multimodal LSTM neural network architecture.
        
        Returns:
            Compiled TensorFlow model
        """
        # Input layers for each modality
        inputs = {}
        lstm_outputs = {}
        
        # Create LSTM branch for each time-series modality
        for modality in [m for m in ModalityType if m != ModalityType.USER_INPUT]:
            input_shape = (self.input_window, self.feature_dims[modality])
            inputs[modality] = tf.keras.layers.Input(shape=input_shape, name=f"{modality.value}_input")
            
            # LSTM layers
            lstm1 = tf.keras.layers.LSTM(64, return_sequences=True)(inputs[modality])
            dropout1 = tf.keras.layers.Dropout(0.2)(lstm1)
            lstm2 = tf.keras.layers.LSTM(32, return_sequences=False)(dropout1)
            dropout2 = tf.keras.layers.Dropout(0.2)(lstm2)
            
            lstm_outputs[modality] = dropout2
        
        # Input for user reported data (non-time series)
        user_input_shape = (self.feature_dims[ModalityType.USER_INPUT],)
        inputs[ModalityType.USER_INPUT] = tf.keras.layers.Input(
            shape=user_input_shape, 
            name=f"{ModalityType.USER_INPUT.value}_input"
        )
        dense_user = tf.keras.layers.Dense(32, activation='relu')(inputs[ModalityType.USER_INPUT])
        dropout_user = tf.keras.layers.Dropout(0.2)(dense_user)
        lstm_outputs[ModalityType.USER_INPUT] = dropout_user
        
        # Apply modality weights
        weighted_outputs = {}
        for modality in ModalityType:
            weight = self.modality_weights.get(modality, 1.0)
            weighted_outputs[modality] = tf.keras.layers.Lambda(
                lambda x, w=weight: x * w,
                name=f"{modality.value}_weight"
            )(lstm_outputs[modality])
        
        # Fusion approach based on setting
        if self.use_attention:
            # Attention-based fusion
            fusion_inputs = list(weighted_outputs.values())
            
            # Reshape to same dimensions
            reshaped_inputs = []
            for i, (modality, output) in enumerate(weighted_outputs.items()):
                reshaped = tf.keras.layers.Reshape((1, 32))(output)
                reshaped_inputs.append(reshaped)
            
            # Concatenate along time dimension
            concat = tf.keras.layers.Concatenate(axis=1)(reshaped_inputs)
            
            # Self-attention
            attention = tf.keras.layers.MultiHeadAttention(
                num_heads=4, key_dim=8
            )(concat, concat)
            
            # Global pooling
            pool = tf.keras.layers.GlobalAveragePooling1D()(attention)
            
            fusion = pool
        else:
            # Simple concatenation fusion
            fusion = tf.keras.layers.Concatenate(axis=1)(list(weighted_outputs.values()))
        
        # Final layers
        dense1 = tf.keras.layers.Dense(64, activation='relu')(fusion)
        dropout1 = tf.keras.layers.Dropout(0.3)(dense1)
        dense2 = tf.keras.layers.Dense(32, activation='relu')(dropout1)
        dropout2 = tf.keras.layers.Dropout(0.3)(dense2)
        
        # Emotion classification outputs
        emotion_outputs = tf.keras.layers.Dense(
            len(self.emotion_categories), 
            activation='sigmoid',
            name='emotion_output'
        )(dropout2)
        
        # Arousal and valence outputs (dimensional emotion model)
        dimension_outputs = tf.keras.layers.Dense(
            2, 
            activation='tanh',  # Range -1 to 1
            name='dimension_output'
        )(dropout2)
        
        # Create the model
        model = tf.keras.Model(
            inputs=list(inputs.values()), 
            outputs=[emotion_outputs, dimension_outputs]
        )
        
        # Compile with appropriate loss functions
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'emotion_output': 'binary_crossentropy',
                'dimension_output': 'mse'
            },
            metrics={
                'emotion_output': ['accuracy'],
                'dimension_output': ['mae']
            }
        )
        
        return model
    
    def preprocess_data(self, 
                       input_data: Dict[str, Any]) -> Dict[ModalityType, np.ndarray]:
        """
        Preprocess raw data from various modalities for model input.
        
        Args:
            input_data: Dictionary containing data from different sources
            
        Returns:
            Dict mapping modality types to preprocessed numpy arrays
        """
        preprocessed = {}
        features_data = input_data.get("features", {})
        
        # Preprocessing for heart rate data
        hr_features = [
            features_data.get("hr_mean", 0.0),
            features_data.get("hr_std", 0.0),
            features_data.get("hrv_sdnn_mean", 0.0),
            features_data.get("hrv_rmssd_mean", 0.0),
            features_data.get("hrv_pnn50_mean", 0.0)
        ]
        preprocessed[ModalityType.HEART_RATE] = np.tile(
            np.array(hr_features, dtype=np.float32),
            (self.input_window, 1)
        )
        
        # Preprocessing for EEG data
        eeg_features = [
            features_data.get("eeg_alpha_mean", 0.0),
            features_data.get("eeg_beta_mean", 0.0),
            features_data.get("eeg_theta_mean", 0.0),
            features_data.get("eeg_delta_mean", 0.0),
            features_data.get("eeg_alpha_beta_ratio", 0.0),
            features_data.get("eeg_attention_mean", 0.0),
            features_data.get("eeg_meditation_mean", 0.0)
        ]
        preprocessed[ModalityType.EEG] = np.tile(
            np.array(eeg_features, dtype=np.float32),
            (self.input_window, 1)
        )
        
        # Preprocessing for GSR data
        gsr_features = [
            features_data.get("gsr_mean", 0.0),
            features_data.get("gsr_std", 0.0),
            features_data.get("gsr_scr_count_mean", 0.0)
        ]
        preprocessed[ModalityType.SKIN_CONDUCTANCE] = np.tile(
            np.array(gsr_features, dtype=np.float32),
            (self.input_window, 1)
        )
        
        # Preprocessing for bioimpedance data
        bio_features = [
            features_data.get("bio_sympathetic_activation", 0.0),
            features_data.get("bio_parasympathetic_activation", 0.0),
            features_data.get("bio_emotional_intensity", 0.0),
            features_data.get("bio_emotional_complexity", 0.0),
            features_data.get("bio_impedance_ratio", 0.0),
            features_data.get("bio_characteristic_freq", 0.0)
        ]
        preprocessed[ModalityType.BIOIMPEDANCE] = np.tile(
            np.array(bio_features, dtype=np.float32),
            (self.input_window, 1)
        )
        
        # Preprocessing for facial expression data
        facial_features = [
            features_data.get("facial_happy_mean", 0.0),
            features_data.get("facial_sad_mean", 0.0),
            features_data.get("facial_angry_mean", 0.0),
            features_data.get("facial_surprised_mean", 0.0),
            features_data.get("facial_fearful_mean", 0.0),
            features_data.get("facial_disgusted_mean", 0.0),
            features_data.get("facial_neutral_mean", 0.0),
            features_data.get("facial_detection_rate", 0.0)
        ]
        preprocessed[ModalityType.FACIAL_EXPRESSION] = np.tile(
            np.array(facial_features, dtype=np.float32),
            (self.input_window, 1)
        )
        
        # Preprocessing for voice data (if available)
        voice_features = [0.0] * self.feature_dims[ModalityType.VOICE]  # Placeholder
        preprocessed[ModalityType.VOICE] = np.tile(
            np.array(voice_features, dtype=np.float32),
            (self.input_window, 1)
        )
        
        # User input features - these should come from user self-reports
        user_input_features = [0.0] * self.feature_dims[ModalityType.USER_INPUT]
        
        # If user input data is available, process it
        if "user_input" in input_data:
            user_data = input_data["user_input"]
            
            # Extract emotion self-reports
            if "emotions" in user_data:
                for i, emotion in enumerate(self.emotion_categories):
                    if i < 8 and emotion in user_data["emotions"]:
                        user_input_features[i] = user_data["emotions"][emotion]
            
            # Extract PAD model values if available
            if "pad_values" in user_data:
                pad = user_data["pad_values"]
                if "pleasure" in pad:
                    user_input_features[8] = (pad["pleasure"] + 1) / 2  # Convert -1..1 to 0..1
                if "arousal" in pad:
                    user_input_features[9] = (pad["arousal"] + 1) / 2   # Convert -1..1 to 0..1
        
        preprocessed[ModalityType.USER_INPUT] = np.array(user_input_features, dtype=np.float32)
        
        return preprocessed
    
    def analyze_emotions(self, 
                        input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze emotions from input data using the multimodal model.
        
        Args:
            input_data: Dictionary containing data from different sources
            
        Returns:
            Dict containing emotion analysis results
        """
        # Preprocess the input data
        preprocessed = self.preprocess_data(input_data)
        
        # Prepare inputs for the model
        model_inputs = [preprocessed[modality] for modality in ModalityType]
        
        # Make prediction
        emotion_probs, dimension_values = self.model.predict(model_inputs)
        
        # Convert outputs to results dictionary
        emotions = {}
        for i, emotion in enumerate(self.emotion_categories):
            emotions[emotion] = float(emotion_probs[0, i])
        
        # Determine dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        
        # Convert dimension values from [-1,1] to interpretable format
        valence = float(dimension_values[0, 0])  # -1 (negative) to 1 (positive)
        arousal = float(dimension_values[0, 1])  # -1 (calm) to 1 (excited)
        
        # Calculate emotional quadrant
        quadrant = self._get_emotion_quadrant(valence, arousal)
        
        # Prepare results
        results = {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "dimensions": {
                "valence": valence,
                "arousal": arousal
            },
            "quadrant": quadrant,
            "timestamp": input_data.get("timestamp", 0)
        }
        
        # Add confidence score (average of top 3 emotion probabilities)
        top_emotions = sorted(emotions.values(), reverse=True)[:3]
        results["confidence"] = sum(top_emotions) / 3 if top_emotions else 0
        
        return results
    
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
    
    def train(self, 
             training_data: List[Dict[str, Any]], 
             labels: Dict[str, np.ndarray],
             validation_split: float = 0.2,
             epochs: int = 50,
             batch_size: int = 32) -> Dict[str, Any]:
        """
        Train the multimodal model on labeled data.
        
        Args:
            training_data: List of input data dictionaries
            labels: Dict with 'emotions' and 'dimensions' numpy arrays
            validation_split: Fraction of data to use for validation
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Dict containing training history
        """
        # Preprocess all training samples
        processed_inputs = []
        for modality in ModalityType:
            modality_data = []
            for sample in training_data:
                preprocessed = self.preprocess_data(sample)
                modality_data.append(preprocessed[modality])
            
            # Stack into a single array
            if modality == ModalityType.USER_INPUT:
                # User input is not time series
                processed_inputs.append(np.stack(modality_data))
            else:
                processed_inputs.append(np.stack(modality_data))
        
        # Train the model
        history = self.model.fit(
            processed_inputs,
            [labels['emotions'], labels['dimensions']],
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=5,
                    restore_best_weights=True
                )
            ]
        )
        
        # Convert history to dictionary
        history_dict = {}
        for key, values in history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        return history_dict
    
    def save_model(self, save_path: str) -> bool:
        """
        Save the model to a file.
        
        Args:
            save_path: Path to save the model weights
            
        Returns:
            bool: True if model was saved successfully
        """
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model weights
            self.model.save_weights(save_path)
            
            # Save model configuration
            config = {
                "input_window": self.input_window,
                "use_attention": self.use_attention,
                "modality_weights": {k.value: v for k, v in self.modality_weights.items()},
                "emotion_categories": self.emotion_categories,
                "feature_dims": {k.value: v for k, v in self.feature_dims.items()}
            }
            
            config_path = save_path + ".config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            return False
    
    @classmethod
    def load_model(cls, load_path: str) -> 'MultimodalProcessor':
        """
        Load a model from a file.
        
        Args:
            load_path: Path to the saved model weights
            
        Returns:
            MultimodalProcessor instance with loaded model
        """
        # Load configuration
        config_path = load_path + ".config.json"
        
        if not os.path.exists(config_path):
            logger.warning(f"Configuration file {config_path} not found, using defaults")
            processor = cls(model_path=load_path)
            return processor
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Convert modality weights back to enum keys
            modality_weights = {}
            if "modality_weights" in config:
                for k, v in config["modality_weights"].items():
                    try:
                        modality_weights[ModalityType(k)] = v
                    except ValueError:
                        logger.warning(f"Unknown modality type: {k}")
            
            # Create processor with loaded configuration
            processor = cls(
                model_path=load_path,
                input_window=config.get("input_window", 30),
                use_attention=config.get("use_attention", True),
                modality_weights=modality_weights
            )
            
            # Set additional configuration
            if "emotion_categories" in config:
                processor.emotion_categories = config["emotion_categories"]
            
            return processor
        except Exception as e:
            logger.error(f"Error loading model configuration: {str(e)}")
            logger.info("Falling back to default configuration")
            processor = cls(model_path=load_path)
            return processor

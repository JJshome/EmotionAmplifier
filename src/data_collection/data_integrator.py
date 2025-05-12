"""
Data Integrator Module

This module handles integration of data from multiple sensors and sources
to create a cohesive multimodal dataset for emotion analysis.
"""

import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import deque
from .sensor_manager import SensorType

logger = logging.getLogger(__name__)

class DataIntegrator:
    """
    Integrates and synchronizes data from multiple sensors.
    
    Handles timing discrepancies between sensors, transforms raw sensor data into
    a standardized format, and creates feature vectors for emotion analysis.
    """
    
    def __init__(self, 
                 window_size: float = 30.0,
                 min_sensors_required: int = 1,
                 storage_enabled: bool = True):
        """
        Initialize the data integrator.
        
        Args:
            window_size: Time window for data integration in seconds
            min_sensors_required: Minimum number of sensors required for valid integration
            storage_enabled: Whether to store integrated data
        """
        self.window_size = window_size
        self.min_sensors_required = min_sensors_required
        self.storage_enabled = storage_enabled
        
        # Data buffers for each sensor type
        self.data_buffers = {sensor_type: deque() for sensor_type in SensorType}
        
        # Last integration timestamp
        self.last_integration_time = 0
        
        # Callbacks for integrated data
        self.callbacks = []
        
        # Storage for integrated data
        self.integrated_data_history = deque(maxlen=100) if storage_enabled else None
        
        logger.info(f"Data integrator initialized with window size: {window_size}s")
    
    def process_sensor_data(self, sensor_data: Dict[str, Any]) -> None:
        """
        Process incoming sensor data and add it to the appropriate buffer.
        
        Args:
            sensor_data: Dictionary of sensor data indexed by sensor ID
        """
        timestamp = sensor_data.get("collection_timestamp", time.time())
        
        # Process each sensor's data
        for sensor_id, data in sensor_data.items():
            # Skip metadata fields
            if sensor_id in ["collection_timestamp", "sensors_count"]:
                continue
            
            # Extract sensor type from the sensor ID (format: type_deviceid)
            try:
                sensor_type_str = sensor_id.split("_")[0]
                sensor_type = SensorType(sensor_type_str)
            except (ValueError, IndexError):
                logger.warning(f"Unknown sensor type in sensor ID: {sensor_id}")
                continue
            
            # Add to the appropriate buffer with timestamp
            data_entry = {
                "timestamp": timestamp,
                "sensor_id": sensor_id,
                "data": data
            }
            
            self.data_buffers[sensor_type].append(data_entry)
            logger.debug(f"Added data from {sensor_id} to buffer")
        
        # Clean up old data
        self._clean_buffers(timestamp)
        
        # Check if it's time to perform integration
        if timestamp - self.last_integration_time >= 1.0:  # Integrate at most once per second
            self._integrate_data(timestamp)
    
    def _clean_buffers(self, current_time: float) -> None:
        """
        Remove data older than the window size from buffers.
        
        Args:
            current_time: Current timestamp
        """
        cutoff_time = current_time - self.window_size
        
        for sensor_type, buffer in self.data_buffers.items():
            # Remove entries older than cutoff time
            while buffer and buffer[0]["timestamp"] < cutoff_time:
                buffer.popleft()
    
    def _integrate_data(self, timestamp: float) -> Optional[Dict[str, Any]]:
        """
        Integrate data from multiple sensors within the current time window.
        
        Args:
            timestamp: Current timestamp
            
        Returns:
            Dict containing integrated data or None if insufficient data
        """
        # Check if we have enough sensor data
        active_sensors = sum(1 for buffer in self.data_buffers.values() if buffer)
        
        if active_sensors < self.min_sensors_required:
            logger.debug(f"Not enough sensors active for integration: {active_sensors}/{self.min_sensors_required}")
            return None
        
        # Update integration timestamp
        self.last_integration_time = timestamp
        
        # Create integrated data structure
        integrated_data = {
            "timestamp": timestamp,
            "window_size": self.window_size,
            "active_sensors": active_sensors
        }
        
        # Process each sensor type
        for sensor_type in SensorType:
            buffer = self.data_buffers[sensor_type]
            
            if not buffer:
                continue
            
            # Extract the most recent data for each unique sensor ID
            sensor_type_data = {}
            seen_sensors = set()
            
            # Go through buffer from newest to oldest
            for entry in sorted(buffer, key=lambda x: x["timestamp"], reverse=True):
                sensor_id = entry["sensor_id"]
                
                # Skip if we already have data for this sensor
                if sensor_id in seen_sensors:
                    continue
                
                seen_sensors.add(sensor_id)
                sensor_type_data[sensor_id] = entry["data"]
            
            # Add to integrated data
            integrated_data[sensor_type.value] = sensor_type_data
        
        # Extract features from the integrated data
        features = self._extract_features(integrated_data)
        integrated_data["features"] = features
        
        # Store the integrated data if storage is enabled
        if self.storage_enabled and self.integrated_data_history is not None:
            self.integrated_data_history.append(integrated_data)
        
        # Call callbacks with the integrated data
        for callback in self.callbacks:
            try:
                callback(integrated_data)
            except Exception as e:
                logger.error(f"Error in integration callback: {str(e)}")
        
        logger.debug(f"Integrated data from {active_sensors} sensors")
        return integrated_data
    
    def _extract_features(self, 
                         integrated_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract standardized features from integrated sensor data.
        
        Args:
            integrated_data: Integrated sensor data
            
        Returns:
            Dict of extracted features
        """
        features = {}
        
        # Extract features from heart rate data
        if SensorType.HEART_RATE.value in integrated_data:
            hr_features = self._extract_heart_rate_features(integrated_data[SensorType.HEART_RATE.value])
            features.update(hr_features)
        
        # Extract features from EEG data
        if SensorType.EEG.value in integrated_data:
            eeg_features = self._extract_eeg_features(integrated_data[SensorType.EEG.value])
            features.update(eeg_features)
        
        # Extract features from skin conductance data
        if SensorType.SKIN_CONDUCTANCE.value in integrated_data:
            gsr_features = self._extract_skin_conductance_features(integrated_data[SensorType.SKIN_CONDUCTANCE.value])
            features.update(gsr_features)
        
        # Extract features from facial expression data
        if SensorType.FACIAL_EXPRESSION.value in integrated_data:
            facial_features = self._extract_facial_features(integrated_data[SensorType.FACIAL_EXPRESSION.value])
            features.update(facial_features)
        
        # Extract features from bioimpedance data
        if SensorType.BIOIMPEDANCE.value in integrated_data:
            bio_features = self._extract_bioimpedance_features(integrated_data[SensorType.BIOIMPEDANCE.value])
            features.update(bio_features)
        
        return features
    
    def _extract_heart_rate_features(self, 
                                    hr_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features from heart rate sensor data.
        
        Args:
            hr_data: Dictionary of heart rate sensor data
            
        Returns:
            Dict of heart rate features
        """
        features = {}
        
        if not hr_data:
            return features
        
        # Combine data from multiple heart rate sensors if available
        heart_rates = []
        hrv_values = {"sdnn": [], "rmssd": [], "pnn50": []}
        
        for sensor_id, data in hr_data.items():
            if "heart_rate" in data:
                heart_rates.append(data["heart_rate"])
            
            if "hrv" in data and data["hrv"]:
                hrv = data["hrv"]
                for key in hrv_values:
                    if key in hrv:
                        hrv_values[key].append(hrv[key])
        
        # Calculate heart rate features
        if heart_rates:
            features["hr_mean"] = np.mean(heart_rates)
            
            if len(heart_rates) > 1:
                features["hr_std"] = np.std(heart_rates)
        
        # Calculate HRV features
        for key, values in hrv_values.items():
            if values:
                features[f"hrv_{key}_mean"] = np.mean(values)
        
        return features
    
    def _extract_eeg_features(self, 
                             eeg_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features from EEG sensor data.
        
        Args:
            eeg_data: Dictionary of EEG sensor data
            
        Returns:
            Dict of EEG features
        """
        features = {}
        
        if not eeg_data:
            return features
        
        # This would involve complex EEG processing in a real implementation
        # For now, we'll use simplified placeholder features
        
        # Combine data from multiple EEG sensors if available
        alpha_powers = []
        beta_powers = []
        theta_powers = []
        delta_powers = []
        attention_values = []
        meditation_values = []
        
        for sensor_id, data in eeg_data.items():
            # Extract band powers if available
            if "bands" in data:
                bands = data["bands"]
                if "alpha" in bands:
                    alpha_powers.append(bands["alpha"])
                if "beta" in bands:
                    beta_powers.append(bands["beta"])
                if "theta" in bands:
                    theta_powers.append(bands["theta"])
                if "delta" in bands:
                    delta_powers.append(bands["delta"])
            
            # Extract derived metrics if available
            if "metrics" in data:
                metrics = data["metrics"]
                if "attention" in metrics:
                    attention_values.append(metrics["attention"])
                if "meditation" in metrics:
                    meditation_values.append(metrics["meditation"])
        
        # Calculate feature values
        if alpha_powers:
            features["eeg_alpha_mean"] = np.mean(alpha_powers)
        if beta_powers:
            features["eeg_beta_mean"] = np.mean(beta_powers)
        if theta_powers:
            features["eeg_theta_mean"] = np.mean(theta_powers)
        if delta_powers:
            features["eeg_delta_mean"] = np.mean(delta_powers)
        
        if alpha_powers and beta_powers:
            features["eeg_alpha_beta_ratio"] = np.mean(alpha_powers) / max(0.001, np.mean(beta_powers))
        
        if attention_values:
            features["eeg_attention_mean"] = np.mean(attention_values)
        if meditation_values:
            features["eeg_meditation_mean"] = np.mean(meditation_values)
        
        return features
    
    def _extract_skin_conductance_features(self, 
                                          gsr_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features from skin conductance sensor data.
        
        Args:
            gsr_data: Dictionary of skin conductance sensor data
            
        Returns:
            Dict of skin conductance features
        """
        features = {}
        
        if not gsr_data:
            return features
        
        # Combine data from multiple GSR sensors if available
        gsr_values = []
        scr_counts = []  # Skin conductance responses
        scr_amplitudes = []
        
        for sensor_id, data in gsr_data.items():
            if "conductance" in data:
                gsr_values.append(data["conductance"])
            
            if "scr_count" in data:
                scr_counts.append(data["scr_count"])
            
            if "scr_amplitude" in data:
                scr_amplitudes.append(data["scr_amplitude"])
        
        # Calculate GSR features
        if gsr_values:
            features["gsr_mean"] = np.mean(gsr_values)
            
            if len(gsr_values) > 1:
                features["gsr_std"] = np.std(gsr_values)
        
        if scr_counts:
            features["gsr_scr_count_mean"] = np.mean(scr_counts)
        
        if scr_amplitudes:
            features["gsr_scr_amplitude_mean"] = np.mean(scr_amplitudes)
        
        return features
    
    def _extract_facial_features(self, 
                                facial_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features from facial expression sensor data.
        
        Args:
            facial_data: Dictionary of facial expression sensor data
            
        Returns:
            Dict of facial expression features
        """
        features = {}
        
        if not facial_data:
            return features
        
        # Combine data from multiple facial expression sensors if available
        emotion_values = {
            "happy": [],
            "sad": [],
            "angry": [],
            "surprised": [],
            "fearful": [],
            "disgusted": [],
            "neutral": []
        }
        
        face_detected_count = 0
        total_sensors = 0
        
        for sensor_id, data in facial_data.items():
            total_sensors += 1
            
            if data.get("face_detected", False):
                face_detected_count += 1
            
            if "emotions" in data:
                emotions = data["emotions"]
                for emotion, value in emotions.items():
                    if emotion in emotion_values:
                        emotion_values[emotion].append(value)
        
        # Calculate face detection rate
        if total_sensors > 0:
            features["facial_detection_rate"] = face_detected_count / total_sensors
        
        # Calculate emotion features
        for emotion, values in emotion_values.items():
            if values:
                features[f"facial_{emotion}_mean"] = np.mean(values)
        
        return features
    
    def _extract_bioimpedance_features(self, 
                                      bio_data: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """
        Extract features from bioimpedance sensor data.
        
        Args:
            bio_data: Dictionary of bioimpedance sensor data
            
        Returns:
            Dict of bioimpedance features
        """
        features = {}
        
        if not bio_data:
            return features
        
        # Process data from the bioimpedance sensor
        # In a real implementation, this would involve complex analysis of the impedance spectra
        
        # Here we'll just use simulated emotion indices if available
        for sensor_id, data in bio_data.items():
            if "emotion_indices" in data:
                indices = data["emotion_indices"]
                for key, value in indices.items():
                    features[f"bio_{key}"] = value
            
            # Extract Cole parameters if available
            if "cole_parameters" in data:
                params = data["cole_parameters"]
                # The r_zero/r_infinity ratio is related to fluid distribution and can indicate emotional state
                if "r_zero" in params and "r_infinity" in params and params["r_infinity"] > 0:
                    features["bio_impedance_ratio"] = params["r_zero"] / params["r_infinity"]
                
                # Characteristic frequency is related to cell membrane properties
                if "tau" in params and params["tau"] > 0:
                    features["bio_characteristic_freq"] = 1.0 / (2.0 * np.pi * params["tau"])
        
        return features
    
    def register_callback(self, callback: callable) -> None:
        """
        Register a callback function for integrated data.
        
        Args:
            callback: Function to call with integrated data
        """
        self.callbacks.append(callback)
        logger.info("Registered new integration callback")
    
    def get_latest_integrated_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent integrated data.
        
        Returns:
            Most recent integrated data or None if no data available
        """
        if not self.storage_enabled or not self.integrated_data_history:
            return None
        
        return self.integrated_data_history[-1]
    
    def get_integrated_data_history(self, 
                                   max_items: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get the history of integrated data.
        
        Args:
            max_items: Maximum number of items to return (newest first)
            
        Returns:
            List of integrated data entries
        """
        if not self.storage_enabled or not self.integrated_data_history:
            return []
        
        history = list(self.integrated_data_history)
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        
        if max_items is not None:
            history = history[:max_items]
        
        return history
    
    def export_features_as_dataframe(self) -> pd.DataFrame:
        """
        Export extracted features as a pandas DataFrame.
        
        Returns:
            DataFrame containing feature history
        """
        if not self.storage_enabled or not self.integrated_data_history:
            return pd.DataFrame()
        
        # Extract timestamps and features from history
        data = []
        for entry in self.integrated_data_history:
            if "features" in entry:
                row = {"timestamp": entry["timestamp"]}
                row.update(entry["features"])
                data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        if not df.empty:
            # Set timestamp as index
            df.set_index("timestamp", inplace=True)
            # Sort by timestamp
            df.sort_index(inplace=True)
        
        return df
    
    def save_features_to_file(self, filepath: str) -> bool:
        """
        Save extracted features to a CSV file.
        
        Args:
            filepath: Path to save the CSV file
            
        Returns:
            bool: True if file was saved successfully
        """
        df = self.export_features_as_dataframe()
        
        if df.empty:
            logger.warning("No data to save")
            return False
        
        try:
            df.to_csv(filepath)
            logger.info(f"Saved features to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving features to {filepath}: {str(e)}")
            return False

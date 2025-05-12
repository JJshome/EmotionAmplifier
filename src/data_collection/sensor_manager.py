"""
Sensor Manager Module

This module manages the connection to and data collection from various physiological sensors
for emotion detection, including heart rate monitors, skin conductance sensors, EEG devices,
and camera-based facial expression analysis.
"""

import time
import threading
import numpy as np
from typing import Dict, List, Optional, Callable, Any
import logging
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class SensorType(Enum):
    """Enum for supported sensor types"""
    HEART_RATE = "heart_rate"
    EEG = "eeg"
    SKIN_CONDUCTANCE = "skin_conductance"
    TEMPERATURE = "temperature"
    FACIAL_EXPRESSION = "facial_expression"
    VOICE = "voice"
    BIOIMPEDANCE = "bioimpedance"
    MOTION = "motion"


class SensorDevice(ABC):
    """Abstract base class for all sensor devices"""
    
    def __init__(self, device_id: str, sensor_type: SensorType):
        """
        Initialize a sensor device.
        
        Args:
            device_id: Unique identifier for the device
            sensor_type: Type of sensor
        """
        self.device_id = device_id
        self.sensor_type = sensor_type
        self.connected = False
        self.sampling_rate = 0
        self.last_data = None
        self.last_timestamp = 0
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the sensor device.
        
        Returns:
            bool: True if connection was successful
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the sensor device.
        
        Returns:
            bool: True if disconnection was successful
        """
        pass
    
    @abstractmethod
    def read_data(self) -> Dict[str, Any]:
        """
        Read data from the sensor.
        
        Returns:
            Dict containing sensor data and metadata
        """
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the sensor.
        
        Returns:
            Dict containing sensor status information
        """
        return {
            "device_id": self.device_id,
            "sensor_type": self.sensor_type.value,
            "connected": self.connected,
            "sampling_rate": self.sampling_rate,
            "last_reading_time": self.last_timestamp
        }


class WearableDevice(SensorDevice):
    """Base class for wearable sensor devices like smartwatches"""
    
    def __init__(self, 
                 device_id: str, 
                 sensor_type: SensorType,
                 device_model: str = "generic",
                 connection_type: str = "bluetooth"):
        """
        Initialize a wearable device.
        
        Args:
            device_id: Unique identifier for the device
            sensor_type: Type of sensor
            device_model: Model name of the device
            connection_type: Connection protocol (e.g., 'bluetooth', 'wifi')
        """
        super().__init__(device_id, sensor_type)
        self.device_model = device_model
        self.connection_type = connection_type
        self.battery_level = 100
        
    def get_battery_level(self) -> int:
        """
        Get the current battery level of the device.
        
        Returns:
            int: Battery level percentage
        """
        # In a real implementation, this would query the actual device
        return self.battery_level


class SmartphoneSensor(SensorDevice):
    """Base class for smartphone-based sensors (camera, microphone, etc.)"""
    
    def __init__(self, 
                 device_id: str, 
                 sensor_type: SensorType,
                 permission_granted: bool = False):
        """
        Initialize a smartphone sensor.
        
        Args:
            device_id: Unique identifier for the device
            sensor_type: Type of sensor
            permission_granted: Whether permission to access the sensor has been granted
        """
        super().__init__(device_id, sensor_type)
        self.permission_granted = permission_granted


class HeartRateSensor(WearableDevice):
    """Handles heart rate sensor data collection"""
    
    def __init__(self, device_id: str, device_model: str = "generic"):
        """
        Initialize a heart rate sensor.
        
        Args:
            device_id: Unique identifier for the device
            device_model: Model name of the device
        """
        super().__init__(device_id, SensorType.HEART_RATE, device_model)
        self.sampling_rate = 1  # Hz (typical for consumer HR monitors)
        self.hrv_enabled = True
        
    def connect(self) -> bool:
        """
        Connect to the heart rate monitor.
        
        Returns:
            bool: True if connection was successful
        """
        # Simulated connection - in a real implementation would connect to the device
        self.connected = True
        logger.info(f"Connected to heart rate sensor {self.device_id}")
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the heart rate monitor.
        
        Returns:
            bool: True if disconnection was successful
        """
        self.connected = False
        logger.info(f"Disconnected from heart rate sensor {self.device_id}")
        return True
    
    def read_data(self) -> Dict[str, Any]:
        """
        Read heart rate data.
        
        Returns:
            Dict containing heart rate, HRV, and metadata
        """
        if not self.connected:
            raise ConnectionError("Device not connected")
        
        # Simulate heart rate data - in a real implementation would read from the device
        hr = 75 + np.random.normal(0, 5)
        
        # If HRV is enabled, calculate simulated HRV metrics
        hrv_data = None
        if self.hrv_enabled:
            # Simulated HRV metrics
            hrv_data = {
                "sdnn": 35 + np.random.normal(0, 5),  # Standard deviation of NN intervals
                "rmssd": 30 + np.random.normal(0, 3),  # Root mean square of successive differences
                "pnn50": 10 + np.random.normal(0, 2)   # Percentage of NN50
            }
        
        # Update last reading information
        self.last_timestamp = time.time()
        
        # Return the data
        result = {
            "heart_rate": hr,
            "timestamp": self.last_timestamp,
            "quality": "good",
            "hrv": hrv_data
        }
        
        self.last_data = result
        return result


class FacialExpressionSensor(SmartphoneSensor):
    """Handles facial expression analysis via smartphone camera"""
    
    def __init__(self, device_id: str):
        """
        Initialize a facial expression sensor.
        
        Args:
            device_id: Unique identifier for the device
        """
        super().__init__(device_id, SensorType.FACIAL_EXPRESSION)
        self.camera_id = 0  # Front camera
        self.frame_rate = 5  # Frames per second for analysis
        self.last_frame = None
        
    def connect(self) -> bool:
        """
        Connect to the camera.
        
        Returns:
            bool: True if connection was successful
        """
        # In a real implementation, this would initialize the camera
        self.connected = True
        logger.info(f"Connected to facial expression sensor on device {self.device_id}")
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the camera.
        
        Returns:
            bool: True if disconnection was successful
        """
        self.connected = False
        logger.info(f"Disconnected from facial expression sensor on device {self.device_id}")
        return True
    
    def read_data(self) -> Dict[str, Any]:
        """
        Analyze facial expression from camera feed.
        
        Returns:
            Dict containing detected emotions and confidence scores
        """
        if not self.connected:
            raise ConnectionError("Device not connected")
        
        if not self.permission_granted:
            raise PermissionError("Camera permission not granted")
        
        # Simulate facial expression analysis - in a real implementation would process camera frame
        emotions = {
            "happy": np.random.uniform(0, 0.8),
            "sad": np.random.uniform(0, 0.3),
            "angry": np.random.uniform(0, 0.2),
            "surprised": np.random.uniform(0, 0.4),
            "fearful": np.random.uniform(0, 0.3),
            "disgusted": np.random.uniform(0, 0.2),
            "neutral": np.random.uniform(0.2, 0.7)
        }
        
        # Normalize to sum to 1.0
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        # Get the dominant emotion
        dominant_emotion = max(emotions, key=emotions.get)
        
        # Update last reading information
        self.last_timestamp = time.time()
        
        # Return the data
        result = {
            "emotions": emotions,
            "dominant_emotion": dominant_emotion,
            "timestamp": self.last_timestamp,
            "face_detected": True,
            "quality": "good"
        }
        
        self.last_data = result
        return result


class BioimpedanceSensor(WearableDevice):
    """Handles bioimpedance measurements for emotional state detection"""
    
    def __init__(self, device_id: str, device_model: str = "EmotionSense-Bio"):
        """
        Initialize a bioimpedance sensor.
        
        Args:
            device_id: Unique identifier for the device
            device_model: Model name of the device
        """
        super().__init__(device_id, SensorType.BIOIMPEDANCE, device_model)
        self.frequency_range = (1000, 100000)  # Hz
        self.frequency_steps = 50
        self.electrode_config = "wrist-wrist"
        self.sampling_rate = 0.2  # Hz (one full scan every 5 seconds)
        
    def connect(self) -> bool:
        """
        Connect to the bioimpedance sensor.
        
        Returns:
            bool: True if connection was successful
        """
        # Simulated connection - in a real implementation would connect to the device
        self.connected = True
        logger.info(f"Connected to bioimpedance sensor {self.device_id}")
        return True
    
    def disconnect(self) -> bool:
        """
        Disconnect from the bioimpedance sensor.
        
        Returns:
            bool: True if disconnection was successful
        """
        self.connected = False
        logger.info(f"Disconnected from bioimpedance sensor {self.device_id}")
        return True
    
    def read_data(self) -> Dict[str, Any]:
        """
        Read bioimpedance data across frequency spectrum.
        
        Returns:
            Dict containing complex impedance measurements at different frequencies
        """
        if not self.connected:
            raise ConnectionError("Device not connected")
        
        # Generate logarithmically spaced frequencies
        frequencies = np.logspace(
            np.log10(self.frequency_range[0]),
            np.log10(self.frequency_range[1]),
            num=self.frequency_steps
        )
        
        # Simulate bioimpedance data - in a real implementation would read from the device
        # Complex impedance values (magnitude and phase)
        magnitudes = np.linspace(500, 100, num=self.frequency_steps) + np.random.normal(0, 20, self.frequency_steps)
        phases = -np.linspace(0, 30, num=self.frequency_steps) + np.random.normal(0, 2, self.frequency_steps)
        
        # Convert to complex values
        complex_impedance = magnitudes * np.exp(1j * np.radians(phases))
        
        # Update last reading information
        self.last_timestamp = time.time()
        
        # Return the data
        result = {
            "frequencies": frequencies.tolist(),
            "complex_impedance": complex_impedance.tolist(),
            "timestamp": self.last_timestamp,
            "electrode_config": self.electrode_config,
            "quality": "good"
        }
        
        self.last_data = result
        return result


class SensorManager:
    """
    Manages multiple sensors for emotion data collection.
    
    Handles sensor discovery, connection, configuration, and data integration
    from multiple sources for comprehensive emotion analysis.
    """
    
    def __init__(self):
        """Initialize the sensor manager."""
        self.sensors = {}
        self.active_sensors = set()
        self.data_callbacks = []
        self.collection_thread = None
        self.is_collecting = False
        self.collection_interval = 1.0  # seconds
        self.logger = logging.getLogger(__name__)
        
    def discover_sensors(self) -> List[Dict[str, Any]]:
        """
        Discover available sensors.
        
        Returns:
            List of discovered sensor information
        """
        # In a real implementation, this would scan for available devices
        # For now, we'll just return some simulated devices
        
        discovered = [
            {
                "device_id": "hr_watch_01",
                "sensor_type": SensorType.HEART_RATE.value,
                "device_model": "FitSense X2",
                "connection_type": "bluetooth"
            },
            {
                "device_id": "smartphone_01",
                "sensor_type": SensorType.FACIAL_EXPRESSION.value,
                "device_model": "Smartphone Camera",
                "connection_type": "internal"
            },
            {
                "device_id": "bio_band_01",
                "sensor_type": SensorType.BIOIMPEDANCE.value,
                "device_model": "EmotionSense-Bio",
                "connection_type": "bluetooth"
            }
        ]
        
        return discovered
    
    def add_sensor(self, sensor: SensorDevice) -> bool:
        """
        Add a sensor to the manager.
        
        Args:
            sensor: Sensor device object
            
        Returns:
            bool: True if sensor was added successfully
        """
        sensor_id = f"{sensor.sensor_type.value}_{sensor.device_id}"
        
        if sensor_id in self.sensors:
            self.logger.warning(f"Sensor {sensor_id} already exists, updating")
            
        self.sensors[sensor_id] = sensor
        self.logger.info(f"Added sensor {sensor_id}")
        return True
    
    def remove_sensor(self, sensor_id: str) -> bool:
        """
        Remove a sensor from the manager.
        
        Args:
            sensor_id: ID of the sensor to remove
            
        Returns:
            bool: True if sensor was removed successfully
        """
        if sensor_id not in self.sensors:
            self.logger.warning(f"Sensor {sensor_id} not found")
            return False
        
        # Stop collection if this sensor is active
        if sensor_id in self.active_sensors:
            self.deactivate_sensor(sensor_id)
        
        del self.sensors[sensor_id]
        self.logger.info(f"Removed sensor {sensor_id}")
        return True
    
    def activate_sensor(self, sensor_id: str) -> bool:
        """
        Activate a sensor for data collection.
        
        Args:
            sensor_id: ID of the sensor to activate
            
        Returns:
            bool: True if sensor was activated successfully
        """
        if sensor_id not in self.sensors:
            self.logger.error(f"Sensor {sensor_id} not found")
            return False
        
        sensor = self.sensors[sensor_id]
        
        if not sensor.connected:
            success = sensor.connect()
            if not success:
                self.logger.error(f"Failed to connect to sensor {sensor_id}")
                return False
        
        self.active_sensors.add(sensor_id)
        self.logger.info(f"Activated sensor {sensor_id}")
        return True
    
    def deactivate_sensor(self, sensor_id: str) -> bool:
        """
        Deactivate a sensor from data collection.
        
        Args:
            sensor_id: ID of the sensor to deactivate
            
        Returns:
            bool: True if sensor was deactivated successfully
        """
        if sensor_id not in self.sensors:
            self.logger.error(f"Sensor {sensor_id} not found")
            return False
        
        if sensor_id in self.active_sensors:
            self.active_sensors.remove(sensor_id)
            self.logger.info(f"Deactivated sensor {sensor_id}")
        
        return True
    
    def register_data_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register a callback function to receive sensor data.
        
        Args:
            callback: Function to call with collected sensor data
        """
        self.data_callbacks.append(callback)
        self.logger.info("Registered new data callback")
    
    def start_collection(self) -> bool:
        """
        Start collecting data from all active sensors.
        
        Returns:
            bool: True if collection was started successfully
        """
        if self.is_collecting:
            self.logger.warning("Data collection already running")
            return False
        
        if not self.active_sensors:
            self.logger.warning("No active sensors to collect data from")
            return False
        
        self.is_collecting = True
        self.collection_thread = threading.Thread(target=self._collection_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        
        self.logger.info("Started data collection")
        return True
    
    def stop_collection(self) -> bool:
        """
        Stop collecting data from sensors.
        
        Returns:
            bool: True if collection was stopped successfully
        """
        if not self.is_collecting:
            self.logger.warning("Data collection not running")
            return False
        
        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
            self.collection_thread = None
        
        self.logger.info("Stopped data collection")
        return True
    
    def _collection_loop(self) -> None:
        """
        Main loop for collecting data from sensors.
        """
        while self.is_collecting:
            collection_start = time.time()
            all_data = {}
            
            # Collect data from each active sensor
            for sensor_id in self.active_sensors:
                try:
                    sensor = self.sensors[sensor_id]
                    sensor_data = sensor.read_data()
                    all_data[sensor_id] = sensor_data
                except Exception as e:
                    self.logger.error(f"Error reading from sensor {sensor_id}: {str(e)}")
            
            # If we have any data, process and dispatch it
            if all_data:
                # Add collection metadata
                all_data["collection_timestamp"] = time.time()
                all_data["sensors_count"] = len(all_data) - 1  # Excluding metadata
                
                # Call all registered callbacks with the data
                for callback in self.data_callbacks:
                    try:
                        callback(all_data)
                    except Exception as e:
                        self.logger.error(f"Error in data callback: {str(e)}")
            
            # Calculate sleep time to maintain collection interval
            elapsed = time.time() - collection_start
            sleep_time = max(0, self.collection_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def get_sensors_status(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status information for all managed sensors.
        
        Returns:
            Dict mapping sensor IDs to their status information
        """
        return {sensor_id: sensor.get_status() for sensor_id, sensor in self.sensors.items()}

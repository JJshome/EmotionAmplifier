"""
Emotion Data Collector Module

This module serves as the main interface for emotion data collection, managing different
sensor inputs and preprocessing the raw data for analysis.
"""
from src.data_collection import sensor_manager, data_integrator, bioimpedance, user_input

class EmotionDataCollector:
    """
    Manages the collection of emotion data from various sensors and user inputs.
    Acts as a facade for the more detailed implementation in the src/data_collection package.
    """
    
    def __init__(self, user_id, sensors=None, sampling_rate=1.0, storage_path="./data"):
        """
        Initialize the emotion data collector.
        
        Args:
            user_id: Unique identifier for the user
            sensors: List of sensors to use (e.g., "heart_rate", "skin_conductance")
            sampling_rate: Data collection frequency in Hz
            storage_path: Path to store collected data
        """
        self.user_id = user_id
        self.sensors = sensors or ["heart_rate", "skin_conductance", "facial_expression", "impedance"]
        self.sampling_rate = sampling_rate
        self.storage_path = storage_path
        
        # Initialize sensor manager
        self.sensor_manager = sensor_manager.SensorManager(
            user_id=user_id,
            enabled_sensors=self.sensors,
            sampling_rate=sampling_rate
        )
        
        # Initialize data integrator
        self.data_integrator = data_integrator.DataIntegrator(
            user_id=user_id,
            storage_path=storage_path
        )
        
        # Initialize user input handler
        self.user_input_handler = user_input.UserInputHandler(user_id=user_id)
        
    def start_collection(self):
        """
        Start collecting emotion data from all enabled sensors.
        
        Returns:
            bool: True if collection started successfully
        """
        return self.sensor_manager.start_all_sensors()
    
    def stop_collection(self):
        """
        Stop collecting emotion data and process the collected data.
        
        Returns:
            dict: Collected and preprocessed emotion data
        """
        # Stop all sensors
        raw_data = self.sensor_manager.stop_all_sensors()
        
        # Process and integrate data
        processed_data = self.data_integrator.process_sensor_data(raw_data)
        
        # Store the data
        self.data_integrator.store_data(processed_data)
        
        return processed_data
    
    def collect_user_input(self, prompt=None):
        """
        Collect explicit emotion data input from the user.
        
        Args:
            prompt: Optional prompt to display to the user
            
        Returns:
            dict: User-provided emotion data
        """
        return self.user_input_handler.collect_emotion_input(prompt)
    
    def get_bioimpedance_data(self, frequency_range=(1000, 100000), steps=10):
        """
        Collect bioimpedance data across a frequency range.
        
        Args:
            frequency_range: Tuple of (min_freq, max_freq) in Hz
            steps: Number of frequency steps to measure
            
        Returns:
            dict: Bioimpedance measurements across frequencies
        """
        bio_sensor = bioimpedance.BioimpedanceSensor(
            user_id=self.user_id,
            frequency_range=frequency_range,
            steps=steps
        )
        return bio_sensor.measure()
    
    def get_supported_sensors(self):
        """
        Get a list of all supported sensors.
        
        Returns:
            list: Names of supported sensors
        """
        return self.sensor_manager.get_available_sensors()
    
    def configure_sensor(self, sensor_name, **settings):
        """
        Configure settings for a specific sensor.
        
        Args:
            sensor_name: Name of the sensor to configure
            **settings: Sensor-specific settings
            
        Returns:
            bool: True if configuration was successful
        """
        return self.sensor_manager.configure_sensor(sensor_name, **settings)


# Example usage when module is run directly
if __name__ == "__main__":
    collector = EmotionDataCollector(user_id="test_user")
    print(f"Available sensors: {collector.get_supported_sensors()}")
    
    # Start collection
    print("Starting data collection...")
    collector.start_collection()
    
    # Simulate some collection time
    import time
    time.sleep(5)
    
    # Stop and get results
    print("Stopping data collection...")
    data = collector.stop_collection()
    
    # Display results summary
    print(f"Collected data for {len(data)} emotion dimensions")
    for emotion, value in data.items():
        print(f"  {emotion}: {value:.2f}")

#!/usr/bin/env python3
"""
EmotionAmplifier: Real-time Emotion Data-based Amplification and Sharing System

This is the main application entry point for the EmotionAmplifier system.
It initializes and coordinates all the system components, including data collection,
emotion analysis, content generation, and social sharing.
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Any

# Import system modules
from src.data_collection import (
    SensorManager, BioimpedanceProcessor, DataIntegrator, UserInputCollector
)
from src.emotion_analysis import (
    EmotionClassifier, MultimodalProcessor, PersonalEmotionModel
)
from src.content_generation import (
    ContentGenerator, HybridGenerator, ContentFormat, ContentType
)
from src.social_sharing import (
    ContentSharing, EmotionSynchronization, BlockchainManager
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('emotion_amplifier.log')
    ]
)

logger = logging.getLogger(__name__)

class EmotionAmplifierSystem:
    """
    Main system class that coordinates all EmotionAmplifier components.
    
    This class initializes and manages the interaction between data collection,
    emotion analysis, content generation, and social sharing modules.
    """
    
    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initialize the EmotionAmplifier system.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.user_id = self.config.get('user_id', 'default_user')
        
        # Initialize components
        self._init_data_collection()
        self._init_emotion_analysis()
        self._init_content_generation()
        self._init_social_sharing()
        
        # Runtime variables
        self.running = False
        self.last_emotion_data = None
        self.current_content = None
        
        logger.info("EmotionAmplifier system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load system configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Dict containing configuration values
        """
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                logger.info(f"Configuration loaded from {config_path}")
                return config
            else:
                logger.warning(f"Configuration file {config_path} not found, using defaults")
                return {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return {}
    
    def _init_data_collection(self) -> None:
        """Initialize data collection components."""
        # Get configuration
        data_collection_config = self.config.get('data_collection', {})
        
        # Create sensor manager
        self.sensor_manager = SensorManager()
        
        # Create bioimpedance processor
        bioimpedance_config = data_collection_config.get('bioimpedance', {})
        self.bioimpedance_processor = BioimpedanceProcessor(
            frequency_range=(
                bioimpedance_config.get('min_freq', 1000.0),
                bioimpedance_config.get('max_freq', 100000.0)
            ),
            frequency_steps=bioimpedance_config.get('freq_steps', 50),
            sample_rate=bioimpedance_config.get('sample_rate', 100.0)
        )
        
        # Create data integrator
        integrator_config = data_collection_config.get('integrator', {})
        self.data_integrator = DataIntegrator(
            window_size=integrator_config.get('window_size', 30.0),
            min_sensors_required=integrator_config.get('min_sensors', 1),
            storage_enabled=integrator_config.get('storage_enabled', True)
        )
        
        # Create user input collector
        self.user_input_collector = UserInputCollector(
            storage_enabled=data_collection_config.get('store_user_input', True)
        )
        
        # Connect data integrator to sensor manager
        self.sensor_manager.register_data_callback(self.data_integrator.process_sensor_data)
        
        logger.info("Data collection components initialized")
    
    def _init_emotion_analysis(self) -> None:
        """Initialize emotion analysis components."""
        # Get configuration
        emotion_analysis_config = self.config.get('emotion_analysis', {})
        
        # Create emotion classifier
        classifier_config = emotion_analysis_config.get('classifier', {})
        self.emotion_classifier = EmotionClassifier(
            primary_model=classifier_config.get('primary_model', 'plutchik'),
            secondary_model=classifier_config.get('secondary_model', 'pad')
        )
        
        # Create multimodal processor
        multimodal_config = emotion_analysis_config.get('multimodal', {})
        self.multimodal_processor = MultimodalProcessor(
            model_path=multimodal_config.get('model_path'),
            input_window=multimodal_config.get('input_window', 30),
            use_attention=multimodal_config.get('use_attention', True)
        )
        
        # Create personal emotion model
        personal_config = emotion_analysis_config.get('personal', {})
        self.personal_model = PersonalEmotionModel(
            user_id=self.user_id,
            base_model=self.multimodal_processor,
            model_dir=personal_config.get('model_dir', 'models/personal'),
            learning_rate=personal_config.get('learning_rate', 0.0005),
            personal_weight=personal_config.get('personal_weight', 0.7)
        )
        
        # Register emotion analysis callback to data integrator
        self.data_integrator.register_callback(self._process_emotion_data)
        
        logger.info("Emotion analysis components initialized")
    
    def _init_content_generation(self) -> None:
        """Initialize content generation components."""
        # Get configuration
        content_gen_config = self.config.get('content_generation', {})
        
        # Create hybrid generator
        generator_config = content_gen_config.get('generator', {})
        self.content_generator = HybridGenerator(
            model_path=generator_config.get('model_path'),
            gan_model_path=generator_config.get('gan_model_path'),
            vae_model_path=generator_config.get('vae_model_path'),
            use_gpu=generator_config.get('use_gpu', False)
        )
        
        logger.info("Content generation components initialized")
    
    def _init_social_sharing(self) -> None:
        """Initialize social sharing components."""
        # Get configuration
        social_config = self.config.get('social_sharing', {})
        
        # Create blockchain manager
        blockchain_config = social_config.get('blockchain', {})
        self.blockchain_manager = BlockchainManager(
            user_id=self.user_id,
            chain_type=blockchain_config.get('chain_type', 'hyperledger'),
            chain_location=blockchain_config.get('chain_location', 'local')
        )
        
        # Create content sharing manager
        sharing_config = social_config.get('sharing', {})
        self.content_sharing = ContentSharing(
            user_id=self.user_id,
            blockchain_manager=self.blockchain_manager,
            server_url=sharing_config.get('server_url', 'https://emotionamplifier.example.com/api'),
            auto_share=sharing_config.get('auto_share', False)
        )
        
        # Create emotion synchronization
        sync_config = social_config.get('synchronization', {})
        self.emotion_sync = EmotionSynchronization(
            user_id=self.user_id,
            sync_interval=sync_config.get('sync_interval', 5.0),
            matching_threshold=sync_config.get('matching_threshold', 0.7)
        )
        
        logger.info("Social sharing components initialized")
    
    def _process_emotion_data(self, integrated_data: Dict[str, Any]) -> None:
        """
        Process integrated data through emotion analysis pipeline.
        
        Args:
            integrated_data: Integrated sensor and user input data
        """
        # Analyze emotions using personal model
        emotion_data = self.personal_model.analyze_emotions(integrated_data)
        
        # Store the latest emotion data
        self.last_emotion_data = emotion_data
        
        # Log emotion state
        if "dominant_emotion" in emotion_data:
            logger.info(f"Detected emotion: {emotion_data['dominant_emotion']}")
        
        # Generate content based on emotion
        self._generate_content(emotion_data)
    
    def _generate_content(self, emotion_data: Dict[str, Any]) -> None:
        """
        Generate content based on emotion data.
        
        Args:
            emotion_data: Emotion analysis results
        """
        # Get user preferences
        user_preferences = self.user_input_collector.get_preferences()
        
        # Generate content
        content = self.content_generator.generate_for_emotion(
            emotion_data=emotion_data,
            user_preferences=user_preferences
        )
        
        # Store current content
        self.current_content = content
        
        # Log content generation
        if "content_format" in content:
            logger.info(f"Generated {content['content_format']} content")
        
        # Handle content sharing if enabled
        if self.config.get('social_sharing', {}).get('auto_share', False):
            self._share_content(content, emotion_data)
    
    def _share_content(self, 
                      content: Dict[str, Any], 
                      emotion_data: Dict[str, Any]) -> None:
        """
        Share content and emotion data with others.
        
        Args:
            content: Generated content
            emotion_data: Emotion analysis results
        """
        # Find matching users
        matching_users = self.emotion_sync.find_matching_users(emotion_data)
        
        if matching_users:
            # Share content with matching users
            self.content_sharing.share_content(
                content=content,
                emotion_data=emotion_data,
                recipients=matching_users
            )
            
            logger.info(f"Shared content with {len(matching_users)} users")
    
    def start(self) -> None:
        """Start the EmotionAmplifier system."""
        if self.running:
            logger.warning("System is already running")
            return
        
        logger.info("Starting EmotionAmplifier system")
        self.running = True
        
        # Activate sensors for data collection
        available_sensors = self.sensor_manager.discover_sensors()
        
        for sensor_info in available_sensors:
            # Create and add sensors
            sensor_type = sensor_info["sensor_type"]
            device_id = sensor_info["device_id"]
            
            if "bioimpedance" in sensor_type:
                # Special handling for bioimpedance sensor
                logger.info(f"Adding bioimpedance sensor: {device_id}")
                # In a real implementation, this would create the appropriate sensor object
            else:
                logger.info(f"Adding sensor: {device_id} ({sensor_type})")
                # In a real implementation, this would create the appropriate sensor object
        
        # Start sensor data collection
        self.sensor_manager.start_collection()
        
        # Start emotion synchronization
        self.emotion_sync.start_sync()
        
        logger.info("EmotionAmplifier system running")
    
    def stop(self) -> None:
        """Stop the EmotionAmplifier system."""
        if not self.running:
            logger.warning("System is not running")
            return
        
        logger.info("Stopping EmotionAmplifier system")
        
        # Stop data collection
        self.sensor_manager.stop_collection()
        
        # Stop emotion synchronization
        self.emotion_sync.stop_sync()
        
        # Save user data and models
        self._save_user_data()
        
        self.running = False
        logger.info("EmotionAmplifier system stopped")
    
    def _save_user_data(self) -> None:
        """Save user data and models for future sessions."""
        # Save personal emotion model
        model_dir = self.config.get('emotion_analysis', {}).get('personal', {}).get('model_dir', 'models/personal')
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, f"personal_model_{self.user_id}")
        self.personal_model.save_model(model_path)
        
        # Save user input history
        data_dir = self.config.get('data_collection', {}).get('data_dir', 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        input_path = os.path.join(data_dir, f"user_input_{self.user_id}.json")
        self.user_input_collector.save_input_history(input_path)
        
        # Save content generator history
        content_path = os.path.join(data_dir, f"content_history_{self.user_id}.json")
        self.content_generator.save_generation_history(content_path)
        
        logger.info("User data and models saved")

def main():
    """Main function to run the EmotionAmplifier system."""
    parser = argparse.ArgumentParser(description='EmotionAmplifier System')
    parser.add_argument('--config', type=str, default='config/config.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    try:
        # Create and start the system
        system = EmotionAmplifierSystem(config_path=args.config)
        system.start()
        
        # Run until interrupted
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down EmotionAmplifier system...")
        finally:
            system.stop()
            
    except Exception as e:
        logger.error(f"Error running EmotionAmplifier system: {str(e)}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

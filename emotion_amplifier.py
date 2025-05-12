"""
EmotionAmplifier Main Application
This is the main entry point for the Emotion Amplifier system, integrating all modules
to provide a complete emotion data collection, analysis, amplification, and sharing platform.
"""
import logging
import argparse
import json
import os
import time
from typing import Dict, List, Any, Optional
import sys

# Import all the components
from emotion_collector import EmotionDataCollector
from emotion_analyzer import EmotionAnalyzer
from content_generator import EmotionContentGenerator
from social_sharing import SocialSharingPlatform
from blockchain_manager import BlockchainManager

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("emotion_amplifier.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class EmotionAmplifierSystem:
    """
    Main class that integrates all components of the Emotion Amplifier system.
    """
    def __init__(self, user_id: str, config_path: Optional[str] = None):
        """
        Initialize the Emotion Amplifier system.
        
        Args:
            user_id: Unique identifier for the user
            config_path: Path to configuration file (optional)
        """
        self.user_id = user_id
        self.config = self._load_config(config_path)
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Emotion Amplifier system initialized for user {user_id}")
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """
        Load configuration from file or use defaults.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        default_config = {
            "data_collection": {
                "sensors": ["heart_rate", "skin_conductance", "facial_expression", "impedance"],
                "sampling_rate": 1.0,  # Hz
                "storage_path": "./data"
            },
            "analysis": {
                "model_path": "./models/emotion_model.h5",
                "use_multimodal": True,
                "emotion_categories": ["joy", "sadness", "anger", "fear", "surprise", "disgust", "trust", "anticipation"]
            },
            "content_generation": {
                "gan_model_path": "./models/gan_model.h5",
                "gpt_model_path": "./models/gpt_model.h5",
                "content_types": ["image", "text", "music"],
                "personalization_level": 0.8
            },
            "social_sharing": {
                "platform_url": "https://emotion-amplifier.social",
                "default_privacy": "friends",
                "content_expiry": 86400  # 24 hours
            },
            "blockchain": {
                "chain_type": "hyperledger",
                "chain_location": "local"
            }
        }
        
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    # Merge user config with defaults
                    for section, settings in user_config.items():
                        if section in default_config:
                            default_config[section].update(settings)
                        else:
                            default_config[section] = settings
                logger.info(f"Loaded configuration from {config_path}")
            except Exception as e:
                logger.error(f"Error loading configuration: {e}")
        
        return default_config
    
    def _init_components(self):
        """Initialize all system components with the loaded configuration."""
        # Initialize emotion data collector
        self.collector = EmotionDataCollector(
            user_id=self.user_id,
            sensors=self.config["data_collection"]["sensors"],
            sampling_rate=self.config["data_collection"]["sampling_rate"],
            storage_path=self.config["data_collection"]["storage_path"]
        )
        
        # Initialize emotion analyzer
        self.analyzer = EmotionAnalyzer(
            model_path=self.config["analysis"]["model_path"],
            use_multimodal=self.config["analysis"]["use_multimodal"],
            emotion_categories=self.config["analysis"]["emotion_categories"]
        )
        
        # Initialize content generator
        self.generator = EmotionContentGenerator(
            user_id=self.user_id,
            gan_model_path=self.config["content_generation"]["gan_model_path"],
            gpt_model_path=self.config["content_generation"]["gpt_model_path"],
            content_types=self.config["content_generation"]["content_types"],
            personalization_level=self.config["content_generation"]["personalization_level"]
        )
        
        # Initialize social sharing platform
        self.social = SocialSharingPlatform(
            user_id=self.user_id,
            platform_url=self.config["social_sharing"]["platform_url"]
        )
        
        # Initialize blockchain manager
        self.blockchain = BlockchainManager(
            user_id=self.user_id,
            chain_type=self.config["blockchain"]["chain_type"],
            chain_location=self.config["blockchain"]["chain_location"]
        )
    
    def collect_and_analyze_emotion(self, duration: float = 30.0) -> Dict[str, float]:
        """
        Collect and analyze emotion data.
        
        Args:
            duration: Duration in seconds to collect data
            
        Returns:
            Analyzed emotion data
        """
        logger.info(f"Collecting emotion data for {duration} seconds...")
        
        # Start data collection
        self.collector.start_collection()
        
        # Collect for specified duration
        time.sleep(duration)
        
        # Stop collection and get data
        emotion_data = self.collector.stop_collection()
        
        # Analyze emotion data
        analysis_result = self.analyzer.analyze_emotions(emotion_data)
        
        logger.info(f"Emotion analysis complete: {analysis_result}")
        
        # Update social profile with new emotion data
        self.social.update_emotion_profile(analysis_result)
        
        # Register on blockchain
        self.blockchain.register_emotion_data(analysis_result)
        
        return analysis_result
    
    def generate_emotion_content(self, emotion_data: Dict[str, float],
                               content_type: str = "auto") -> Dict[str, Any]:
        """
        Generate emotion-amplified content.
        
        Args:
            emotion_data: Emotion data to base content on
            content_type: Type of content to generate ("image", "text", "music", or "auto")
            
        Returns:
            Generated content
        """
        logger.info(f"Generating {content_type} content based on emotions...")
        
        # Determine content type if set to auto
        if content_type == "auto":
            # Choose based on dominant emotion
            dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
            if dominant_emotion in ["joy", "surprise"]:
                content_type = "image"
            elif dominant_emotion in ["sadness", "fear"]:
                content_type = "music"
            else:
                content_type = "text"
        
        # Generate content
        content = self.generator.generate_content(
            emotion_data=emotion_data,
            content_type=content_type
        )
        
        # Register on blockchain
        self.blockchain.register_content(content, content_type)
        
        logger.info(f"Generated {content_type} content")
        return content
    
    def share_emotion_content(self, content: Dict[str, Any], 
                           emotion_data: Dict[str, float],
                           privacy_level: str = "friends") -> str:
        """
        Share generated content on the social platform.
        
        Args:
            content: Content to share
            emotion_data: Emotion data associated with the content
            privacy_level: Privacy level for sharing
            
        Returns:
            share_id: ID of the shared content
        """
        logger.info(f"Sharing content with privacy level: {privacy_level}")
        
        # Share content
        share_id = self.social.share_content(
            content_data=content,
            emotion_context=emotion_data,
            privacy_level=privacy_level,
            duration=self.config["social_sharing"]["content_expiry"]
        )
        
        logger.info(f"Content shared with ID: {share_id}")
        return share_id
    
    def find_emotional_matches(self, emotion_data: Dict[str, float], 
                            threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Find users with matching emotional states.
        
        Args:
            emotion_data: Current emotion data
            threshold: Similarity threshold
            
        Returns:
            List of matching users
        """
        logger.info(f"Finding emotional matches with threshold {threshold}...")
        
        # Find matches
        matches = self.social.find_emotion_matches(
            emotion_data=emotion_data,
            threshold=threshold
        )
        
        logger.info(f"Found {len(matches)} emotional matches")
        return matches
    
    def create_emotion_session(self, name: str, 
                            emotion_data: Dict[str, float],
                            description: str = "") -> str:
        """
        Create an emotional synchronization session.
        
        Args:
            name: Session name
            emotion_data: Emotion theme for the session
            description: Session description
            
        Returns:
            session_id: ID of the created session
        """
        logger.info(f"Creating emotion session: {name}")
        
        # Create room
        room_id = self.social.create_emotion_room(
            name=name,
            emotion_theme=emotion_data,
            description=description
        )
        
        logger.info(f"Emotion session created with ID: {room_id}")
        return room_id
    
    def run_complete_pipeline(self, duration: float = 30.0, 
                           content_type: str = "auto",
                           share: bool = True) -> Dict[str, Any]:
        """
        Run the complete emotion amplification pipeline.
        
        Args:
            duration: Duration to collect emotion data
            content_type: Type of content to generate
            share: Whether to share the generated content
            
        Returns:
            Dictionary with results from each stage
        """
        logger.info("Starting complete emotion amplification pipeline...")
        
        # Collect and analyze emotions
        emotion_data = self.collect_and_analyze_emotion(duration)
        
        # Generate content
        content = self.generate_emotion_content(emotion_data, content_type)
        
        # Share if requested
        share_id = None
        if share:
            share_id = self.share_emotion_content(
                content=content,
                emotion_data=emotion_data,
                privacy_level=self.config["social_sharing"]["default_privacy"]
            )
        
        # Find emotional matches
        matches = self.find_emotional_matches(emotion_data)
        
        results = {
            "emotions": emotion_data,
            "content": {
                "type": content.get("type", "unknown"),
                "summary": content.get("summary", "No summary")
            },
            "share_id": share_id,
            "matches_count": len(matches)
        }
        
        logger.info("Complete pipeline finished successfully")
        return results

def main():
    """Main function to run the Emotion Amplifier system from command line."""
    parser = argparse.ArgumentParser(description="Emotion Amplifier System")
    parser.add_argument("--user", type=str, required=True, help="User ID")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--duration", type=float, default=30.0, help="Data collection duration in seconds")
    parser.add_argument("--content", type=str, default="auto", choices=["auto", "image", "text", "music"],
                       help="Type of content to generate")
    parser.add_argument("--share", action="store_true", help="Share generated content")
    
    args = parser.parse_args()
    
    # Initialize the system
    system = EmotionAmplifierSystem(user_id=args.user, config_path=args.config)
    
    # Run the pipeline
    results = system.run_complete_pipeline(
        duration=args.duration,
        content_type=args.content,
        share=args.share
    )
    
    # Print results
    print("\n=== Emotion Amplifier Results ===")
    print(f"Detected emotions: {results['emotions']}")
    print(f"Generated content type: {results['content']['type']}")
    print(f"Content summary: {results['content']['summary']}")
    
    if results['share_id']:
        print(f"Content shared with ID: {results['share_id']}")
        print(f"Found {results['matches_count']} emotional matches")
    
    print("===============================\n")

if __name__ == "__main__":
    main()

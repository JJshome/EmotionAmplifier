"""
Content Generator Module

This module creates personalized, emotion-based content using generative models
including GANs, VAEs, and language models.
"""
from src.content_generation import content_generator, content_types, hybrid_generator

class EmotionContentGenerator:
    """
    Generates personalized content based on emotional data to amplify or modulate emotions.
    Acts as a facade for the more detailed implementation in the src/content_generation package.
    """
    
    def __init__(self, user_id, gan_model_path=None, gpt_model_path=None, 
                content_types=None, personalization_level=0.8):
        """
        Initialize the content generator.
        
        Args:
            user_id: Unique identifier for the user
            gan_model_path: Path to pre-trained GAN model
            gpt_model_path: Path to pre-trained language model
            content_types: List of supported content types to generate
            personalization_level: Level of personalization (0.0 to 1.0)
        """
        self.user_id = user_id
        self.gan_model_path = gan_model_path or "./models/gan_model.h5"
        self.gpt_model_path = gpt_model_path or "./models/gpt_model.h5"
        self.content_types = content_types or ["image", "text", "music"]
        self.personalization_level = personalization_level
        
        # Initialize content type handlers
        self.type_handlers = content_types.ContentTypeManager(self.content_types)
        
        # Initialize generator
        self.generator = content_generator.ContentGenerator(
            user_id=user_id, 
            model_paths={
                "gan": self.gan_model_path,
                "gpt": self.gpt_model_path
            }
        )
        
        # Initialize hybrid generator
        self.hybrid_generator = hybrid_generator.HybridGenerator(
            user_id=user_id,
            personalization_level=personalization_level
        )
    
    def generate_content(self, emotion_data, content_type="auto", preferences=None):
        """
        Generate content based on emotion data.
        
        Args:
            emotion_data: Dictionary containing emotion data
            content_type: Type of content to generate (image, text, music, or auto)
            preferences: Optional user preferences
            
        Returns:
            dict: Generated content with metadata
        """
        # Determine content type if set to auto
        if content_type == "auto":
            content_type = self._determine_optimal_content_type(emotion_data)
        
        # Validate content type
        if content_type not in self.content_types:
            raise ValueError(f"Unsupported content type: {content_type}. Supported types: {self.content_types}")
        
        # Process user preferences
        processed_preferences = self._process_preferences(preferences)
        
        # Generate content using hybrid generator
        content = self.hybrid_generator.generate(
            emotion_data=emotion_data,
            content_type=content_type,
            preferences=processed_preferences
        )
        
        # Add metadata
        content["type"] = content_type
        content["timestamp"] = self._get_timestamp()
        content["emotion_context"] = emotion_data
        
        return content
    
    def generate_multi_modal_content(self, emotion_data, preferences=None):
        """
        Generate multiple types of content based on emotion data.
        
        Args:
            emotion_data: Dictionary containing emotion data
            preferences: Optional user preferences
            
        Returns:
            dict: Dictionary mapping content types to generated content
        """
        result = {}
        
        for content_type in self.content_types:
            try:
                result[content_type] = self.generate_content(
                    emotion_data=emotion_data,
                    content_type=content_type,
                    preferences=preferences
                )
            except Exception as e:
                print(f"Error generating {content_type} content: {e}")
        
        return result
    
    def adapt_content(self, content, target_emotion_data):
        """
        Adapt existing content to target a different emotional state.
        
        Args:
            content: Original content
            target_emotion_data: Target emotion state
            
        Returns:
            dict: Adapted content
        """
        content_type = content.get("type", "unknown")
        
        if content_type not in self.content_types:
            raise ValueError(f"Unsupported content type: {content_type}")
        
        # Use the generator to adapt content
        adapted_content = self.generator.adapt_content(
            original_content=content,
            target_emotion=target_emotion_data
        )
        
        # Add metadata
        adapted_content["type"] = content_type
        adapted_content["timestamp"] = self._get_timestamp()
        adapted_content["emotion_context"] = target_emotion_data
        adapted_content["derived_from"] = content.get("id", None)
        
        return adapted_content
    
    def get_supported_content_types(self):
        """
        Get list of supported content types.
        
        Returns:
            list: Supported content types
        """
        return self.content_types
    
    def _determine_optimal_content_type(self, emotion_data):
        """
        Determine the optimal content type based on emotion data.
        
        Args:
            emotion_data: Dictionary containing emotion data
            
        Returns:
            str: Selected content type
        """
        # Find dominant emotion
        dominant_emotion = max(emotion_data.items(), key=lambda x: x[1])[0]
        
        # Map dominant emotions to content types
        emotion_to_content = {
            "joy": "image",
            "sadness": "music",
            "anger": "music",
            "fear": "text",
            "surprise": "image",
            "disgust": "text",
            "trust": "image",
            "anticipation": "text"
        }
        
        # Return mapped content type or default to "image"
        return emotion_to_content.get(dominant_emotion, "image")
    
    def _process_preferences(self, preferences):
        """
        Process and validate user preferences.
        
        Args:
            preferences: User preferences
            
        Returns:
            dict: Processed preferences
        """
        if not preferences:
            return {}
        
        return preferences
    
    def _get_timestamp(self):
        """
        Get current timestamp.
        
        Returns:
            float: Current timestamp
        """
        import time
        return time.time()


# Example usage when module is run directly
if __name__ == "__main__":
    # Sample emotion data
    sample_emotion = {
        "joy": 0.7,
        "sadness": 0.1,
        "anger": 0.05,
        "fear": 0.05,
        "surprise": 0.1
    }
    
    # Initialize generator
    generator = EmotionContentGenerator(user_id="test_user")
    
    # Generate content
    content = generator.generate_content(
        emotion_data=sample_emotion,
        content_type="auto"
    )
    
    # Display result
    print(f"Generated {content['type']} content")
    print(f"Summary: {content.get('summary', 'No summary available')}")
    
    # Generate content of specific type
    text_content = generator.generate_content(
        emotion_data=sample_emotion,
        content_type="text"
    )
    
    print(f"\nGenerated text content:")
    print(f"Summary: {text_content.get('summary', 'No summary available')}")

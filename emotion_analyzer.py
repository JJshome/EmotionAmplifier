"""
Emotion Analyzer Module

This module processes collected emotion data to identify and quantify emotional states
using multi-modal deep learning approaches.
"""
from src.emotion_analysis import emotion_classifier, emotion_model, multimodal_processor, personal_model

class EmotionAnalyzer:
    """
    Analyzes emotion data using various deep learning models to identify and quantify emotions.
    Acts as a facade for the more detailed implementation in the src/emotion_analysis package.
    """
    
    def __init__(self, model_path=None, use_multimodal=True, emotion_categories=None):
        """
        Initialize the emotion analyzer.
        
        Args:
            model_path: Path to pre-trained emotion model
            use_multimodal: Whether to use multi-modal analysis
            emotion_categories: List of emotion categories to analyze
        """
        self.model_path = model_path or "./models/emotion_model.h5"
        self.use_multimodal = use_multimodal
        self.emotion_categories = emotion_categories or [
            "joy", "sadness", "anger", "fear", 
            "surprise", "disgust", "trust", "anticipation"
        ]
        
        # Initialize the main emotion model
        self.model = emotion_model.EmotionModel(
            model_path=self.model_path,
            emotion_categories=self.emotion_categories
        )
        
        # Initialize multimodal processor if enabled
        if self.use_multimodal:
            self.multimodal_processor = multimodal_processor.MultimodalProcessor(
                emotion_categories=self.emotion_categories
            )
        else:
            self.multimodal_processor = None
        
        # Initialize emotion classifier
        self.classifier = emotion_classifier.EmotionClassifier(
            emotion_categories=self.emotion_categories
        )
        
    def analyze_emotions(self, emotion_data):
        """
        Analyze emotion data to identify and quantify emotions.
        
        Args:
            emotion_data: Dictionary containing collected emotion data
            
        Returns:
            dict: Analyzed emotions with intensity values
        """
        # Process through multimodal processor if enabled
        if self.use_multimodal and self.multimodal_processor:
            processed_data = self.multimodal_processor.process(emotion_data)
        else:
            processed_data = emotion_data
        
        # Extract features and analyze
        features = self.model.extract_features(processed_data)
        emotion_scores = self.classifier.classify(features)
        
        # Normalize scores
        normalized_scores = self._normalize_scores(emotion_scores)
        
        return normalized_scores
    
    def create_personal_model(self, user_id, base_model_path=None):
        """
        Create a personalized emotion model for a specific user.
        
        Args:
            user_id: Unique identifier for the user
            base_model_path: Path to base model to use for personalization
            
        Returns:
            PersonalModel: Personalized emotion model
        """
        base_model_path = base_model_path or self.model_path
        
        return personal_model.PersonalModel(
            user_id=user_id,
            base_model_path=base_model_path,
            emotion_categories=self.emotion_categories
        )
    
    def analyze_with_personal_model(self, emotion_data, personal_model_instance):
        """
        Analyze emotion data using a personalized model.
        
        Args:
            emotion_data: Dictionary containing collected emotion data
            personal_model_instance: PersonalModel instance
            
        Returns:
            dict: Analyzed emotions with intensity values
        """
        # Process through multimodal processor
        if self.use_multimodal and self.multimodal_processor:
            processed_data = self.multimodal_processor.process(emotion_data)
        else:
            processed_data = emotion_data
        
        # Use personal model for analysis
        emotion_scores = personal_model_instance.analyze(processed_data)
        
        # Normalize scores
        normalized_scores = self._normalize_scores(emotion_scores)
        
        return normalized_scores
    
    def get_emotion_trend(self, emotion_history, window_size=10):
        """
        Analyze emotion trends over time.
        
        Args:
            emotion_history: List of emotion data dictionaries over time
            window_size: Size of the sliding window for trend analysis
            
        Returns:
            dict: Emotion trends and change rates
        """
        return self.model.analyze_trend(emotion_history, window_size)
    
    def _normalize_scores(self, emotion_scores):
        """
        Normalize emotion scores to ensure they sum to 1.0.
        
        Args:
            emotion_scores: Raw emotion scores
            
        Returns:
            dict: Normalized emotion scores
        """
        total = sum(emotion_scores.values())
        if total > 0:
            return {emotion: score/total for emotion, score in emotion_scores.items()}
        return emotion_scores


# Example usage when module is run directly
if __name__ == "__main__":
    # Create sample emotion data
    sample_data = {
        "heart_rate": 85,
        "hrv": 45,
        "skin_conductance": 0.7,
        "facial_features": [0.2, 0.1, 0.3, 0.1, 0.05, 0.05],
        "voice_features": [0.3, 0.2, 0.1, 0.2, 0.1, 0.1],
        "user_report": {
            "joy": 0.6,
            "sadness": 0.2,
            "anger": 0.1,
            "fear": 0.1
        }
    }
    
    # Initialize analyzer
    analyzer = EmotionAnalyzer()
    
    # Analyze emotions
    result = analyzer.analyze_emotions(sample_data)
    
    # Display results
    print("Emotion Analysis Results:")
    for emotion, intensity in result.items():
        print(f"  {emotion}: {intensity:.2f}")

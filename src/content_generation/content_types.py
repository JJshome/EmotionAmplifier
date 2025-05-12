"""
Content Types Module

This module defines enumerations and data types for content generation,
including content types, formats, and modalities.
"""

from enum import Enum, auto
from typing import Dict, List, Set, Optional, Any

class ContentType(Enum):
    """Enum for content type categories"""
    VISUAL = "visual"
    AUDIO = "audio"
    TEXT = "text"
    HAPTIC = "haptic"
    MULTIMODAL = "multimodal"


class ContentFormat(Enum):
    """Enum for specific content formats"""
    # Visual formats
    IMAGE = "image"
    ANIMATION = "animation"
    COLOR_SCHEME = "color_scheme"
    PATTERN = "pattern"
    
    # Audio formats
    MUSIC = "music"
    SOUND_EFFECT = "sound_effect"
    AMBIENCE = "ambience"
    VOICE = "voice"
    
    # Text formats
    STORY = "story"
    POEM = "poem"
    AFFIRMATION = "affirmation"
    QUOTE = "quote"
    
    # Haptic formats
    VIBRATION_PATTERN = "vibration_pattern"
    PRESSURE_PATTERN = "pressure_pattern"
    TEMPERATURE = "temperature"
    
    # Multimodal formats
    AUDIO_VISUAL = "audio_visual"
    TEXT_IMAGE = "text_image"
    FULL_SENSORY = "full_sensory"


class ModalityType(Enum):
    """Enum for sensory modalities and emotional dimensions"""
    VISUAL = auto()
    AUDITORY = auto()
    TACTILE = auto()
    VERBAL = auto()
    AROUSAL = auto()
    VALENCE = auto()
    PLEASURE = auto()
    DOMINANCE = auto()


# Mapping between content formats and types
FORMAT_TO_TYPE_MAP: Dict[ContentFormat, ContentType] = {
    # Visual formats
    ContentFormat.IMAGE: ContentType.VISUAL,
    ContentFormat.ANIMATION: ContentType.VISUAL,
    ContentFormat.COLOR_SCHEME: ContentType.VISUAL,
    ContentFormat.PATTERN: ContentType.VISUAL,
    
    # Audio formats
    ContentFormat.MUSIC: ContentType.AUDIO,
    ContentFormat.SOUND_EFFECT: ContentType.AUDIO,
    ContentFormat.AMBIENCE: ContentType.AUDIO,
    ContentFormat.VOICE: ContentType.AUDIO,
    
    # Text formats
    ContentFormat.STORY: ContentType.TEXT,
    ContentFormat.POEM: ContentType.TEXT,
    ContentFormat.AFFIRMATION: ContentType.TEXT,
    ContentFormat.QUOTE: ContentType.TEXT,
    
    # Haptic formats
    ContentFormat.VIBRATION_PATTERN: ContentType.HAPTIC,
    ContentFormat.PRESSURE_PATTERN: ContentType.HAPTIC,
    ContentFormat.TEMPERATURE: ContentType.HAPTIC,
    
    # Multimodal formats
    ContentFormat.AUDIO_VISUAL: ContentType.MULTIMODAL,
    ContentFormat.TEXT_IMAGE: ContentType.MULTIMODAL,
    ContentFormat.FULL_SENSORY: ContentType.MULTIMODAL
}

# Mapping between emotions and recommended modalities
EMOTION_TO_MODALITY_MAP: Dict[str, List[ModalityType]] = {
    "joy": [ModalityType.VISUAL, ModalityType.AUDITORY, ModalityType.VERBAL],
    "trust": [ModalityType.TACTILE, ModalityType.VERBAL, ModalityType.AUDITORY],
    "fear": [ModalityType.AUDITORY, ModalityType.TACTILE, ModalityType.VISUAL],
    "surprise": [ModalityType.VISUAL, ModalityType.AUDITORY],
    "sadness": [ModalityType.AUDITORY, ModalityType.VERBAL],
    "disgust": [ModalityType.VISUAL, ModalityType.TACTILE],
    "anger": [ModalityType.TACTILE, ModalityType.AUDITORY],
    "anticipation": [ModalityType.VERBAL, ModalityType.VISUAL]
}

# Mapping between emotions and recommended content formats
EMOTION_TO_FORMAT_MAP: Dict[str, List[ContentFormat]] = {
    "joy": [
        ContentFormat.ANIMATION, ContentFormat.MUSIC, ContentFormat.AFFIRMATION,
        ContentFormat.COLOR_SCHEME, ContentFormat.AUDIO_VISUAL
    ],
    "trust": [
        ContentFormat.AMBIENCE, ContentFormat.QUOTE, ContentFormat.PRESSURE_PATTERN,
        ContentFormat.COLOR_SCHEME
    ],
    "fear": [
        ContentFormat.SOUND_EFFECT, ContentFormat.VIBRATION_PATTERN, ContentFormat.ANIMATION
    ],
    "surprise": [
        ContentFormat.ANIMATION, ContentFormat.SOUND_EFFECT, ContentFormat.AUDIO_VISUAL
    ],
    "sadness": [
        ContentFormat.MUSIC, ContentFormat.POEM, ContentFormat.AMBIENCE,
        ContentFormat.COLOR_SCHEME
    ],
    "disgust": [
        ContentFormat.PATTERN, ContentFormat.COLOR_SCHEME, ContentFormat.VIBRATION_PATTERN
    ],
    "anger": [
        ContentFormat.VIBRATION_PATTERN, ContentFormat.SOUND_EFFECT, ContentFormat.MUSIC
    ],
    "anticipation": [
        ContentFormat.STORY, ContentFormat.ANIMATION, ContentFormat.TEXT_IMAGE
    ]
}

# Mapping between dimensional emotions (quadrants) and content formats
QUADRANT_TO_FORMAT_MAP: Dict[str, List[ContentFormat]] = {
    "happy-excited": [
        ContentFormat.ANIMATION, ContentFormat.MUSIC, ContentFormat.AFFIRMATION,
        ContentFormat.AUDIO_VISUAL
    ],
    "relaxed-content": [
        ContentFormat.AMBIENCE, ContentFormat.COLOR_SCHEME, ContentFormat.POEM,
        ContentFormat.PRESSURE_PATTERN
    ],
    "angry-stressed": [
        ContentFormat.VIBRATION_PATTERN, ContentFormat.SOUND_EFFECT, ContentFormat.PATTERN
    ],
    "sad-depressed": [
        ContentFormat.MUSIC, ContentFormat.POEM, ContentFormat.COLOR_SCHEME,
        ContentFormat.AMBIENCE
    ]
}

def get_recommended_formats(emotion_data: Dict[str, Any]) -> List[ContentFormat]:
    """
    Get recommended content formats based on emotion data.
    
    Args:
        emotion_data: Dictionary containing emotion analysis results
        
    Returns:
        List of recommended ContentFormat values
    """
    recommended_formats = set()
    
    # Check for emotions
    if "emotions" in emotion_data:
        emotions = emotion_data["emotions"]
        
        # Get top 2 emotions
        top_emotions = sorted(
            emotions.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:2]
        
        # Add formats for top emotions
        for emotion, _ in top_emotions:
            if emotion in EMOTION_TO_FORMAT_MAP:
                for format_type in EMOTION_TO_FORMAT_MAP[emotion]:
                    recommended_formats.add(format_type)
    
    # Check for emotion quadrant
    if "quadrant" in emotion_data:
        quadrant = emotion_data["quadrant"]
        if quadrant in QUADRANT_TO_FORMAT_MAP:
            for format_type in QUADRANT_TO_FORMAT_MAP[quadrant]:
                recommended_formats.add(format_type)
    
    # Add a fallback if we didn't get any recommendations
    if not recommended_formats:
        recommended_formats.add(ContentFormat.COLOR_SCHEME)
        recommended_formats.add(ContentFormat.MUSIC)
        recommended_formats.add(ContentFormat.AFFIRMATION)
    
    return list(recommended_formats)


def get_format_by_name(format_name: str) -> Optional[ContentFormat]:
    """
    Get ContentFormat enum value by name.
    
    Args:
        format_name: Name of the content format
        
    Returns:
        ContentFormat enum value or None if not found
    """
    try:
        return ContentFormat(format_name)
    except ValueError:
        for fmt in ContentFormat:
            if fmt.value == format_name:
                return fmt
    return None


def get_type_for_format(content_format: ContentFormat) -> ContentType:
    """
    Get the ContentType for a given ContentFormat.
    
    Args:
        content_format: ContentFormat enum value
        
    Returns:
        ContentType enum value
    """
    return FORMAT_TO_TYPE_MAP.get(content_format, ContentType.MULTIMODAL)

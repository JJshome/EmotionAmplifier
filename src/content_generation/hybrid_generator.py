"""
Hybrid Generator Module

This module implements a hybrid generative model that combines GAN (Generative 
Adversarial Network) and VAE (Variational Autoencoder) approaches for creating
personalized emotion-amplifying content.
"""

import numpy as np
import tensorflow as tf
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import os
import time
import json
from .content_generator import ContentGenerator, ContentParameters
from .content_types import ContentType, ContentFormat, ModalityType
from .gan_generator import GANGenerator
from .vae_generator import VAEGenerator

logger = logging.getLogger(__name__)

class HybridGenerator(ContentGenerator):
    """
    Hybrid content generator combining GAN and VAE approaches.
    
    Leverages the strengths of both GAN (high-quality, realistic outputs)
    and VAE (structured representation, interpolation) models to create
    personalized emotion-amplifying content.
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 gan_model_path: Optional[str] = None,
                 vae_model_path: Optional[str] = None,
                 supported_formats: Optional[List[ContentFormat]] = None,
                 use_gpu: bool = False,
                 interpolation_weight: float = 0.5):
        """
        Initialize the hybrid generator.
        
        Args:
            model_path: Path to hybrid model files
            gan_model_path: Optional path to GAN model files
            vae_model_path: Optional path to VAE model files
            supported_formats: List of supported content formats
            use_gpu: Whether to use GPU for generation
            interpolation_weight: Weight for interpolation between GAN and VAE
        """
        # Default supported formats if none provided
        if supported_formats is None:
            supported_formats = [
                ContentFormat.IMAGE,
                ContentFormat.ANIMATION,
                ContentFormat.COLOR_SCHEME,
                ContentFormat.MUSIC,
                ContentFormat.AMBIENCE,
                ContentFormat.AFFIRMATION,
                ContentFormat.AUDIO_VISUAL
            ]
        
        super().__init__(
            generator_name="HybridGenerativeModel",
            supported_formats=supported_formats,
            model_path=model_path,
            use_gpu=use_gpu
        )
        
        # Initialize sub-generators
        self.gan_generator = GANGenerator(
            model_path=gan_model_path or (model_path + "/gan" if model_path else None),
            use_gpu=use_gpu
        )
        
        self.vae_generator = VAEGenerator(
            model_path=vae_model_path or (model_path + "/vae" if model_path else None),
            use_gpu=use_gpu
        )
        
        self.interpolation_weight = interpolation_weight
        
        # Latent space dimensions
        self.latent_dim = 128
        
        # Initialize internal models
        self.encoder = None
        self.decoder = None
        self.generator = None
        self.discriminator = None
        
        logger.info("Initialized hybrid generator")
    
    def load_models(self) -> bool:
        """
        Load hybrid generative models.
        
        Returns:
            bool: True if models were loaded successfully
        """
        try:
            # Load GAN and VAE sub-generators
            gan_loaded = self.gan_generator.load_models()
            vae_loaded = self.vae_generator.load_models()
            
            # If both sub-generators are loaded, we're good
            if gan_loaded and vae_loaded:
                self.models_loaded = True
                logger.info("Loaded hybrid generator models")
                return True
            
            # If we have a specific hybrid model path, try to load it
            if self.model_path and os.path.exists(self.model_path):
                # Load encoder (VAE part)
                encoder_path = os.path.join(self.model_path, "encoder")
                if os.path.exists(encoder_path):
                    self.encoder = tf.keras.models.load_model(encoder_path)
                    logger.info("Loaded encoder model")
                
                # Load decoder (VAE part)
                decoder_path = os.path.join(self.model_path, "decoder")
                if os.path.exists(decoder_path):
                    self.decoder = tf.keras.models.load_model(decoder_path)
                    logger.info("Loaded decoder model")
                
                # Load generator (GAN part)
                generator_path = os.path.join(self.model_path, "generator")
                if os.path.exists(generator_path):
                    self.generator = tf.keras.models.load_model(generator_path)
                    logger.info("Loaded generator model")
                
                # Load discriminator (GAN part)
                discriminator_path = os.path.join(self.model_path, "discriminator")
                if os.path.exists(discriminator_path):
                    self.discriminator = tf.keras.models.load_model(discriminator_path)
                    logger.info("Loaded discriminator model")
                
                # Check if all required models are loaded
                if self.encoder and self.decoder and self.generator:
                    self.models_loaded = True
                    logger.info("Loaded all hybrid generator models")
                    return True
            
            # If we get here, models weren't fully loaded
            if not gan_loaded:
                logger.warning("Failed to load GAN models")
            if not vae_loaded:
                logger.warning("Failed to load VAE models")
            
            self.models_loaded = False
            return False
            
        except Exception as e:
            logger.error(f"Error loading hybrid models: {str(e)}")
            self.models_loaded = False
            return False
    
    def generate_content(self, params: ContentParameters) -> Dict[str, Any]:
        """
        Generate content using the hybrid approach.
        
        Args:
            params: Content generation parameters
            
        Returns:
            Dict containing generated content
        """
        # Check if models are loaded
        if not self.models_loaded and not (self.gan_generator.models_loaded or self.vae_generator.models_loaded):
            logger.error("Models not loaded, cannot generate content")
            return {"error": "Models not loaded"}
        
        # Get content format
        content_format = params.content_format
        content_type = params.content_type
        
        # Decide which approach to use based on format and type
        if content_type == ContentType.VISUAL:
            # Visual content generation is best with the hybrid approach
            return self._generate_visual_content(params)
        elif content_type == ContentType.AUDIO:
            # Audio content might be better with GANs
            if self.gan_generator.models_loaded:
                return self.gan_generator.generate_content(params)
            else:
                return self._generate_audio_content(params)
        elif content_type == ContentType.TEXT:
            # Text content is better with VAEs for structure
            if self.vae_generator.models_loaded:
                return self.vae_generator.generate_content(params)
            else:
                return self._generate_text_content(params)
        elif content_type == ContentType.MULTIMODAL:
            # Multimodal content uses both approaches
            return self._generate_multimodal_content(params)
        else:
            # Default to VAE if available, else GAN
            if self.vae_generator.models_loaded:
                return self.vae_generator.generate_content(params)
            elif self.gan_generator.models_loaded:
                return self.gan_generator.generate_content(params)
            else:
                logger.error(f"Unsupported content type: {content_type}")
                return {"error": f"Unsupported content type: {content_type}"}
    
    def _generate_visual_content(self, params: ContentParameters) -> Dict[str, Any]:
        """
        Generate visual content using hybrid approach.
        
        Args:
            params: Content generation parameters
            
        Returns:
            Dict containing generated visual content
        """
        # Extract parameters
        valence = params.valence
        arousal = params.arousal
        intensity = params.emotion_intensity
        content_format = params.content_format
        
        # Create latent vector from emotion parameters
        latent_vector = self._emotion_to_latent(valence, arousal, intensity)
        
        # Generate content based on format
        if content_format == ContentFormat.IMAGE:
            return self._generate_image(latent_vector, params)
        elif content_format == ContentFormat.ANIMATION:
            return self._generate_animation(latent_vector, params)
        elif content_format == ContentFormat.COLOR_SCHEME:
            return self._generate_color_scheme(valence, arousal, intensity, params)
        elif content_format == ContentFormat.PATTERN:
            return self._generate_pattern(latent_vector, params)
        else:
            logger.warning(f"Unsupported visual format: {content_format}")
            # Fall back to GAN if available
            if self.gan_generator.models_loaded:
                return self.gan_generator.generate_content(params)
            return {"error": f"Unsupported visual format: {content_format}"}
    
    def _generate_audio_content(self, params: ContentParameters) -> Dict[str, Any]:
        """
        Generate audio content using hybrid approach.
        
        Args:
            params: Content generation parameters
            
        Returns:
            Dict containing generated audio content
        """
        # Extract parameters
        valence = params.valence
        arousal = params.arousal
        intensity = params.emotion_intensity
        content_format = params.content_format
        
        # Create latent vector from emotion parameters
        latent_vector = self._emotion_to_latent(valence, arousal, intensity)
        
        # Generate content based on format
        if content_format == ContentFormat.MUSIC:
            return self._generate_music(latent_vector, params)
        elif content_format == ContentFormat.AMBIENCE:
            return self._generate_ambience(latent_vector, params)
        elif content_format == ContentFormat.SOUND_EFFECT:
            return self._generate_sound_effect(valence, arousal, intensity, params)
        else:
            logger.warning(f"Unsupported audio format: {content_format}")
            # Fall back to GAN if available
            if self.gan_generator.models_loaded:
                return self.gan_generator.generate_content(params)
            return {"error": f"Unsupported audio format: {content_format}"}
    
    def _generate_text_content(self, params: ContentParameters) -> Dict[str, Any]:
        """
        Generate text content using hybrid approach.
        
        Args:
            params: Content generation parameters
            
        Returns:
            Dict containing generated text content
        """
        # For text, we primarily use the VAE component
        if self.vae_generator.models_loaded:
            return self.vae_generator.generate_content(params)
        
        # Fallback implementation if VAE is not available
        content_format = params.content_format
        dominant_emotion = params.dominant_emotion
        
        # Generate simple text based on emotion and format
        if content_format == ContentFormat.AFFIRMATION:
            affirmations = {
                "joy": "You are filled with radiant joy that brightens everyone around you.",
                "trust": "You are worthy of trust and capable of trusting yourself.",
                "fear": "Your courage is greater than any fear you face.",
                "surprise": "Life is full of wonderful surprises waiting to be discovered.",
                "sadness": "It's okay to feel sad; this feeling will pass like clouds in the sky.",
                "disgust": "You can transform negativity into motivation for positive change.",
                "anger": "Your passion can be channeled into powerful, constructive energy.",
                "anticipation": "Exciting possibilities await you just beyond the horizon."
            }
            
            text = affirmations.get(dominant_emotion, "You are exactly where you need to be right now.")
            
        elif content_format == ContentFormat.QUOTE:
            quotes = {
                "joy": "Happiness is not something ready-made. It comes from your own actions. - Dalai Lama",
                "trust": "Trust yourself. Create the kind of self that you will be happy to live with all your life. - Golda Meir",
                "fear": "Fear is only as deep as the mind allows. - Japanese Proverb",
                "surprise": "Life is full of surprises and serendipity. Being open to unexpected turns in the road is an important part of success. - Henry Ford",
                "sadness": "The word 'happiness' would lose its meaning if it were not balanced by sadness. - Carl Jung",
                "disgust": "What we see depends mainly on what we look for. - John Lubbock",
                "anger": "Anger is an acid that can do more harm to the vessel in which it is stored than to anything on which it is poured. - Mark Twain",
                "anticipation": "The best way to predict the future is to create it. - Abraham Lincoln"
            }
            
            text = quotes.get(dominant_emotion, "The future belongs to those who believe in the beauty of their dreams. - Eleanor Roosevelt")
            
        else:
            text = f"Generated text content for {dominant_emotion} emotion in {content_format.value} format."
        
        # Create content object
        return self.create_content_object(
            content_data=text,
            content_format=content_format,
            params=params,
            metadata={"generation_method": "fallback_text_generation"}
        )
    
    def _generate_multimodal_content(self, params: ContentParameters) -> Dict[str, Any]:
        """
        Generate multimodal content using hybrid approach.
        
        Args:
            params: Content generation parameters
            
        Returns:
            Dict containing generated multimodal content
        """
        content_format = params.content_format
        
        # Generate components for multimodal content
        if content_format == ContentFormat.AUDIO_VISUAL:
            # Generate visual and audio components
            visual_params = ContentParameters(
                content_format=ContentFormat.IMAGE,
                emotion_data=params.emotion_data,
                user_preferences=params.user_preferences,
                content_settings=params.content_settings
            )
            
            audio_params = ContentParameters(
                content_format=ContentFormat.MUSIC,
                emotion_data=params.emotion_data,
                user_preferences=params.user_preferences,
                content_settings=params.content_settings
            )
            
            visual_content = self._generate_visual_content(visual_params)
            audio_content = self._generate_audio_content(audio_params)
            
            # Check for errors
            if "error" in visual_content or "error" in audio_content:
                error_msg = visual_content.get("error", "") or audio_content.get("error", "")
                return {"error": f"Error generating multimodal content: {error_msg}"}
            
            # Combine the components
            multimodal_data = {
                "visual": visual_content.get("content_data"),
                "audio": audio_content.get("content_data")
            }
            
            # Create content object
            return self.create_content_object(
                content_data=multimodal_data,
                content_format=content_format,
                params=params,
                metadata={
                    "visual_metadata": visual_content.get("metadata", {}),
                    "audio_metadata": audio_content.get("metadata", {})
                }
            )
            
        elif content_format == ContentFormat.TEXT_IMAGE:
            # Generate text and image components
            text_params = ContentParameters(
                content_format=ContentFormat.AFFIRMATION,
                emotion_data=params.emotion_data,
                user_preferences=params.user_preferences,
                content_settings=params.content_settings
            )
            
            image_params = ContentParameters(
                content_format=ContentFormat.IMAGE,
                emotion_data=params.emotion_data,
                user_preferences=params.user_preferences,
                content_settings=params.content_settings
            )
            
            text_content = self._generate_text_content(text_params)
            image_content = self._generate_visual_content(image_params)
            
            # Check for errors
            if "error" in text_content or "error" in image_content:
                error_msg = text_content.get("error", "") or image_content.get("error", "")
                return {"error": f"Error generating multimodal content: {error_msg}"}
            
            # Combine the components
            multimodal_data = {
                "text": text_content.get("content_data"),
                "image": image_content.get("content_data")
            }
            
            # Create content object
            return self.create_content_object(
                content_data=multimodal_data,
                content_format=content_format,
                params=params,
                metadata={
                    "text_metadata": text_content.get("metadata", {}),
                    "image_metadata": image_content.get("metadata", {})
                }
            )
            
        else:
            logger.warning(f"Unsupported multimodal format: {content_format}")
            return {"error": f"Unsupported multimodal format: {content_format}"}
    
    def _emotion_to_latent(self, 
                          valence: float, 
                          arousal: float, 
                          intensity: float) -> np.ndarray:
        """
        Convert emotion parameters to latent space vector.
        
        Args:
            valence: Emotional valence (-1 to 1)
            arousal: Emotional arousal (-1 to 1)
            intensity: Emotional intensity (0 to 1)
            
        Returns:
            Latent space vector
        """
        # Initialize with random noise
        z = np.random.normal(0, 1, self.latent_dim)
        
        # Modify first few dimensions based on emotion parameters
        z[0] = valence * intensity * 2  # Scale to approximately -2 to 2
        z[1] = arousal * intensity * 2
        
        # Additional emotion encoding (simplified version)
        # High valence -> increase certain features
        if valence > 0:
            z[2] = valence * 1.5
            z[3] = max(0, valence - 0.3) * 2
        else:
            z[4] = abs(valence) * 1.5
            z[5] = max(0, abs(valence) - 0.3) * 2
        
        # High arousal -> increase certain features
        if arousal > 0:
            z[6] = arousal * 1.5
            z[7] = max(0, arousal - 0.3) * 2
        else:
            z[8] = abs(arousal) * 1.5
            z[9] = max(0, abs(arousal) - 0.3) * 2
        
        # Scale by intensity
        z[2:10] *= intensity
        
        return z
    
    # Implementation of specific content generation methods
    
    def _generate_image(self, 
                       latent_vector: np.ndarray, 
                       params: ContentParameters) -> Dict[str, Any]:
        """Generate an image using hybrid approach"""
        # Placeholder implementation
        # In a real implementation, this would use the GAN generator with VAE-guided latent space
        
        # Get relevant parameters
        valence = params.valence
        arousal = params.arousal
        
        # Use different base colors depending on emotional quadrant
        if valence >= 0 and arousal >= 0:  # Happy-excited
            base_color = [240, 180, 20]  # Yellow/gold
        elif valence >= 0 and arousal < 0:  # Relaxed-content
            base_color = [100, 200, 180]  # Teal/turquoise
        elif valence < 0 and arousal >= 0:  # Angry-stressed
            base_color = [220, 40, 40]  # Red
        else:  # Sad-depressed
            base_color = [80, 80, 220]  # Blue
        
        # Scale intensity
        intensity = params.emotion_intensity
        
        # Create simple image data (base64 placeholder)
        image_data = f"base64_encoded_image_data_for_{params.dominant_emotion}_emotion"
        
        # Image dimensions
        width = params.get_parameter("width", 512)
        height = params.get_parameter("height", 512)
        
        # Metadata
        metadata = {
            "width": width,
            "height": height,
            "base_color": base_color,
            "generation_method": "hybrid_image_generation",
            "latent_dimensions": self.latent_dim,
            "interpolation_weight": self.interpolation_weight
        }
        
        # Create content object
        return self.create_content_object(
            content_data=image_data,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )
    
    def _generate_animation(self, 
                           latent_vector: np.ndarray, 
                           params: ContentParameters) -> Dict[str, Any]:
        """Generate an animation using hybrid approach"""
        # Placeholder implementation
        
        # Animation data (would be a sequence of frames in real implementation)
        animation_data = f"base64_encoded_animation_data_for_{params.dominant_emotion}_emotion"
        
        # Animation parameters
        duration = params.get_parameter("duration", 5.0)  # seconds
        frame_rate = params.get_parameter("frame_rate", 30)  # fps
        
        # Metadata
        metadata = {
            "duration": duration,
            "frame_rate": frame_rate,
            "frame_count": int(duration * frame_rate),
            "generation_method": "hybrid_animation_generation",
            "interpolation_weight": self.interpolation_weight
        }
        
        # Create content object
        return self.create_content_object(
            content_data=animation_data,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )
    
    def _generate_color_scheme(self, 
                              valence: float, 
                              arousal: float, 
                              intensity: float,
                              params: ContentParameters) -> Dict[str, Any]:
        """Generate a color scheme based on emotion parameters"""
        # This is a simplified implementation that creates color schemes
        # based on emotional parameters
        
        # Base hue selection based on emotion
        if valence > 0.5:  # Very positive
            base_hue = 60  # Yellow
        elif valence > 0:  # Positive
            base_hue = 120  # Green
        elif valence > -0.5:  # Slightly negative
            base_hue = 240  # Blue
        else:  # Very negative
            base_hue = 280  # Purple
        
        # Adjust for arousal
        if arousal > 0:
            base_hue = (base_hue + 30 * arousal) % 360  # Shift toward warmer colors
        else:
            base_hue = (base_hue - 30 * abs(arousal)) % 360  # Shift toward cooler colors
        
        # Calculate saturation and brightness based on intensity
        saturation = 0.3 + (0.7 * intensity)
        brightness = 0.4 + (0.6 * intensity)
        
        # Create complementary and analogous colors
        colors = []
        
        # Main color
        colors.append(self._hsv_to_rgb(base_hue, saturation, brightness))
        
        # Complementary color
        complementary_hue = (base_hue + 180) % 360
        colors.append(self._hsv_to_rgb(complementary_hue, saturation * 0.9, brightness * 0.9))
        
        # Analogous colors
        analog1_hue = (base_hue + 30) % 360
        analog2_hue = (base_hue - 30) % 360
        colors.append(self._hsv_to_rgb(analog1_hue, saturation * 0.8, brightness * 1.1))
        colors.append(self._hsv_to_rgb(analog2_hue, saturation * 0.8, brightness * 1.1))
        
        # Accent color
        accent_hue = (complementary_hue + 30) % 360
        colors.append(self._hsv_to_rgb(accent_hue, saturation * 1.2, brightness * 0.8))
        
        # Create color scheme data
        color_scheme = {
            "main": colors[0],
            "complementary": colors[1],
            "analogous": [colors[2], colors[3]],
            "accent": colors[4],
            "palette": colors
        }
        
        # Metadata
        metadata = {
            "base_hue": base_hue,
            "saturation": saturation,
            "brightness": brightness,
            "generation_method": "emotion_based_color_scheme"
        }
        
        # Create content object
        return self.create_content_object(
            content_data=color_scheme,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )
    
    def _hsv_to_rgb(self, h: float, s: float, v: float) -> List[int]:
        """Convert HSV color to RGB"""
        h = h % 360
        h_i = int(h / 60)
        f = h / 60 - h_i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        
        if h_i == 0:
            r, g, b = v, t, p
        elif h_i == 1:
            r, g, b = q, v, p
        elif h_i == 2:
            r, g, b = p, v, t
        elif h_i == 3:
            r, g, b = p, q, v
        elif h_i == 4:
            r, g, b = t, p, v
        else:
            r, g, b = v, p, q
        
        return [int(r * 255), int(g * 255), int(b * 255)]
    
    def _generate_pattern(self, 
                         latent_vector: np.ndarray, 
                         params: ContentParameters) -> Dict[str, Any]:
        """Generate a pattern using hybrid approach"""
        # Placeholder implementation
        pattern_data = f"base64_encoded_pattern_data_for_{params.dominant_emotion}_emotion"
        
        # Metadata
        metadata = {
            "pattern_type": "procedural",
            "generation_method": "hybrid_pattern_generation"
        }
        
        # Create content object
        return self.create_content_object(
            content_data=pattern_data,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )
    
    def _generate_music(self, 
                       latent_vector: np.ndarray, 
                       params: ContentParameters) -> Dict[str, Any]:
        """Generate music using hybrid approach"""
        # Placeholder implementation
        music_data = f"base64_encoded_music_data_for_{params.dominant_emotion}_emotion"
        
        # Music parameters
        duration = params.get_parameter("duration", 60.0)  # seconds
        tempo = 60 + int(params.arousal * 80)  # bpm: 20-140
        
        # Metadata
        metadata = {
            "duration": duration,
            "tempo": tempo,
            "key": "C major",
            "format": "mp3",
            "generation_method": "hybrid_music_generation"
        }
        
        # Create content object
        return self.create_content_object(
            content_data=music_data,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )
    
    def _generate_ambience(self, 
                          latent_vector: np.ndarray, 
                          params: ContentParameters) -> Dict[str, Any]:
        """Generate ambient soundscape using hybrid approach"""
        # Placeholder implementation
        ambience_data = f"base64_encoded_ambience_data_for_{params.dominant_emotion}_emotion"
        
        # Ambience parameters
        duration = params.get_parameter("duration", 120.0)  # seconds
        
        # Metadata
        metadata = {
            "duration": duration,
            "format": "mp3",
            "generation_method": "hybrid_ambience_generation"
        }
        
        # Create content object
        return self.create_content_object(
            content_data=ambience_data,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )
    
    def _generate_sound_effect(self, 
                              valence: float, 
                              arousal: float, 
                              intensity: float,
                              params: ContentParameters) -> Dict[str, Any]:
        """Generate sound effect based on emotion parameters"""
        # Placeholder implementation
        sound_effect_data = f"base64_encoded_sound_effect_for_{params.dominant_emotion}_emotion"
        
        # Sound effect parameters
        duration = 2.0 + (intensity * 3.0)  # seconds
        
        # Metadata
        metadata = {
            "duration": duration,
            "format": "wav",
            "generation_method": "hybrid_sound_effect_generation"
        }
        
        # Create content object
        return self.create_content_object(
            content_data=sound_effect_data,
            content_format=params.content_format,
            params=params,
            metadata=metadata
        )

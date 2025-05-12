"""
Social Sharing Module
This module provides functionality for sharing emotion-amplified content 
with other users and facilitating emotional connections between users.
"""
import logging
import time
import uuid
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)

class SocialSharingPlatform:
    """
    Manages social sharing capabilities for emotion-amplified content.
    Enables users to share their content, connect based on emotional synchronicity,
    and interact in emotion-synchronized environments.
    """
    def __init__(self, user_id: str, platform_url: str = "https://emotion-amplifier.social"):
        """
        Initialize the social sharing platform.
        
        Args:
            user_id: Unique identifier for the user
            platform_url: Base URL for the social platform
        """
        self.user_id = user_id
        self.platform_url = platform_url
        # In-memory storage for shared content (would be a database in production)
        self.shared_content = {}
        # User connections (emotion-based)
        self.user_connections = {}
        # Emotion synchronization rooms
        self.emotion_rooms = {}
        # User profiles
        self.user_profiles = {}
        
        # Create default profile for current user
        self._init_user_profile()
        
        logger.info(f"Social sharing platform initialized for user {user_id}")
    
    def _init_user_profile(self):
        """Initialize the user profile with default values"""
        self.user_profiles[self.user_id] = {
            "id": self.user_id,
            "created_at": time.time(),
            "emotion_profile": {},
            "sharing_preferences": {
                "public_sharing": False,
                "allow_anonymous": False,
                "emotion_matching_threshold": 0.7
            },
            "shared_content_count": 0,
            "emotion_synchronizations": 0
        }
    
    def update_emotion_profile(self, emotion_data: Dict[str, float]):
        """
        Update the user's emotion profile with new emotion data.
        
        Args:
            emotion_data: Dictionary mapping emotion types to intensity values
        """
        if self.user_id not in self.user_profiles:
            self._init_user_profile()
            
        profile = self.user_profiles[self.user_id]
        
        # Initialize emotion profile if not exists
        if "emotion_profile" not in profile:
            profile["emotion_profile"] = {}
            
        # Update with new emotion data
        for emotion, value in emotion_data.items():
            if emotion not in profile["emotion_profile"]:
                profile["emotion_profile"][emotion] = []
                
            # Add new data point with timestamp
            profile["emotion_profile"][emotion].append({
                "value": value,
                "timestamp": time.time()
            })
            
            # Keep only the last 100 data points
            profile["emotion_profile"][emotion] = \
                profile["emotion_profile"][emotion][-100:]
        
        logger.info(f"Updated emotion profile for user {self.user_id}")
    
    def share_content(self, content_data: Dict[str, Any], 
                     emotion_context: Dict[str, float],
                     privacy_level: str = "friends",
                     duration: int = 86400) -> str:
        """
        Share emotion-amplified content with others.
        
        Args:
            content_data: Dictionary containing the content to be shared
            emotion_context: Emotion data associated with the content
            privacy_level: Privacy level for sharing ("public", "friends", "private")
            duration: Duration in seconds for which content is shared (default: 24 hours)
            
        Returns:
            share_id: Unique identifier for the shared content
        """
        # Generate a unique share ID
        share_id = str(uuid.uuid4())
        
        # Create share object
        share_object = {
            "id": share_id,
            "user_id": self.user_id,
            "content": content_data,
            "emotion_context": emotion_context,
            "privacy_level": privacy_level,
            "created_at": time.time(),
            "expires_at": time.time() + duration,
            "interactions": {
                "views": 0,
                "emotional_resonance": [],
                "comments": []
            }
        }
        
        # Store the shared content
        self.shared_content[share_id] = share_object
        
        # Update user profile
        if self.user_id in self.user_profiles:
            self.user_profiles[self.user_id]["shared_content_count"] += 1
        
        logger.info(f"Content shared with share ID: {share_id}")
        return share_id
    
    def find_emotion_matches(self, emotion_data: Dict[str, float], 
                           threshold: float = 0.7,
                           max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Find users with matching emotion patterns.
        
        Args:
            emotion_data: Current emotion data to match
            threshold: Similarity threshold for matching (0.0 to 1.0)
            max_results: Maximum number of matches to return
            
        Returns:
            List of matching user profiles
        """
        matches = []
        
        # Calculate emotional distance to other users
        for user_id, profile in self.user_profiles.items():
            # Skip self
            if user_id == self.user_id:
                continue
                
            # Skip users with no emotion profile
            if "emotion_profile" not in profile or not profile["emotion_profile"]:
                continue
                
            # Calculate emotional similarity
            similarity = self._calculate_emotion_similarity(emotion_data, profile["emotion_profile"])
            
            # Add match if similarity exceeds threshold
            if similarity >= threshold:
                matches.append({
                    "user_id": user_id,
                    "similarity": similarity,
                    "shared_content_count": profile.get("shared_content_count", 0)
                })
        
        # Sort by similarity (highest first)
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Return top matches
        return matches[:max_results]
    
    def _calculate_emotion_similarity(self, current_emotions: Dict[str, float], 
                                     profile_emotions: Dict[str, List[Dict[str, Any]]]) -> float:
        """
        Calculate the similarity between current emotions and a user's emotion profile.
        
        Args:
            current_emotions: Current emotion data
            profile_emotions: Historical emotion profile data
            
        Returns:
            similarity: Emotional similarity score (0.0 to 1.0)
        """
        # Extract most recent emotion values from profile
        profile_current = {}
        for emotion, history in profile_emotions.items():
            if history:  # If there's data for this emotion
                profile_current[emotion] = history[-1]["value"]
        
        # Get intersection of emotion types
        common_emotions = set(current_emotions.keys()) & set(profile_current.keys())
        
        # If no common emotions, return 0
        if not common_emotions:
            return 0.0
        
        # Calculate similarity for common emotions
        similarity_sum = 0.0
        for emotion in common_emotions:
            # Calculate absolute difference
            diff = abs(current_emotions[emotion] - profile_current[emotion])
            # Convert to similarity (1.0 - diff)
            sim = max(0.0, 1.0 - diff)
            similarity_sum += sim
        
        # Calculate average similarity
        return similarity_sum / len(common_emotions)
    
    def create_emotion_room(self, name: str, 
                          emotion_theme: Dict[str, float],
                          description: str = "",
                          max_participants: int = 10) -> str:
        """
        Create a virtual room for emotional synchronization.
        
        Args:
            name: Name of the emotion room
            emotion_theme: Target emotions for the room
            description: Room description
            max_participants: Maximum number of participants
            
        Returns:
            room_id: Unique identifier for the room
        """
        # Generate room ID
        room_id = str(uuid.uuid4())
        
        # Create room object
        room = {
            "id": room_id,
            "name": name,
            "creator_id": self.user_id,
            "emotion_theme": emotion_theme,
            "description": description,
            "created_at": time.time(),
            "max_participants": max_participants,
            "participants": [self.user_id],
            "messages": [],
            "shared_content": [],
            "emotional_synchronicity": 0.0
        }
        
        # Store the room
        self.emotion_rooms[room_id] = room
        
        logger.info(f"Emotion room created: {name} (ID: {room_id})")
        return room_id
    
    def join_emotion_room(self, room_id: str, user_emotion: Dict[str, float]) -> bool:
        """
        Join an existing emotion room.
        
        Args:
            room_id: ID of the room to join
            user_emotion: Current emotion data of the joining user
            
        Returns:
            bool: True if successfully joined, False otherwise
        """
        # Check if room exists
        if room_id not in self.emotion_rooms:
            logger.warning(f"Room {room_id} not found")
            return False
        
        room = self.emotion_rooms[room_id]
        
        # Check if room is full
        if len(room["participants"]) >= room["max_participants"]:
            logger.warning(f"Room {room_id} is full")
            return False
        
        # Check if user is already in the room
        if self.user_id in room["participants"]:
            logger.warning(f"User {self.user_id} is already in room {room_id}")
            return False
        
        # Add user to room
        room["participants"].append(self.user_id)
        
        # Update emotional synchronicity
        self._update_room_synchronicity(room_id, user_emotion)
        
        logger.info(f"User {self.user_id} joined room {room_id}")
        return True
    
    def leave_emotion_room(self, room_id: str) -> bool:
        """
        Leave an emotion room.
        
        Args:
            room_id: ID of the room to leave
            
        Returns:
            bool: True if successfully left, False otherwise
        """
        # Check if room exists
        if room_id not in self.emotion_rooms:
            logger.warning(f"Room {room_id} not found")
            return False
        
        room = self.emotion_rooms[room_id]
        
        # Check if user is in the room
        if self.user_id not in room["participants"]:
            logger.warning(f"User {self.user_id} is not in room {room_id}")
            return False
        
        # Remove user from room
        room["participants"].remove(self.user_id)
        
        # If room is now empty and user was not the creator, delete the room
        if not room["participants"] and room["creator_id"] != self.user_id:
            del self.emotion_rooms[room_id]
            logger.info(f"Empty room {room_id} deleted")
        
        logger.info(f"User {self.user_id} left room {room_id}")
        return True
    
    def share_in_room(self, room_id: str, 
                     content_data: Dict[str, Any],
                     emotion_context: Dict[str, float]) -> str:
        """
        Share content in an emotion room.
        
        Args:
            room_id: ID of the room to share in
            content_data: Content to be shared
            emotion_context: Emotion data associated with the content
            
        Returns:
            share_id: ID of the shared content
        """
        # Check if room exists
        if room_id not in self.emotion_rooms:
            raise ValueError(f"Room {room_id} not found")
        
        room = self.emotion_rooms[room_id]
        
        # Check if user is in the room
        if self.user_id not in room["participants"]:
            raise ValueError(f"User {self.user_id} is not in room {room_id}")
        
        # Share content (create public share)
        share_id = self.share_content(
            content_data=content_data,
            emotion_context=emotion_context,
            privacy_level="public"
        )
        
        # Add to room's shared content
        room["shared_content"].append({
            "share_id": share_id,
            "user_id": self.user_id,
            "timestamp": time.time()
        })
        
        # Update room synchronicity
        self._update_room_synchronicity(room_id, emotion_context)
        
        logger.info(f"Content shared in room {room_id} with share ID: {share_id}")
        return share_id
    
    def _update_room_synchronicity(self, room_id: str, new_emotion: Dict[str, float]):
        """
        Update the emotional synchronicity of a room.
        
        Args:
            room_id: ID of the room
            new_emotion: New emotion data to consider
        """
        if room_id not in self.emotion_rooms:
            return
        
        room = self.emotion_rooms[room_id]
        
        # Get room's emotion theme
        theme = room["emotion_theme"]
        
        # Calculate similarity between new emotion and theme
        similarity = self._calculate_emotion_similarity(new_emotion, 
                                                      {k: [{"value": v}] for k, v in theme.items()})
        
        # Update room's emotional synchronicity (rolling average)
        current_sync = room["emotional_synchronicity"]
        room["emotional_synchronicity"] = (current_sync * 0.8) + (similarity * 0.2)
        
        logger.debug(f"Room {room_id} synchronicity updated to {room['emotional_synchronicity']}")
    
    def get_recommended_content(self, current_emotion: Dict[str, float], 
                              max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Get content recommendations based on current emotional state.
        
        Args:
            current_emotion: Current emotion data
            max_results: Maximum number of recommendations
            
        Returns:
            List of recommended content items
        """
        recommendations = []
        
        # Get all shared content
        for share_id, share in self.shared_content.items():
            # Skip expired content
            if share["expires_at"] < time.time():
                continue
                
            # Skip private content not from friends
            if share["privacy_level"] == "private" and share["user_id"] != self.user_id:
                continue
            
            # Calculate emotional resonance
            resonance = self._calculate_emotion_similarity(
                current_emotion, 
                {"emotion": [{"value": v} for v in share["emotion_context"].values()]}
            )
            
            # Add to recommendations
            recommendations.append({
                "share_id": share_id,
                "content_type": share["content"].get("type", "unknown"),
                "user_id": share["user_id"],
                "resonance": resonance,
                "created_at": share["created_at"]
            })
        
        # Sort by resonance (highest first)
        recommendations.sort(key=lambda x: x["resonance"], reverse=True)
        
        # Return top recommendations
        return recommendations[:max_results]

# Example usage
if __name__ == "__main__":
    # Initialize social platform
    social = SocialSharingPlatform(user_id="user123")
    
    # Update emotion profile
    emotion_data = {
        "joy": 0.8,
        "sadness": 0.1,
        "anger": 0.0,
        "fear": 0.1
    }
    social.update_emotion_profile(emotion_data)
    
    # Share content
    content = {
        "type": "image",
        "format": "jpg",
        "size": "512x512",
        "data": "base64_encoded_image_data_here"
    }
    share_id = social.share_content(content, emotion_data)
    print(f"Content shared with ID: {share_id}")
    
    # Create emotion room
    room_id = social.create_emotion_room(
        name="Joyful Space",
        emotion_theme={"joy": 0.8, "excitement": 0.7},
        description="A space for sharing joyful moments"
    )
    print(f"Created emotion room with ID: {room_id}")
    
    # Get recommendations
    recommendations = social.get_recommended_content(emotion_data)
    print(f"Got {len(recommendations)} content recommendations")

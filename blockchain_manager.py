"""
Blockchain Manager Module
This module provides blockchain-based security and ownership for emotional data and generated content, ensuring privacy and control for users.
"""
import json
import logging
import time
from typing import Dict, List, Optional, Any, Union
import hashlib
import uuid

logger = logging.getLogger(__name__)

class BlockchainManager:
    """
    Manages blockchain integration for emotional data security.
    Provides functionality for securing and verifying ownership of emotional data and generated content using blockchain technology.
    """
    def __init__(self, user_id: str, chain_type: str = 'hyperledger', chain_location: str = 'local'):
        """
        Initialize the blockchain manager.
        Args:
            user_id: Unique identifier for the user
            chain_type: Type of blockchain to use (e.g., 'hyperledger', 'ethereum')
            chain_location: Location of the blockchain ('local', 'remote')
        """
        self.user_id = user_id
        self.chain_type = chain_type
        self.chain_location = chain_location
        # Connection to blockchain (would be a real connection in production)
        self.blockchain_connection = None
        # Record of registered content
        self.registered_content = {}
        # Initialize blockchain connection
        self._init_blockchain()
        logger.info(f"Blockchain manager initialized with {chain_type} on {chain_location}")
    
    def _init_blockchain(self):
        """Initialize connection to the blockchain"""
        # In a real implementation, this would connect to the actual blockchain
        # For now, we'll simulate the blockchain with local data
        if self.chain_location == 'local':
            self.blockchain_connection = {}
            logger.info("Connected to simulated local blockchain")
        else:
            # In production, connect to remote blockchain
            logger.info("Connected to remote blockchain (simulation)")
    
    def register_emotion_data(self, emotion_data: Dict[str, Any]) -> str:
        """
        Register emotion data on the blockchain to ensure ownership and integrity.
        
        Args:
            emotion_data: Dictionary containing emotion data to be registered
            
        Returns:
            transaction_id: Unique identifier for the blockchain transaction
        """
        # Create a unique hash for the data
        data_hash = self._create_hash(emotion_data)
        
        # Create a blockchain transaction (simulated)
        transaction = {
            'user_id': self.user_id,
            'data_hash': data_hash,
            'timestamp': time.time(),
            'data_type': 'emotion_data',
            'transaction_type': 'register'
        }
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4())
        
        # Store in our simulated blockchain
        if self.blockchain_connection is not None:
            self.blockchain_connection[transaction_id] = transaction
        
        # Keep record locally
        self.registered_content[transaction_id] = {
            'data_hash': data_hash,
            'timestamp': transaction['timestamp'],
            'type': 'emotion_data'
        }
        
        logger.info(f"Emotion data registered with transaction ID: {transaction_id}")
        return transaction_id
    
    def register_content(self, content_data: Dict[str, Any], content_type: str) -> str:
        """
        Register generated content on the blockchain to ensure ownership.
        
        Args:
            content_data: Dictionary containing content data to be registered
            content_type: Type of content (e.g., 'image', 'music', 'text')
            
        Returns:
            transaction_id: Unique identifier for the blockchain transaction
        """
        # Create a unique hash for the content
        content_hash = self._create_hash(content_data)
        
        # Create a blockchain transaction (simulated)
        transaction = {
            'user_id': self.user_id,
            'content_hash': content_hash,
            'timestamp': time.time(),
            'content_type': content_type,
            'transaction_type': 'register_content'
        }
        
        # Generate transaction ID
        transaction_id = str(uuid.uuid4())
        
        # Store in our simulated blockchain
        if self.blockchain_connection is not None:
            self.blockchain_connection[transaction_id] = transaction
        
        # Keep record locally
        self.registered_content[transaction_id] = {
            'content_hash': content_hash,
            'timestamp': transaction['timestamp'],
            'type': content_type
        }
        
        logger.info(f"{content_type} content registered with transaction ID: {transaction_id}")
        return transaction_id
    
    def verify_ownership(self, transaction_id: str) -> bool:
        """
        Verify ownership of registered data or content.
        
        Args:
            transaction_id: ID of the transaction to verify
            
        Returns:
            bool: True if ownership is verified, False otherwise
        """
        # Check if transaction exists in our simulated blockchain
        if (self.blockchain_connection is not None and 
            transaction_id in self.blockchain_connection):
            transaction = self.blockchain_connection[transaction_id]
            return transaction['user_id'] == self.user_id
        
        return False
    
    def _create_hash(self, data: Any) -> str:
        """
        Create a hash of the provided data.
        
        Args:
            data: Data to be hashed
            
        Returns:
            str: Hash of the data
        """
        # Convert data to JSON string
        data_str = json.dumps(data, sort_keys=True)
        # Create SHA-256 hash
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def get_transaction_history(self) -> List[Dict[str, Any]]:
        """
        Get history of all transactions for the current user.
        
        Returns:
            List of transaction dictionaries
        """
        if self.blockchain_connection is None:
            return []
        
        # Filter transactions for current user
        history = [
            {
                'transaction_id': tx_id,
                'timestamp': tx_data['timestamp'],
                'type': tx_data.get('data_type', tx_data.get('content_type', 'unknown'))
            }
            for tx_id, tx_data in self.blockchain_connection.items()
            if tx_data['user_id'] == self.user_id
        ]
        
        # Sort by timestamp (newest first)
        history.sort(key=lambda x: x['timestamp'], reverse=True)
        return history
    
    def share_content_ownership(self, transaction_id: str, recipient_id: str) -> str:
        """
        Share ownership of content with another user.
        
        Args:
            transaction_id: ID of the content transaction
            recipient_id: ID of the user to share with
            
        Returns:
            new_transaction_id: ID of the sharing transaction
        """
        # Verify ownership
        if not self.verify_ownership(transaction_id):
            raise ValueError("Cannot share content: ownership verification failed")
        
        # Create a sharing transaction
        original_tx = self.blockchain_connection[transaction_id]
        sharing_tx = {
            'user_id': self.user_id,
            'recipient_id': recipient_id,
            'original_transaction_id': transaction_id,
            'content_hash': original_tx.get('content_hash', original_tx.get('data_hash')),
            'timestamp': time.time(),
            'transaction_type': 'share'
        }
        
        # Generate new transaction ID
        new_transaction_id = str(uuid.uuid4())
        
        # Store in our simulated blockchain
        if self.blockchain_connection is not None:
            self.blockchain_connection[new_transaction_id] = sharing_tx
        
        logger.info(f"Content shared with user {recipient_id}, transaction ID: {new_transaction_id}")
        return new_transaction_id

# Example usage
if __name__ == "__main__":
    # Initialize blockchain manager
    blockchain = BlockchainManager(user_id="user123")
    
    # Register emotion data
    emotion_data = {
        "joy": 0.8,
        "sadness": 0.1,
        "anger": 0.0,
        "fear": 0.1,
        "timestamp": time.time()
    }
    tx_id = blockchain.register_emotion_data(emotion_data)
    print(f"Registered emotion data with transaction ID: {tx_id}")
    
    # Register generated content
    content_data = {
        "type": "image",
        "format": "jpg",
        "size": "512x512",
        "data": "base64_encoded_image_data_here",
        "timestamp": time.time()
    }
    content_tx_id = blockchain.register_content(content_data, "image")
    print(f"Registered content with transaction ID: {content_tx_id}")
    
    # Verify ownership
    is_owner = blockchain.verify_ownership(tx_id)
    print(f"Ownership verification: {is_owner}")
    
    # Get transaction history
    history = blockchain.get_transaction_history()
    print(f"Transaction history: {history}")

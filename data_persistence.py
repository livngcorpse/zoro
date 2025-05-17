import os
import json
import pickle
import logging
from logger import setup_logger

# Set up logging
logger = setup_logger('data_persistence')

class DataPersistence:
    """Handles saving and loading game data and RL model state."""
    
    def __init__(self, data_dir='data'):
        """Initialize data persistence manager."""
        self.data_dir = data_dir
        self.model_file = os.path.join(data_dir, 'rl_model.pkl')
        self.game_history_file = os.path.join(data_dir, 'game_history.json')
        self.config_file = os.path.join(data_dir, 'user_config.json')
        self.data_storage_permission = False
        
        # Create data directory if it doesn't exist
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            logger.info(f"Created data directory: {data_dir}")
    
    def save_permission_status(self, permission_granted):
        """Save user's permission status for data storage."""
        self.data_storage_permission = permission_granted
        config = {'data_storage_permission': permission_granted}
        
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f)
            logger.info(f"Saved permission status: {permission_granted}")
            return True
        except Exception as e:
            logger.error(f"Failed to save permission status: {str(e)}")
            return False
    
    def load_permission_status(self):
        """Load user's permission status for data storage."""
        if not os.path.exists(self.config_file):
            logger.info("No permission configuration found")
            return False
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            self.data_storage_permission = config.get('data_storage_permission', False)
            logger.info(f"Loaded permission status: {self.data_storage_permission}")
            return self.data_storage_permission
        except Exception as e:
            logger.error(f"Failed to load permission status: {str(e)}")
            return False
    
    def save_model_data(self, q_table):
        """Save the RL model's Q-table to disk."""
        if not self.data_storage_permission:
            logger.warning("Data storage permission not granted, skipping model save")
            return False
        
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(q_table, f)
            logger.info(f"Saved model data to {self.model_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model data: {str(e)}")
            return False
    
    def load_model_data(self):
        """Load the RL model's Q-table from disk."""
        if not self.data_storage_permission:
            logger.warning("Data storage permission not granted, skipping model load")
            return {}
        
        if not os.path.exists(self.model_file):
            logger.info("No model file found, starting with empty Q-table")
            return {}
        
        try:
            with open(self.model_file, 'rb') as f:
                q_table = pickle.load(f)
            logger.info(f"Loaded model data from {self.model_file}")
            return q_table
        except Exception as e:
            logger.error(f"Failed to load model data: {str(e)}")
            return {}
    
    def save_game_history(self, game_history):
        """Save game history data to disk."""
        if not self.data_storage_permission:
            logger.warning("Data storage permission not granted, skipping history save")
            return False
        
        try:
            # Convert to a serializable format
            serializable_history = {
                'games_played': game_history.get('games_played', 0),
                'wins': game_history.get('wins', 0),
                'losses': game_history.get('losses', 0),
                'win_rate': game_history.get('win_rate', 0.0),
                'average_diamonds': game_history.get('average_diamonds', 0.0),
                'timestamp': game_history.get('timestamp', '')
            }
            
            with open(self.game_history_file, 'w') as f:
                json.dump(serializable_history, f)
            logger.info(f"Saved game history to {self.game_history_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save game history: {str(e)}")
            return False
    
    def load_game_history(self):
        """Load game history data from disk."""
        if not self.data_storage_permission:
            logger.warning("Data storage permission not granted, skipping history load")
            return {}
        
        if not os.path.exists(self.game_history_file):
            logger.info("No game history file found")
            return {}
        
        try:
            with open(self.game_history_file, 'r') as f:
                game_history = json.load(f)
            logger.info(f"Loaded game history from {self.game_history_file}")
            return game_history
        except Exception as e:
            logger.error(f"Failed to load game history: {str(e)}")
            return {}
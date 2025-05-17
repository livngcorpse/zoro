from typing import Dict, List, Tuple, Set
import random
import logging
import json
import time
from datetime import datetime
from config import DEFAULT_BET, DEFAULT_BOMBS, GRID_SIZE, TRAINING_CASHOUT
from data_persistence import DataPersistence
from logger import setup_logger

# Set up logging
logger = setup_logger('state_manager')

class StateManager:
    """Manages the state of the game and AI."""
    
    def __init__(self):
        """Initialize the state manager with default values."""
        # Game configuration
        self.bet_amount = DEFAULT_BET
        self.bombs = DEFAULT_BOMBS
        self.grid_size = GRID_SIZE
        
        # Game state
        self.is_running = False
        self.is_game_active = False
        self.waiting_for_resume = False
        self.training_mode = True  # Start in training mode by default
        self.use_rl = False
        self.manual_intervention = False  # Flag for manual intervention
        
        # Game statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        self.total_diamonds_found = 0
        self.session_start_time = datetime.now()
        
        # Current game state
        self.current_grid = self._initialize_empty_grid()
        self.revealed_positions = set()
        self.revealed_diamonds = 0
        self.current_multiplier = 1.0
        self.last_action = None
        self.game_history = {}
        
        # RL learning data
        self.q_table = {}  # State -> action -> value mapping
        self.bomb_history = {}  # Maps game IDs to bomb positions
        self.diamond_history = {}  # Maps game IDs to diamond positions
        
        # Data persistence
        self.data_persistence = DataPersistence()
        self.data_storage_permission = self.data_persistence.load_permission_status()
        
        # Load saved data if permission is granted
        if self.data_storage_permission:
            self._load_saved_data()
    
    def _initialize_empty_grid(self) -> List[List[str]]:
        """Initialize an empty grid with the specified size."""
        try:
            return [['unknown' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        except Exception as e:
            logger.error(f"Error initializing grid: {str(e)}")
            # Fallback to a default 5x5 grid
            return [['unknown' for _ in range(5)] for _ in range(5)]
    
    def reset_game_state(self):
        """Reset the current game state."""
        try:
            logger.info("Resetting game state")
            self.is_game_active = False
            self.current_grid = self._initialize_empty_grid()
            self.revealed_positions = set()
            self.revealed_diamonds = 0
            self.current_multiplier = 1.0
            self.last_action = None
        except Exception as e:
            logger.error(f"Error resetting game state: {str(e)}")
            # Force a complete reset in case of error
            self.__init__()
    
    def record_game_outcome(self, won: bool, diamonds_revealed: int, bombs_hit: int = 0):
        """Record the outcome of a game."""
        try:
            self.games_played += 1
            self.total_diamonds_found += diamonds_revealed
            
            if won:
                self.wins += 1
                logger.info(f"Game won! Total wins: {self.wins}, Total games played: {self.games_played}")
            else:
                self.losses += 1
                logger.info(f"Game lost. Total losses: {self.losses}, Total games played: {self.games_played}")
            
            # Update game history
            self._update_game_history(won, diamonds_revealed)
            
            # Save data periodically (every 5 games)
            if self.games_played % 5 == 0 and self.data_storage_permission:
                self._save_current_data()
                
        except Exception as e:
            logger.error(f"Error recording game outcome: {str(e)}")
    
    def _update_game_history(self, won: bool, diamonds_revealed: int):
        """Update the game history statistics."""
        try:
            win_rate = self.wins / self.games_played if self.games_played > 0 else 0
            avg_diamonds = self.total_diamonds_found / self.games_played if self.games_played > 0 else 0
            
            self.game_history = {
                'games_played': self.games_played,
                'wins': self.wins,
                'losses': self.losses,
                'win_rate': win_rate,
                'average_diamonds': avg_diamonds,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error updating game history: {str(e)}")
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid (unrevealed) positions on the grid."""
        try:
            valid_moves = []
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    if (i, j) not in self.revealed_positions:
                        valid_moves.append((i, j))
            
            logger.debug(f"Valid moves remaining: {len(valid_moves)}")
            return valid_moves
        except Exception as e:
            logger.error(f"Error getting valid moves: {str(e)}")
            # Return some default positions in case of error
            return [(i, j) for i in range(self.grid_size) for j in range(self.grid_size) 
                    if (i, j) not in self.revealed_positions]
    
    def should_cash_out_training(self) -> bool:
        """Determine if the AI should cash out in training mode."""
        try:
            should_cash = self.revealed_diamonds >= TRAINING_CASHOUT
            if should_cash:
                logger.info(f"Training mode cash out condition met: {self.revealed_diamonds} diamonds revealed, threshold is {TRAINING_CASHOUT}")
            return should_cash
        except Exception as e:
            logger.error(f"Error in cash out decision: {str(e)}")
            # Default to cashing out after 3 diamonds in case of error
            return self.revealed_diamonds >= 3
    
    def get_state_hash(self) -> str:
        """Create a hash representing the current state for the Q-table."""
        try:
            # Create a more informative state hash that includes revealed positions
            revealed_str = ','.join([f"{pos[0]},{pos[1]}" for pos in sorted(self.revealed_positions)])
            return f"{revealed_str}_{self.bombs}_{self.revealed_diamonds}"
        except Exception as e:
            logger.error(f"Error creating state hash: {str(e)}")
            # Fallback to a simple state representation
            return f"state_{self.bombs}_{self.revealed_diamonds}"
    
    def update_grid(self, row: int, col: int, result: str):
        """Update the grid with the result of clicking a position."""
        try:
            # Validate indices
            if not (0 <= row < self.grid_size and 0 <= col < self.grid_size):
                logger.warning(f"Invalid grid position: ({row}, {col})")
                return
                
            self.current_grid[row][col] = result
            self.revealed_positions.add((row, col))
            
            # Update diamond count if a diamond was found
            if result == 'diamond':
                self.revealed_diamonds += 1
                logger.info(f"Diamond found! Total diamonds: {self.revealed_diamonds}")
        except Exception as e:
            logger.error(f"Error updating grid: {str(e)}")
    
    def update_diamond_count(self, new_count=None):
        """Update the diamond count by incrementing or setting a specific value."""
        try:
            if new_count is not None:
                # Set to specific value
                self.revealed_diamonds = new_count
            else:
                # Increment by 1
                self.revealed_diamonds += 1
            
            logger.info(f"Updated diamond count: {self.revealed_diamonds}")
        except Exception as e:
            logger.error(f"Error updating diamond count: {str(e)}")
    
    def set_permission_status(self, permission_granted):
        """Set the data storage permission status."""
        try:
            self.data_storage_permission = permission_granted
            result = self.data_persistence.save_permission_status(permission_granted)
            
            if permission_granted and result:
                # If permission was just granted, save current data
                self._save_current_data()
                
            return result
        except Exception as e:
            logger.error(f"Error setting permission status: {str(e)}")
            return False
    
    def _save_current_data(self):
        """Save current game data and model to disk."""
        try:
            if not self.data_storage_permission:
                logger.debug("Data storage permission not granted, skipping save")
                return False
                
            # Save Q-table
            self.data_persistence.save_model_data(self.q_table)
            
            # Save game history
            self.data_persistence.save_game_history(self.game_history)
            
            logger.info("Game data and model saved successfully")
            return True
        except Exception as e:
            logger.error(f"Error saving current data: {str(e)}")
            return False
    
    def _load_saved_data(self):
        """Load saved game data and model from disk."""
        try:
            if not self.data_storage_permission:
                logger.debug("Data storage permission not granted, skipping load")
                return False
                
            # Load Q-table
            q_table = self.data_persistence.load_model_data()
            if q_table:
                self.q_table = q_table
            
            # Load game history
            game_history = self.data_persistence.load_game_history()
            if game_history:
                self.games_played = game_history.get('games_played', 0)
                self.wins = game_history.get('wins', 0)
                self.losses = game_history.get('losses', 0)
                
            logger.info("Game data and model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading saved data: {str(e)}")
            return False
    
    def choose_action_training(self) -> Tuple[int, int]:
        """Choose a random valid move for training."""
        try:
            valid_moves = self.get_valid_moves()
            if not valid_moves:
                logger.warning("No valid moves left for training")
                return None  # No valid moves left
            
            chosen_move = random.choice(valid_moves)
            logger.debug(f"Training mode: Chose random position {chosen_move}")
            self.last_action = chosen_move
            return chosen_move
        except Exception as e:
            logger.error(f"Error choosing training action: {str(e)}")
            # Return a safe fallback if possible
            valid_moves = self.get_valid_moves()
            if valid_moves:
                return random.choice(valid_moves)
            return None
    
    def choose_action_rl(self) -> Tuple[int, int]:
        """Choose the best action based on the trained Q-values."""
        try:
            state = self.get_state_hash()
            valid_moves = self.get_valid_moves()
            
            if not valid_moves:
                logger.warning("No valid moves left for RL")
                return None  # No valid moves left
            
            # If the state is unknown or exploration is triggered, choose randomly
            if state not in self.q_table or not self.q_table[state]:
                chosen_move = random.choice(valid_moves)
                logger.debug(f"RL mode: No Q-values, choosing random position {chosen_move}")
                self.last_action = chosen_move
                return chosen_move
            
            # Find the best action based on Q-values
            best_value = -float('inf')
            best_actions = []
            
            for action in valid_moves:
                action_str = f"{action[0]},{action[1]}"
                if action_str in self.q_table[state]:
                    q_value = self.q_table[state][action_str]
                    if q_value > best_value:
                        best_value = q_value
                        best_actions = [action]
                    elif q_value == best_value:
                        best_actions.append(action)
            
            # If we found actions with positive Q-values, choose among them
            if best_actions and best_value > 0:
                chosen_move = random.choice(best_actions)
                logger.debug(f"RL mode: Chose best action {chosen_move} with Q-value {best_value}")
                self.last_action = chosen_move
                return chosen_move
            
            # If no good moves are known, choose randomly
            chosen_move = random.choice(valid_moves)
            logger.debug(f"RL mode: No positive Q-values, choosing random position {chosen_move}")
            self.last_action = chosen_move
            return chosen_move
        except Exception as e:
            logger.error(f"Error choosing RL action: {str(e)}")
            # Return a safe fallback if possible
            valid_moves = self.get_valid_moves()
            if valid_moves:
                chosen_move = random.choice(valid_moves)
                self.last_action = chosen_move
                return chosen_move
            return None
    
    def pause_for_manual_intervention(self):
        """Pause the game for manual intervention."""
        try:
            logger.info("Pausing for manual intervention")
            self.waiting_for_resume = True
            self.manual_intervention = True
        except Exception as e:
            logger.error(f"Error pausing for manual intervention: {str(e)}")
    
    def resume_from_manual_intervention(self):
        """Resume the game after manual intervention."""
        try:
            logger.info("Resuming from manual intervention")
            self.waiting_for_resume = False
            self.manual_intervention = False
        except Exception as e:
            logger.error(f"Error resuming from manual intervention: {str(e)}")
    
    def record_bomb_positions(self, game_id: str, positions: List[Tuple[int, int]]):
        """Record the positions of bombs for a game."""
        try:
            self.bomb_history[game_id] = positions
            logger.debug(f"Recorded bomb positions for game {game_id}: {positions}")
        except Exception as e:
            logger.error(f"Error recording bomb positions: {str(e)}")
    
    def record_diamond_positions(self, game_id: str, positions: List[Tuple[int, int]]):
        """Record the positions of diamonds for a game."""
        try:
            self.diamond_history[game_id] = positions
            logger.debug(f"Recorded diamond positions for game {game_id}: {positions}")
        except Exception as e:
            logger.error(f"Error recording diamond positions: {str(e)}")
    
    def get_stats_summary(self):
        """Get a summary of current game statistics."""
        try:
            win_rate = self.wins / self.games_played if self.games_played > 0 else 0
            win_rate_percent = round(win_rate * 100, 2)
            
            avg_diamonds = self.total_diamonds_found / self.games_played if self.games_played > 0 else 0
            avg_diamonds_rounded = round(avg_diamonds, 2)
            
            session_duration = datetime.now() - self.session_start_time
            hours, remainder = divmod(session_duration.seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            
            return {
                "games_played": self.games_played,
                "wins": self.wins,
                "losses": self.losses,
                "win_rate": f"{win_rate_percent}%",
                "avg_diamonds": avg_diamonds_rounded,
                "session_time": f"{hours}h {minutes}m {seconds}s",
                "data_storage": "Enabled" if self.data_storage_permission else "Disabled"
            }
        except Exception as e:
            logger.error(f"Error getting stats summary: {str(e)}")
            return {"error": "Failed to generate statistics"}
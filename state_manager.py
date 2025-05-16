from typing import Dict, List, Tuple, Set
import random
import logging
from config import DEFAULT_BET, DEFAULT_BOMBS, GRID_SIZE, TRAINING_CASHOUT
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
        
        # Game statistics
        self.games_played = 0
        self.wins = 0
        self.losses = 0
        
        # Current game state
        self.current_grid = self._initialize_empty_grid()
        self.revealed_positions = set()
        self.revealed_diamonds = 0
        self.current_multiplier = 1.0
        
        # RL learning data
        self.q_table = {}  # State -> action -> value mapping
        self.bomb_history = {}  # Maps game IDs to bomb positions
        self.diamond_history = {}  # Maps game IDs to diamond positions
        
    def _initialize_empty_grid(self) -> List[List[str]]:
        """Initialize an empty grid with the specified size."""
        return [['unknown' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
    
    def reset_game_state(self):
        """Reset the current game state."""
        logger.info("Resetting game state")
        self.is_game_active = False
        self.current_grid = self._initialize_empty_grid()
        self.revealed_positions = set()
        self.revealed_diamonds = 0
        self.current_multiplier = 1.0
    
    def record_game_outcome(self, won: bool, diamonds_revealed: int, bombs_hit: int = 0):
        """Record the outcome of a game."""
        self.games_played += 1
        if won:
            self.wins += 1
            logger.info(f"Game won! Total wins: {self.wins}, Total games played: {self.games_played}")
        else:
            self.losses += 1
            logger.info(f"Game lost. Total losses: {self.losses}, Total games played: {self.games_played}")
    
    def get_valid_moves(self) -> List[Tuple[int, int]]:
        """Get all valid (unrevealed) positions on the grid."""
        valid_moves = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if (i, j) not in self.revealed_positions:
                    valid_moves.append((i, j))
        
        logger.debug(f"Valid moves remaining: {len(valid_moves)}")
        return valid_moves
    
    def should_cash_out_training(self) -> bool:
        """Determine if the AI should cash out in training mode."""
        should_cash = self.revealed_diamonds >= TRAINING_CASHOUT
        if should_cash:
            logger.info(f"Training mode cash out condition met: {self.revealed_diamonds} diamonds revealed, threshold is {TRAINING_CASHOUT}")
        return should_cash
    
    def get_state_hash(self) -> str:
        """Create a hash representing the current state for the Q-table."""
        # Flatten the grid and join as a string
        grid_str = ''.join([''.join(row) for row in self.current_grid])
        return f"{grid_str}_{self.bombs}_{self.revealed_diamonds}"
    
    def update_grid(self, row: int, col: int, result: str):
        """Update the grid with the result of clicking a position."""
        self.current_grid[row][col] = result
        self.revealed_positions.add((row, col))
        
        if result == 'diamond':
            self.revealed_diamonds += 1
            logger.info(f"Diamond found! Total diamonds: {self.revealed_diamonds}")
    
    def update_q_value(self, state: str, action: Tuple[int, int], new_value: float):
        """Update the Q-value for a state-action pair."""
        action_str = f"{action[0]},{action[1]}"
        if state not in self.q_table:
            self.q_table[state] = {}
        
        self.q_table[state][action_str] = new_value
    
    def get_q_value(self, state: str, action: Tuple[int, int]) -> float:
        """Get the Q-value for a state-action pair."""
        action_str = f"{action[0]},{action[1]}"
        if state not in self.q_table or action_str not in self.q_table[state]:
            return 0.0
        return self.q_table[state][action_str]
    
    def choose_action_training(self) -> Tuple[int, int]:
        """Choose a random valid move for training."""
        valid_moves = self.get_valid_moves()
        if not valid_moves:
            logger.warning("No valid moves left for training")
            return None  # No valid moves left
        
        chosen_move = random.choice(valid_moves)
        logger.debug(f"Training mode: Chose random position {chosen_move}")
        return chosen_move
    
    def choose_action_rl(self) -> Tuple[int, int]:
        """Choose the best action based on the trained Q-values."""
        state = self.get_state_hash()
        valid_moves = self.get_valid_moves()
        
        if not valid_moves:
            logger.warning("No valid moves left for RL")
            return None  # No valid moves left
        
        # If the state is unknown or exploration is triggered, choose randomly
        if state not in self.q_table or not self.q_table[state]:
            chosen_move = random.choice(valid_moves)
            logger.debug(f"RL mode: No Q-values, choosing random position {chosen_move}")
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
            return chosen_move
        
        # If no good moves are known, choose randomly
        chosen_move = random.choice(valid_moves)
        logger.debug(f"RL mode: No positive Q-values, choosing random position {chosen_move}")
        return chosen_move
    
    def record_bomb_positions(self, game_id: str, positions: List[Tuple[int, int]]):
        """Record the positions of bombs for a game."""
        self.bomb_history[game_id] = positions
        logger.debug(f"Recorded bomb positions for game {game_id}: {positions}")
    
    def record_diamond_positions(self, game_id: str, positions: List[Tuple[int, int]]):
        """Record the positions of diamonds for a game."""
        self.diamond_history[game_id] = positions
        logger.debug(f"Recorded diamond positions for game {game_id}: {positions}")
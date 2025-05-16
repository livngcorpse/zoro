from typing import Tuple, Dict, List
import random
import numpy as np
from config import LEARNING_RATE, DISCOUNT_FACTOR, GRID_SIZE
from state_manager import StateManager

class RLModel:
    """Reinforcement Learning model using Q-learning for the Mines game."""
    
    def __init__(self, state_manager: StateManager):
        """Initialize the RL model."""
        self.state_manager = state_manager
        self.learning_rate = LEARNING_RATE
        self.discount_factor = DISCOUNT_FACTOR
        
    def get_best_action(self, state: str) -> Tuple[int, int]:
        """Get the best action for a given state according to the Q-table."""
        valid_moves = self.state_manager.get_valid_moves()
        
        if not valid_moves:
            return None  # No valid moves left
        
        # If the state is unknown, choose randomly
        if state not in self.state_manager.q_table:
            return random.choice(valid_moves)
        
        # Find the best action based on Q-values
        best_value = -float('inf')
        best_actions = []
        
        for action in valid_moves:
            action_str = f"{action[0]},{action[1]}"
            if action_str in self.state_manager.q_table[state]:
                q_value = self.state_manager.q_table[state][action_str]
                if q_value > best_value:
                    best_value = q_value
                    best_actions = [action]
                elif q_value == best_value:
                    best_actions.append(action)
        
        # If we found actions with positive Q-values, choose among them
        if best_actions:
            return random.choice(best_actions)
        
        # If no good moves are known, choose randomly
        return random.choice(valid_moves)
    
    def update_q_values(self, old_state: str, action: Tuple[int, int], reward: float, new_state: str):
        """Update Q-values using the Q-learning update rule."""
        action_str = f"{action[0]},{action[1]}"
        
        # Initialize state in Q-table if not present
        if old_state not in self.state_manager.q_table:
            self.state_manager.q_table[old_state] = {}
        
        # Get current Q-value
        current_q = self.state_manager.q_table[old_state].get(action_str, 0.0)
        
        # Calculate maximum Q-value for next state
        max_next_q = 0.0
        if new_state in self.state_manager.q_table:
            valid_moves = self.state_manager.get_valid_moves()
            for next_action in valid_moves:
                next_action_str = f"{next_action[0]},{next_action[1]}"
                if next_action_str in self.state_manager.q_table[new_state]:
                    max_next_q = max(max_next_q, self.state_manager.q_table[new_state][next_action_str])
        
        # Q-learning update rule
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        
        # Update Q-table
        self.state_manager.q_table[old_state][action_str] = new_q
    
    def calculate_reward(self, old_diamonds: int, new_diamonds: int, hit_bomb: bool) -> float:
        """Calculate the reward for a state transition."""
        if hit_bomb:
            return -10.0  # Heavy penalty for hitting a bomb
        
        diamonds_revealed = new_diamonds - old_diamonds
        if diamonds_revealed > 0:
            return 1.0 * diamonds_revealed  # Reward for each diamond revealed
        
        return -0.1  # Small penalty for clicking a tile with no diamond
    
    def should_cash_out_rl(self, state: str) -> bool:
        """Decide whether to cash out based on the current state."""
        # Get the maximum Q-value for any action in this state
        max_q = -float('inf')
        if state in self.state_manager.q_table:
            for action_str, q_value in self.state_manager.q_table[state].items():
                max_q = max(max_q, q_value)
        
        # If the maximum Q-value is negative or very low, cash out
        if max_q < 0.2:
            return True
        
        # Cash out if we've revealed a good number of diamonds
        if self.state_manager.revealed_diamonds >= 5:
            return True
        
        # Probability of cashing out increases with each diamond
        cash_out_prob = self.state_manager.revealed_diamonds * 0.15
        return random.random() < cash_out_prob
    
    def learn_from_game(self, game_id: str, won: bool):
        """Learn from a completed game by updating Q-values."""
        if game_id not in self.state_manager.bomb_history and game_id not in self.state_manager.diamond_history:
            return  # No data to learn from
        
        # Get bomb and diamond positions
        bomb_positions = self.state_manager.bomb_history.get(game_id, [])
        diamond_positions = self.state_manager.diamond_history.get(game_id, [])
        
        # Update Q-values based on this knowledge
        # This is a simplified implementation - a full implementation would
        # reconstruct the game and update Q-values for all states and actions
        
        # Example: Decrease Q-values for actions that lead to bombs
        for bomb_pos in bomb_positions:
            state_key = f"unknown_{self.state_manager.bombs}_{0}"  # Initial state
            action_str = f"{bomb_pos[0]},{bomb_pos[1]}"
            
            if state_key not in self.state_manager.q_table:
                self.state_manager.q_table[state_key] = {}
            
            # Assign negative value to bomb positions
            self.state_manager.q_table[state_key][action_str] = -5.0
        
        # Example: Increase Q-values for actions that lead to diamonds
        for diamond_pos in diamond_positions:
            state_key = f"unknown_{self.state_manager.bombs}_{0}"  # Initial state
            action_str = f"{diamond_pos[0]},{diamond_pos[1]}"
            
            if state_key not in self.state_manager.q_table:
                self.state_manager.q_table[state_key] = {}
            
            # Assign positive value to diamond positions
            self.state_manager.q_table[state_key][action_str] = 2.0
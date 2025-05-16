from telethon import TelegramClient, events
from telethon.tl.custom import Button, Message
import asyncio
import logging
import re
import random
import time
import uuid
from typing import List, Dict, Optional, Tuple, Set

from config import (
    GAME_BOT_USERNAME, MAX_WAIT_TIME, GRID_SIZE,
    LEARNING_RATE, DISCOUNT_FACTOR, EXPLORATION_RATE, 
    EXPLORATION_DECAY, MIN_EXPLORATION_RATE, GROUP_ID
)
from state_manager import StateManager
from rl_model import RLModel
from logger import setup_logger

# Set up logging
logger = setup_logger('ai_game_handler')

class AIGameHandler:
    """Handles the AI gameplay logic."""
    
    def __init__(self, state_manager: StateManager):
        """Initialize the AI game handler."""
        self.state_manager = state_manager
        self.rl_model = RLModel(state_manager)
        self.current_game_id = None
        self.current_message = None
        self.exploration_rate = EXPLORATION_RATE
        self.last_response_time = time.time()
    
    async def process_game_message(self, event):
        """Process messages from the game bot."""
        self.last_response_time = time.time()
        message = event.message
        
        # Update the current message if it contains buttons (the game grid)
        if message.buttons:
            self.current_message = message
            self.state_manager.is_game_active = True
            
            # Check if this is a new game
            if not self.current_game_id:
                self.current_game_id = str(uuid.uuid4())
                logger.info(f"New game started with ID: {self.current_game_id}")
        
        # Process game state updates
        text = message.text if message.text else ""
        
        # Game finished with a bomb
        if "💥" in text and "Game over" in text:
            self.state_manager.record_game_outcome(False, self.state_manager.revealed_diamonds, 1)
            logger.info(f"Game {self.current_game_id} lost: Hit a bomb after revealing {self.state_manager.revealed_diamonds} diamonds")
            
            # Extract final state if possible
            await self._extract_final_state(text)
            
            # Reset for next game
            self.current_game_id = None
            self.state_manager.reset_game_state()
        
        # Game cashed out successfully
        elif "You won" in text and "multiplier" in text:
            win_amount = self._extract_win_amount(text)
            multiplier = self._extract_multiplier(text)
            
            self.state_manager.record_game_outcome(True, self.state_manager.revealed_diamonds)
            logger.info(f"Game {self.current_game_id} won: Cashed out with {self.state_manager.revealed_diamonds} diamonds. Multiplier: {multiplier}, Win: {win_amount}")
            
            # Reset for next game
            self.current_game_id = None
            self.state_manager.reset_game_state()
    
    async def game_loop(self, client):
        """Main game loop that automatically plays the game."""
        logger.info("Starting game loop")
        
        while self.state_manager.is_running:
            try:
                # Check if we're waiting for manual intervention
                if self.state_manager.waiting_for_resume:
                    await asyncio.sleep(1)
                    continue
                
                # Check if we need to start a new game
                if not self.state_manager.is_game_active:
                    await self._start_new_game(client)
                    await asyncio.sleep(2)  # Wait for game to start
                    continue
                
                # Check if the game bot is responding
                if time.time() - self.last_response_time > MAX_WAIT_TIME:
                    logger.warning("⚠️ Game bot is slow, manual intervention needed.")
                    await client.send_message(GROUP_ID, "⚠️ Game bot is slow, manual intervention needed. Waiting for /resume or /startai...")
                    self.state_manager.waiting_for_resume = True
                    continue
                
                # Play a move if we have an active game with a grid
                if self.state_manager.is_game_active and self.current_message and self.current_message.buttons:
                    # Choose action based on mode
                    if self.state_manager.training_mode:
                        await self._play_training_move(client)
                    else:
                        await self._play_rl_move(client)
                
                # Small delay between actions
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in game loop: {str(e)}")
                await asyncio.sleep(5)  # Wait a bit before retrying
    
    async def _start_new_game(self, client):
        """Start a new game using the current bet and bombs settings."""
        try:
            command = f"/mines {self.state_manager.bet_amount} {self.state_manager.bombs}"
            logger.info(f"Starting new game with command: {command}")
            await client.send_message(GAME_BOT_USERNAME, command)
            self.last_response_time = time.time()
        except Exception as e:
            logger.error(f"Error starting new game: {str(e)}")
    
    async def _play_training_move(self, client):
        """Play a move in training mode (random exploration)."""
        # Check if we should cash out based on diamonds revealed
        if self.state_manager.should_cash_out_training():
            await self._cash_out(client)
            return
        
        # Choose a random unrevealed position
        position = self.state_manager.choose_action_training()
        if position is None:
            await self._cash_out(client)  # No valid moves left, cash out
            return
        
        row, col = position
        await self._click_position(client, row, col)
    
    async def _play_rl_move(self, client):
        """Play a move using the RL model."""
        # Get current state and available actions
        state = self.state_manager.get_state_hash()
        
        # Exploration vs exploitation
        if random.random() < self.exploration_rate:
            # Explore: choose random action
            position = self.state_manager.choose_action_training()
        else:
            # Exploit: choose best action according to RL model
            position = self.state_manager.choose_action_rl()
        
        if position is None:
            await self._cash_out(client)  # No valid moves left, cash out
            return
        
        row, col = position
        
        # Record the state and action before taking it
        old_state = state
        old_revealed = self.state_manager.revealed_diamonds
        
        # Click the position
        await self._click_position(client, row, col)
        
        # Update exploration rate
        self.exploration_rate = max(MIN_EXPLORATION_RATE, 
                                   self.exploration_rate * EXPLORATION_DECAY)
    
    async def _click_position(self, client, row, col):
        """Click a position on the game grid."""
        if not self.current_message or not self.current_message.buttons:
            logger.warning("No buttons available to click")
            return
        
        try:
            # Calculate button index (5x5 grid)
            button_row = row
            button_col = col
            
            # Click the button
            self.last_response_time = time.time()
            await self.current_message.click(button_row, button_col)
            logger.debug(f"Clicked position ({row}, {col})")
            
            # Wait briefly for the result
            await asyncio.sleep(0.5)
            
        except Exception as e:
            logger.error(f"Error clicking position ({row}, {col}): {str(e)}")
    
    async def _cash_out(self, client):
        """Cash out the current game if possible."""
        if not self.current_message or not self.current_message.buttons:
            logger.warning("No buttons available to cash out")
            return
        
        try:
            # Look for the "Cash Out" button - usually in the last row
            for row_idx, row in enumerate(self.current_message.buttons):
                for col_idx, button in enumerate(row):
                    if button.text and "Cash Out" in button.text:
                        self.last_response_time = time.time()
                        await self.current_message.click(row_idx, col_idx)
                        logger.info(f"Cashed out with {self.state_manager.revealed_diamonds} diamonds")
                        return
            
            logger.warning("No Cash Out button found")
            
        except Exception as e:
            logger.error(f"Error cashing out: {str(e)}")
    
    def _extract_win_amount(self, text: str) -> float:
        """Extract the win amount from the game result text."""
        try:
            match = re.search(r"You won (\d+(?:\.\d+)?)", text)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return 0.0
    
    def _extract_multiplier(self, text: str) -> float:
        """Extract the multiplier from the game result text."""
        try:
            match = re.search(r"multiplier (\d+(?:\.\d+)?)", text)
            if match:
                return float(match.group(1))
        except Exception:
            pass
        return 1.0
    
    async def _extract_final_state(self, text: str):
        """Extract the final state of the game from the game over message."""
        # This would depend on the exact format of the game bot's messages
        # Here we'll implement a simple version
        bomb_positions = []
        diamond_positions = []
        
        # Hypothetical parsing - would need to be adapted to the actual game bot's output
        # For example, if the text contains coordinates of bombs/diamonds
        
        # Save the data for RL training
        if bomb_positions:
            self.state_manager.record_bomb_positions(self.current_game_id, bomb_positions)
        
        if diamond_positions:
            self.state_manager.record_diamond_positions(self.current_game_id, diamond_positions)
from telethon import TelegramClient, events
from telethon.tl.custom import Button, Message
import asyncio
import logging
import re
from typing import List, Dict, Optional

from config import (
    API_ID, API_HASH, SESSION_NAME, GROUP_ID, AUTHORIZED_USERS,
    GAME_BOT_USERNAME, DEFAULT_BET, DEFAULT_BOMBS
)
from state_manager import StateManager
from ai_game_handler import AIGameHandler
from logger import setup_logger

# Set up logging
logger = setup_logger('bot_controller')

class BotController:
    def __init__(self):
        """Initialize the Telegram client and set up event handlers."""
        self.client = TelegramClient(SESSION_NAME, API_ID, API_HASH)
        self.state_manager = StateManager()
        self.ai_handler = AIGameHandler(self.state_manager)
        self.game_task = None
        self.setup_handlers()

    def setup_handlers(self):
        """Set up event handlers for incoming messages."""
        
        @self.client.on(events.NewMessage(chats=GROUP_ID))
        async def command_handler(event):
            """Handle commands from authorized users."""
            if event.sender_id not in AUTHORIZED_USERS:
                return
                
            message = event.message.text
            
            if message.startswith('/startai'):
                await self.handle_start_ai(event)
            elif message.startswith('/stopai'):
                await self.handle_stop_ai(event)
            elif message.startswith('/trainrl'):
                await self.handle_train_rl(event)
            elif message.startswith('/userl'):
                await self.handle_use_rl(event)
            elif message.startswith('/setbet'):
                await self.handle_set_bet(event)
            elif message.startswith('/setbombs'):
                await self.handle_set_bombs(event)
            elif message.startswith('/status'):
                await self.handle_status(event)
            elif message.startswith('/resume'):
                await self.handle_resume(event)
        
        # Handler for game bot's messages and buttons
        @self.client.on(events.NewMessage(from_users=GAME_BOT_USERNAME.replace('@', '')))
        async def game_message_handler(event):
            """Handle messages from the game bot."""
            # Forward to AI handler
            await self.ai_handler.process_game_message(event)

    async def handle_start_ai(self, event):
        """Start the automated game loop."""
        if self.state_manager.is_running:
            await self.client.send_message(GROUP_ID, "🔄 AI is already running.")
            return
            
        self.state_manager.is_running = True
        await self.client.send_message(GROUP_ID, "🔄 /startai received – The script is playing the game.")
        
        # Start the game loop in a separate task
        if self.game_task is None or self.game_task.done():
            self.game_task = asyncio.create_task(self.ai_handler.game_loop(self.client))

    async def handle_stop_ai(self, event):
        """Stop the automated game loop."""
        if not self.state_manager.is_running:
            await self.client.send_message(GROUP_ID, "⚠️ AI is not currently running.")
            return
            
        self.state_manager.is_running = False
        
        if self.game_task and not self.game_task.done():
            self.game_task.cancel()
            
        await self.client.send_message(GROUP_ID, "🛑 AI stopped. Send /startai to resume.")

    async def handle_train_rl(self, event):
        """Switch to training mode."""
        self.state_manager.use_rl = False
        self.state_manager.training_mode = True
        await self.client.send_message(GROUP_ID, "🧠 Switched to training mode. Will cash out after 3 diamonds.")

    async def handle_use_rl(self, event):
        """Switch to live RL mode."""
        self.state_manager.use_rl = True
        self.state_manager.training_mode = False
        await self.client.send_message(GROUP_ID, "🎮 Switched to live RL mode. Using trained model for gameplay.")

    async def handle_set_bet(self, event):
        """Set the bet amount."""
        message = event.message.text
        try:
            bet = int(message.split(' ')[1])
            if bet <= 0:
                await self.client.send_message(GROUP_ID, "⚠️ Bet amount must be positive.")
                return
                
            self.state_manager.bet_amount = bet
            await self.client.send_message(GROUP_ID, f"💰 Bet amount set to {bet}.")
        except (IndexError, ValueError):
            await self.client.send_message(GROUP_ID, "⚠️ Invalid format. Use: /setbet <amount>")

    async def handle_set_bombs(self, event):
        """Set the number of bombs."""
        message = event.message.text
        try:
            bombs = int(message.split(' ')[1])
            if bombs <= 2 or bombs >= 24:  # Maximum 24 bombs in a 5x5 grid
                await self.client.send_message(GROUP_ID, "⚠️ Number of bombs must be between 3 and 23.")
                return
                
            self.state_manager.bombs = bombs
            await self.client.send_message(GROUP_ID, f"💣 Number of bombs set to {bombs}.")
        except (IndexError, ValueError):
            await self.client.send_message(GROUP_ID, "⚠️ Invalid format. Use: /setbombs <number>")

    async def handle_status(self, event):
        """Show the current status."""
        status_message = (
            f"📊 **Current Status**\n"
            f"Running: {'Yes' if self.state_manager.is_running else 'No'}\n"
            f"Mode: {'Training' if self.state_manager.training_mode else 'Live RL'}\n"
            f"Bet Amount: {self.state_manager.bet_amount}\n"
            f"Bombs: {self.state_manager.bombs}\n"
            f"Games Played: {self.state_manager.games_played}\n"
            f"Wins: {self.state_manager.wins}\n"
            f"Losses: {self.state_manager.losses}"
        )
        await self.client.send_message(GROUP_ID, status_message)

    async def handle_resume(self, event):
        """Resume after manual intervention."""
        if not self.state_manager.waiting_for_resume:
            await self.client.send_message(GROUP_ID, "⚠️ AI is not waiting to be resumed.")
            return
            
        self.state_manager.waiting_for_resume = False
        await self.client.send_message(GROUP_ID, "▶️ Resuming gameplay.")
        
        if self.game_task is None or self.game_task.done():
            self.game_task = asyncio.create_task(self.ai_handler.game_loop(self.client))

    async def run(self):
        """Start the client and run the bot."""
        await self.client.start()
        logger.info("✅ Script started – Waiting for /startai command.")
        await self.client.send_message(GROUP_ID, "✅ Script started – Waiting for /startai command.")
        
        # Run the client until disconnected
        await self.client.run_until_disconnected()

if __name__ == "__main__":
    bot = BotController()
    asyncio.run(bot.run())
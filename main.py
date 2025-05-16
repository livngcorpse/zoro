import asyncio
import json
import os
from telethon import TelegramClient, events
from bot_controller import BotController
from logger import setup_logger

# Set up logger
logger = setup_logger('main')

async def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting Autonomous Telegram Mines Game Script")
        
        # Create the BotController instance
        bot = BotController()
        
        # Run the bot
        await bot.run()
        
    except Exception as e:
        logger.error(f"Critical error: {str(e)}")
        
if __name__ == "__main__":
    asyncio.run(main())
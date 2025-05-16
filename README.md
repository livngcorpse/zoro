# Autonomous Telegram Mines Game Script

This script uses reinforcement learning to autonomously play a Mines-style game in Telegram.

## Prerequisites

- Python 3.10
- A Telegram account for the bot to use
- A Telegram group with the game bot and accounts

## Installation

1. Clone the repository
2. Install dependencies:

pip install -r requirements.txt

3. Configure the script by editing `config.py` with your Telegram API credentials and other settings.

## Usage

1. Run the script:

python main.py

2. The script will start and wait for commands in the configured Telegram group.

### Commands

The following commands can be used in the Telegram group to control the automated account:

| Command              |  Description   ----------------------------------- | 
| ------------------------------------------------------------------------  |
| `/startai`           | Start automatic game loop (script begins running). |
| `/stopai`            | Stop the script.                                   |
| `/trainrl`           | Enable training mode (clicks random tiles, cashes out after 3 diamonds). |
| `/userl`             | Enable live mode (uses RL to maximize rewards).    |
| `/setbet <amount>`   | Set custom bet amount (default: 20).               |
| `/setbombs <number>` | Set the number of bombs (default: 3).              |
| `/status`            | Show the current status, including bet and bombs.  |
| `/resume`            | Resume gameplay after manual intervention.         |

## Project Structure

- `config.py`: Configuration settings for the bot
- `main.py`: Main entry point for the application
- `bot_controller.py`: Handles Telegram commands and event handlers
- `state_manager.py`: Manages the state of the game and AI
- `ai_game_handler.py`: Core game logic for automated play
- `rl_model.py`: Reinforcement learning model for decision making
- `logger.py`: Logging utilities

## Files Requirements

Create a `requirements.txt` file with the following dependencies:

telethon==1.28.5
numpy==1.24.3
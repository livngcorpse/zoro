# Configuration settings for the Mines game automation

# Telegram API configuration
API_ID = '28730671'  # Get from https://my.telegram.org/apps
API_HASH = '705524ea5d906623ccd4027aa44c2bc7'  # Get from https://my.telegram.org/apps
SESSION_NAME = 'minesbot'  # Session name for telethon

# Game settings
DEFAULT_BET = 20
DEFAULT_BOMBS = 3
GAME_BOT_USERNAME = '@roronoa_zoro_robot'  # Replace with the game bot's username
GROUP_ID = -1002655598369  # Replace with your group ID

# Authorized users who can send commands (Telegram user IDs)
AUTHORIZED_USERS = [
    7993404275,  # Replace with actual user IDs
]

# Game parameters
GRID_SIZE = 5
MAX_WAIT_TIME = 30  # Maximum time (in seconds) to wait for game bot response
TRAINING_CASHOUT = 3  # Number of diamonds to reveal before cashing out in training mode

# RL parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EXPLORATION_RATE = 0.2
EXPLORATION_DECAY = 0.995
MIN_EXPLORATION_RATE = 0.01

# Logging levels
LOG_LEVEL = 'INFO'
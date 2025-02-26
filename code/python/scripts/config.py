'''
Configuration shared across notebooks
'''
from pathlib import Path
from dotenv import load_dotenv
import os

# Navigate to the repository root (assumes "aus-historical-solar-droughts" is the repo name)
root = Path(__file__).resolve().parents[3]  # Go up 3 levels to reach the repo root

# Load the .env file from the repository root
env_path = root / ".env"
load_dotenv(env_path)

# Example: Accessing an environment variable
DATA_DIRS = os.getenv("DATA_DIRS")
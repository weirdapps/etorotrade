import os
from dotenv import load_dotenv

def load_environment():
    """Load environment variables from .env file"""
    load_dotenv()
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please make sure it's added to the .env file.")
    return api_key
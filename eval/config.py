import os

from dotenv import load_dotenv

load_dotenv()

DIAL_URL = os.getenv("DIAL_URL")
DIAL_API_KEY = os.getenv("DIAL_API_KEY")
MIND_MAP_FRONTEND_URL = os.getenv("MIND_MAP_FRONTEND_URL")

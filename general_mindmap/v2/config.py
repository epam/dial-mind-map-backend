import os

if (DIAL_URL := os.getenv("DIAL_URL")) is None:
    from dotenv import load_dotenv

    load_dotenv()
    DIAL_URL = os.getenv("DIAL_URL") or ""

if DIAL_URL == "":
    raise ValueError("DIAL_URL is not specified")

import os

if (DIAL_URL := os.getenv("DIAL_URL")) is None:
    from dotenv import load_dotenv

    load_dotenv()
    DIAL_URL = os.getenv("DIAL_URL") or ""

if DIAL_URL == "":
    raise ValueError("DIAL_URL is not specified")

os.environ["AZURE_OPENAI_ENDPOINT"] = DIAL_URL
os.environ["AZURE_OPENAI_API_KEY"] = "dial_api_key"
os.environ["OPENAI_API_VERSION"] = "2024-10-21"

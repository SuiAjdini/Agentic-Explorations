import os
from dotenv import load_dotenv

# Load environment variables from the .env file in the project root
load_dotenv()

# --- API Keys ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")

# --- Model Configuration ---
LLM_MODEL_NAME = "models/gemini-1.5-flash-latest"
EMBED_MODEL_NAME = "local:BAAI/bge-small-en-v1.5"

# --- File Paths ---
CV_DATA_PATH = "./cv_data"
FINAL_OUTPUT_FILENAME = "extracted_info_final.json"
import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent.parent
load_dotenv(APP_DIR / ".env")

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash")
DB_PATH = Path(os.getenv("DB_PATH", ROOT_DIR / "database" / "clinical_trials.db"))
UPLOAD_DIR = ROOT_DIR / "backend" / "data" / "uploads"
FDA_DIR = ROOT_DIR / "fda"
MODELS_DIR = APP_DIR / "ml" / "models"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

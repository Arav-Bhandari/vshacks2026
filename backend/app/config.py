import os
from pathlib import Path

from dotenv import load_dotenv

APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent.parent
load_dotenv(APP_DIR / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DB_PATH = Path(os.getenv("DB_PATH", ROOT_DIR / "database" / "clinical_trials.db"))
UPLOAD_DIR = ROOT_DIR / "backend" / "data" / "uploads"
FDA_DIR = ROOT_DIR / "fda"
MODELS_DIR = APP_DIR / "ml" / "models"

SONNET_MODEL = os.getenv("SONNET_MODEL", "claude-sonnet-4-5")
HAIKU_MODEL = os.getenv("HAIKU_MODEL", "claude-haiku-4-5-20251001")

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

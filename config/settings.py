"""
Application configuration and path settings for AgriPredict AI.
Handles dynamic relative path resolution, dataset discovery, and UI styling parameters.
"""

from pathlib import Path
import os

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# Core subdirectories
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAVED_MODELS_DIR = BASE_DIR / "saved_models"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure runtime directories exist
for path in [PROCESSED_DATA_DIR, SAVED_MODELS_DIR, REPORTS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# Dataset file specifications (preferred name, alternate names)
DATASET_SPECS = {
    "crop_production": {
        "title": "Crop Production Across India",
        "filenames": ["crop_production.csv"],
        "description": "Historical crop cultivation area and production metrics across Indian states and districts (1997-2015)."
    },
    "fertilizer": {
        "title": "Fertilizer Recommendation Dataset",
        "filenames": ["Fertilizer Prediction.csv", "fertilizer_prediction.csv"],
        "description": "Soil conditions, crop types, and recommended chemical fertilizer formulations."
    },
    "market_prices": {
        "title": "Indian Agricultural Market Prices (APMC/MSP)",
        "filenames": ["Indian Agricultural Market Prices.csv", "market_prices.csv"],
        "description": "Market modal, minimum, and maximum prices alongside Government Minimum Support Prices (MSP) in Rs/Quintal."
    },
    "soil_recommendation": {
        "title": "Soil Nutrients and Crop Recommendation",
        "filenames": [
            "Soil Nutrients and Crop Recommendation Dataset.csv",
            "Soil Nutrients and Crop Recommendation.csv",
            "soil_recommendation.csv"
        ],
        "description": "Soil macronutrients (N, P, K), pH, electrical conductivity, and micronutrients mapped to suitable crops."
    },
    "weather": {
        "title": "Historical Weather Observations (8 Indian Cities)",
        "folder_names": ["WEATHER DATASET", "weather_dataset", "WEATHER_DATASET"],
        "cities": ["bengaluru", "bombay", "delhi", "hyderabad", "jaipur", "kanpur", "nagpur", "pune"],
        "description": "Hourly historical meteorological records (2009-2020) across 8 major agricultural/urban regional centers."
    }
}


def resolve_dataset_path(dataset_key: str) -> Path | None:
    """
    Dynamically locate a dataset file or folder.
    Searches first in data/raw/, then in the workspace root directory.
    Returns Path if found, otherwise None.
    """
    if dataset_key not in DATASET_SPECS:
        return None

    spec = DATASET_SPECS[dataset_key]

    if dataset_key == "weather":
        # Search for weather directory
        for folder_name in spec["folder_names"]:
            raw_path = RAW_DATA_DIR / folder_name
            if raw_path.is_dir():
                return raw_path
            root_path = BASE_DIR / folder_name
            if root_path.is_dir():
                return root_path
        return None
    else:
        # Search for CSV file
        for fname in spec["filenames"]:
            raw_path = RAW_DATA_DIR / fname
            if raw_path.is_file():
                return raw_path
            root_path = BASE_DIR / fname
            if root_path.is_file():
                return root_path
        return None


def get_all_dataset_status() -> dict:
    """
    Returns a status dictionary detailing the existence and paths of all configured datasets.
    """
    status = {}
    for key, spec in DATASET_SPECS.items():
        resolved = resolve_dataset_path(key)
        status[key] = {
            "title": spec["title"],
            "found": resolved is not None,
            "path": str(resolved) if resolved else None,
            "description": spec["description"]
        }
    return status


# UI & Design System Tokens (Agriculture & Modern Premium Theme)
THEME = {
    "primary": "#1B4D3E",       # Deep forest / pine green
    "primary_light": "#2E7D32", # Rich emerald green
    "secondary": "#4CAF50",     # Vibrant leaf green
    "accent": "#81C784",        # Soft mint green
    "accent_gold": "#E6A100",   # Golden harvest yellow for highlights
    "bg_subtle": "#F4F7F4",     # Pristine off-white/cream with subtle cool tint
    "card_bg": "#FFFFFF",       # Pure white for cards
    "card_border": "#E2E8F0",   # Soft slate border
    "text_dark": "#132A13",     # Near-black organic green
    "text_muted": "#556B2F",    # Muted olive for captions
    "danger": "#D32F2F",        # Red
    "warning": "#F57C00",       # Amber
    "success": "#2E7D32",       # Forest green
    "info": "#0288D1"           # Sky blue
}

PLOTLY_COLORS = [
    "#1B4D3E", "#2E7D32", "#4CAF50", "#81C784", 
    "#E6A100", "#FF7043", "#26A69A", "#42A5F5", 
    "#7E57C2", "#EC407A", "#8D6E63", "#78909C"
]

CUSTOM_CSS = """
<style>
    /* Global modern typography & styling */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #1B4D3E 0%, #2E7D32 50%, #388E3C 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(27, 77, 62, 0.25);
    }
    .app-header h1 {
        color: white !important;
        font-weight: 700;
        font-size: 2.2rem;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: #E8F5E9 !important;
        font-size: 1.05rem;
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    /* Modern Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(27, 77, 62, 0.08);
        border-color: #A5D6A7;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #556B2F;
        margin-bottom: 0.4rem;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #132A13;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 0.3rem;
    }

    /* Status indicators */
    .badge-success {
        background-color: #E8F5E9;
        color: #2E7D32;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
    .badge-warning {
        background-color: #FFF3E0;
        color: #E65100;
        padding: 0.25rem 0.6rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.8rem;
        display: inline-block;
    }
</style>
"""

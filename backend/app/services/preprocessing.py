import json
import os
import pandas as pd
try:
    from app.schemas.prediction import PredictionRequest
    from app.utils.logging_config import logger
except ImportError:
    from backend.app.schemas.prediction import PredictionRequest
    from backend.app.utils.logging_config import logger

# Load valid locations list
_allowed_locations = set()

def load_allowed_locations(locations_path: str):
    global _allowed_locations
    if os.path.exists(locations_path):
        try:
            with open(locations_path, "r", encoding="utf-8") as f:
                locs = json.load(f)
                _allowed_locations = set(str(loc).strip().lower() for loc in locs)
                logger.info(f"Loaded {len(_allowed_locations)} allowed locations from {locations_path}")
        except Exception as e:
            logger.warning(f"Could not load locations from {locations_path}: {e}")
    else:
        logger.warning(f"Locations file not found at {locations_path}")

def format_inr_price(amount: float) -> str:
    """Format currency nicely in Lacs / Crores."""
    if amount >= 1e7:
        cr = amount / 1e7
        return f"₹ {cr:.2f} Cr"
    else:
        lac = amount / 1e5
        return f"₹ {lac:.2f} Lac"

def preprocess_request(request: PredictionRequest) -> pd.DataFrame:
    """
    Transforms a PredictionRequest into a 1-row DataFrame with the exact column names
    expected by the trained Scikit-Learn ColumnTransformer pipeline.
    Unknown locations are mapped to 'other'.
    """
    loc = request.location.strip().lower()
    if _allowed_locations and loc not in _allowed_locations:
        location_grouped = "other"
    else:
        location_grouped = loc

    row_data = {
        "carpet_area_sqft": [float(request.carpet_area_sqft)],
        "floor_num": [int(request.floor_num)],
        "bathroom": [int(request.bathroom)],
        "balcony": [int(request.balcony)],
        "location_grouped": [location_grouped],
        "Furnishing": [request.furnishing.strip()],
        "Transaction": [request.transaction.strip()],
        "Ownership": [request.ownership.strip()],
        "facing": [request.facing.strip()]
    }

    df = pd.DataFrame(row_data)
    return df

import joblib
import os
import pandas as pd
from backend.app.schemas.prediction import PredictionResponse
from backend.app.services.preprocessing import format_inr_price
from backend.app.utils.logging_config import logger

def load_model(model_path: str):
    """Load serialized scikit-learn pipeline from disk."""
    if not os.path.exists(model_path):
        # Fallback to alternate path if running from different cwd
        alt_path = os.path.join("backend", model_path)
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {alt_path}")
    
    logger.info(f"Loading model pipeline from {model_path}...")
    model = joblib.load(model_path)
    logger.info("Model pipeline loaded successfully.")
    return model

def predict_price(model, df: pd.DataFrame) -> PredictionResponse:
    """Run model inference on 1-row DataFrame and build formatted response."""
    pred_array = model.predict(df)
    predicted_val = float(pred_array[0])
    # Guard against negative or absurd low predictions
    predicted_val = max(100000.0, predicted_val)

    formatted = format_inr_price(predicted_val)
    price_lac = round(predicted_val / 1e5, 2)
    price_cr = round(predicted_val / 1e7, 4)

    return PredictionResponse(
        predicted_price=round(predicted_val, 2),
        formatted_price=formatted,
        currency="INR",
        price_lac=price_lac,
        price_cr=price_cr,
        status="success"
    )

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

from backend.app.core.config import settings
from backend.app.api.routes.prediction import router as prediction_router
from backend.app.services.inference import load_model
from backend.app.services.preprocessing import load_allowed_locations
from backend.app.utils.logging_config import logger

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager to load the model once at application startup."""
    logger.info("Initializing application resources...")
    # Determine model and locations paths
    model_path = settings.MODEL_PATH
    if not os.path.exists(model_path):
        candidates = [
            os.path.join(os.path.dirname(__file__), "..", "models", "house_price.pkl"),
            os.path.join(os.path.dirname(__file__), "..", "..", "models", "house_price.pkl"),
            "models/house_price.pkl",
            "backend/models/house_price.pkl"
        ]
        for c in candidates:
            if os.path.exists(c):
                model_path = c
                break

    locs_path = settings.LOCATIONS_PATH
    if not os.path.exists(locs_path):
        candidates_loc = [
            os.path.join(os.path.dirname(__file__), "..", "locations.json"),
            os.path.join(os.path.dirname(__file__), "..", "..", "locations.json"),
            "locations.json",
            "backend/locations.json"
        ]
        for c in candidates_loc:
            if os.path.exists(c):
                locs_path = c
                break

    try:
        app.state.model = load_model(model_path)
        load_allowed_locations(locs_path)
        logger.info(f"Startup complete: Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        app.state.model = None

    yield

    logger.info("Shutting down application resources...")
    app.state.model = None

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Production API serving an end-to-end trained House Price Prediction Machine Learning Pipeline.",
    lifespan=lifespan
)

# CORS middleware allowing frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include prediction router
app.include_router(prediction_router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {
        "message": "House Price Prediction API is online.",
        "docs": "/docs",
        "health": "/health"
    }

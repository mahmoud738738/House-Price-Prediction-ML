from fastapi import APIRouter, Request, HTTPException, status
try:
    from app.schemas.prediction import PredictionRequest, PredictionResponse, HealthResponse
    from app.services.preprocessing import preprocess_request
    from app.services.inference import predict_price
    from app.utils.logging_config import logger
except ImportError:
    from backend.app.schemas.prediction import PredictionRequest, PredictionResponse, HealthResponse
    from backend.app.services.preprocessing import preprocess_request
    from backend.app.services.inference import predict_price
    from backend.app.utils.logging_config import logger

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request):
    """Health check endpoint to verify service and model availability."""
    model_loaded = hasattr(request.app.state, "model") and request.app.state.model is not None
    return HealthResponse(status="ok", model_loaded=model_loaded, version="1.0.0")

@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
async def predict_house_price(prediction_request: PredictionRequest, request: Request):
    """
    Predict property price based on physical specifications, floor, and location.
    Preprocessing converts categories and handles unobserved locations automatically.
    """
    model = getattr(request.app.state, "model", None)
    if model is None:
        logger.error("Inference attempted but model is not loaded in application state.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not currently loaded."
        )

    try:
        df_input = preprocess_request(prediction_request)
        response = predict_price(model, df_input)
        return response
    except Exception as e:
        logger.error(f"Inference error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference computation failed: {str(e)}"
        )

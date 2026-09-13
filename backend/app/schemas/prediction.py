from pydantic import BaseModel, Field
from typing import Optional

class PredictionRequest(BaseModel):
    location: str = Field(..., description="Property location/locality", min_length=1, examples=["thane"])
    carpet_area_sqft: float = Field(..., description="Carpet area in square feet", gt=0, examples=[850.0])
    floor_num: int = Field(..., description="Floor number (0 for Ground, -1 for Basement)", ge=-2, le=150, examples=[3])
    bathroom: int = Field(..., description="Number of bathrooms", ge=1, le=20, examples=[2])
    balcony: int = Field(..., description="Number of balconies", ge=0, le=10, examples=[1])
    furnishing: str = Field(..., description="Furnishing status ('Furnished', 'Semi-Furnished', 'Unfurnished')", examples=["Semi-Furnished"])
    transaction: str = Field(..., description="Transaction type ('Resale', 'New Property')", examples=["Resale"])
    ownership: str = Field(default="Freehold", description="Ownership type", examples=["Freehold"])
    facing: str = Field(default="East", description="Facing direction", examples=["East"])

class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., description="Estimated property value in Indian Rupees (INR)")
    formatted_price: str = Field(..., description="Human-readable formatted price (e.g., ₹ 85.50 Lac, ₹ 1.25 Cr)")
    currency: str = Field(default="INR", description="Currency unit")
    price_lac: float = Field(..., description="Price in Lacs (₹ 100,000s)")
    price_cr: float = Field(..., description="Price in Crores (₹ 10,000,000s)")
    status: str = Field(default="success", description="Prediction status")

class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool = True
    version: str = "1.0.0"

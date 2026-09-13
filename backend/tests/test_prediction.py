import pytest
from fastapi.testclient import TestClient
try:
    from app.main import app
except ImportError:
    from backend.app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True

def test_predict_happy_path(client):
    payload = {
        "location": "thane",
        "carpet_area_sqft": 850.0,
        "floor_num": 3,
        "bathroom": 2,
        "balcony": 1,
        "furnishing": "Semi-Furnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_price" in data
    assert isinstance(data["predicted_price"], (int, float))
    assert data["predicted_price"] > 0
    assert "formatted_price" in data
    assert "₹" in data["formatted_price"]
    assert data["status"] == "success"

def test_predict_invalid_input_422_negative_area(client):
    # Carpet area must be > 0
    payload = {
        "location": "thane",
        "carpet_area_sqft": -50.0,
        "floor_num": 2,
        "bathroom": 1,
        "balcony": 0,
        "furnishing": "Unfurnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "East"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_input_422_missing_field(client):
    # Missing required carpet_area_sqft and location
    payload = {
        "floor_num": 2,
        "bathroom": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_unknown_location_maps_to_other(client):
    # Unseen location should be safely mapped to "other" without throwing an error
    payload = {
        "location": "unknown_future_city_99",
        "carpet_area_sqft": 1200.0,
        "floor_num": 5,
        "bathroom": 3,
        "balcony": 2,
        "furnishing": "Furnished",
        "transaction": "New Property",
        "ownership": "Freehold",
        "facing": "North"
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_price"] > 0

def test_predict_ground_and_basement_floors(client):
    payload_ground = {
        "location": "thane",
        "carpet_area_sqft": 600.0,
        "floor_num": 0,
        "bathroom": 1,
        "balcony": 0,
        "furnishing": "Unfurnished",
        "transaction": "Resale",
        "ownership": "Freehold",
        "facing": "West"
    }
    res_ground = client.post("/predict", json=payload_ground)
    assert res_ground.status_code == 200

export interface PredictionRequest {
  location: string;
  carpet_area_sqft: number;
  floor_num: number;
  bathroom: number;
  balcony: number;
  furnishing: string;
  transaction: string;
  ownership: string;
  facing: string;
}

export interface PredictionResponse {
  predicted_price: number;
  formatted_price: string;
  currency: string;
  price_lac: number;
  price_cr: number;
  status: string;
}

export interface PredictionResultState {
  input: PredictionRequest;
  result: PredictionResponse;
}

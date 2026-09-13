import type { PredictionRequest, PredictionResponse } from '../types/prediction';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export async function checkHealth(): Promise<{ status: string; model_loaded: boolean }> {
  try {
    const res = await fetch(`${BASE_URL}/health`);
    if (!res.ok) {
      throw new Error(`Server returned status ${res.status}`);
    }
    return await res.json();
  } catch (error) {
    console.error('Health check failed:', error);
    throw new Error('Unable to connect to the backend server. Please verify the API is running.');
  }
}

export async function predictPrice(data: PredictionRequest): Promise<PredictionResponse> {
  try {
    const response = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      let errorMessage = `Prediction request failed with status: ${response.status}`;
      try {
        const errorData = await response.json();
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            errorMessage = errorData.detail.map((d: any) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join(', ');
          } else {
            errorMessage = errorData.detail;
          }
        }
      } catch {
        // use default error message
      }
      throw new Error(errorMessage);
    }

    return await response.json();
  } catch (err: any) {
    console.error('Error calling /predict:', err);
    throw new Error(err.message || 'Network error occurred while calling the prediction service.');
  }
}

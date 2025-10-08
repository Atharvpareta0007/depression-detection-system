// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001';

export const API_ENDPOINTS = {
  predict: `${API_BASE_URL}/api/predict`,
  metrics: `${API_BASE_URL}/api/metrics`,
};

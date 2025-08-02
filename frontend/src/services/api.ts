import axios, { AxiosResponse, AxiosError } from 'axios';
import { 
  AnalyticsStats, 
  Transaction, 
  FraudAlert, 
  UserRiskProfile, 
  ModelMetrics, 
  HealthCheck,
  TransactionRequest,
  FraudDetectionResult,
  ApiError,
  TransactionFilters,
  AlertFilters
} from '../types';

// Create axios instance with default config
const api = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8000/api',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add auth token if available
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response: AxiosResponse) => {
    return response;
  },
  (error: AxiosError) => {
    const apiError: ApiError = {
      message: error.response?.data?.detail || error.message || 'An error occurred',
      status: error.response?.status || 500,
      details: error.response?.data,
    };
    return Promise.reject(apiError);
  }
);

// API endpoints
export class ApiService {
  // Health check
  static async healthCheck(): Promise<HealthCheck> {
    const response = await api.get('/health');
    return response.data;
  }

  // Transaction endpoints
  static async checkTransaction(transaction: TransactionRequest): Promise<FraudDetectionResult> {
    const response = await api.post('/transactions/check', transaction);
    return response.data;
  }

  static async checkBulkTransactions(transactions: TransactionRequest[]): Promise<FraudDetectionResult[]> {
    const response = await api.post('/transactions/bulk', { transactions });
    return response.data;
  }

  static async getTransactionHistory(
    filters: TransactionFilters & { limit?: number; offset?: number } = {}
  ): Promise<Transaction[]> {
    const response = await api.get('/transactions/history', { params: filters });
    return response.data;
  }

  static async getTransaction(transactionId: number): Promise<Transaction> {
    const response = await api.get(`/transactions/${transactionId}`);
    return response.data;
  }

  // Analytics endpoints
  static async getAnalyticsStats(days: number = 7): Promise<AnalyticsStats> {
    const response = await api.get('/analytics/stats', { params: { days } });
    return response.data;
  }

  static async getUserRiskProfile(userId: string): Promise<UserRiskProfile> {
    const response = await api.get(`/analytics/user-risk/${userId}`);
    return response.data;
  }

  // Alert endpoints
  static async getAlerts(filters: AlertFilters & { limit?: number } = {}): Promise<FraudAlert[]> {
    const response = await api.get('/alerts', { params: filters });
    return response.data;
  }

  static async updateAlertStatus(
    alertId: number, 
    status: string, 
    notes?: string
  ): Promise<{ alert_id: number; status: string; updated_at: string }> {
    const response = await api.put(`/alerts/${alertId}/status`, { status, notes });
    return response.data;
  }

  // Model endpoints
  static async getModelMetrics(): Promise<ModelMetrics[]> {
    const response = await api.get('/models/metrics');
    return response.data;
  }

  static async retrainModels(): Promise<{ message: string; training_samples: number }> {
    const response = await api.post('/models/retrain');
    return response.data;
  }
}

// Export the api instance for custom requests
export default api;
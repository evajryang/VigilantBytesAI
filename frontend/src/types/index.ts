// API Response Types
export interface FraudDetectionResult {
  is_fraud: boolean;
  fraud_score: number;
  confidence: number;
  fraud_reasons: string[];
  model_version: string;
  processing_time_ms: number;
}

export interface Transaction {
  id: number;
  user_id: string;
  merchant_id: string;
  amount: number;
  currency: string;
  transaction_type: string;
  timestamp: string;
  latitude?: number;
  longitude?: number;
  country?: string;
  city?: string;
  is_fraud: boolean;
  fraud_score: number;
  fraud_reasons?: string[];
  model_version?: string;
  processed_at: string;
  processing_time_ms?: number;
}

export interface AnalyticsStats {
  total_transactions: number;
  fraud_transactions: number;
  fraud_rate: number;
  total_volume: number;
  fraud_volume: number;
  avg_transaction_amount: number;
  avg_fraud_amount: number;
  top_fraud_reasons: Array<{
    reason: string;
    count: number;
  }>;
  hourly_stats: Array<{
    hour: number;
    total_transactions: number;
    fraud_transactions: number;
    fraud_rate: number;
    volume: number;
  }>;
  daily_stats: Array<{
    date: string;
    total_transactions: number;
    fraud_transactions: number;
    fraud_rate: number;
    volume: number;
  }>;
}

export interface FraudAlert {
  id: number;
  transaction_id: number;
  alert_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  details?: Record<string, any>;
  status: 'active' | 'investigated' | 'resolved' | 'false_positive';
  created_at: string;
}

export interface UserRiskProfile {
  user_id: string;
  risk_score: number;
  transaction_count: number;
  avg_amount: number;
  countries_count: number;
  merchants_count: number;
  fraud_count: number;
  last_transaction?: string;
  risk_factors: string[];
}

export interface ModelMetrics {
  model_name: string;
  model_version: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  auc_roc?: number;
  training_samples?: number;
  fraud_samples?: number;
  legitimate_samples?: number;
  deployed_at: string;
  is_active: boolean;
}

export interface HealthCheck {
  status: string;
  timestamp: string;
  version: string;
  database_status: string;
  redis_status: string;
  model_status: string;
}

// WebSocket Message Types
export interface WSMessage {
  type: string;
  data: any;
  timestamp: string;
}

export interface RealTimeAlert {
  alert_id: string;
  transaction_id: number;
  user_id: string;
  alert_type: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  fraud_score: number;
  amount: number;
  merchant_id: string;
  timestamp: string;
  requires_action: boolean;
}

// Form Types
export interface TransactionRequest {
  user_id: string;
  merchant_id: string;
  amount: number;
  currency: string;
  transaction_type: 'purchase' | 'withdrawal' | 'transfer' | 'deposit' | 'refund';
  latitude?: number;
  longitude?: number;
  country?: string;
  city?: string;
  device_id?: string;
  ip_address?: string;
  user_agent?: string;
  session_id?: string;
  description?: string;
  merchant_category?: string;
  payment_method?: 'credit_card' | 'debit_card' | 'bank_transfer' | 'digital_wallet' | 'cryptocurrency' | 'cash';
}

// UI State Types
export interface DashboardState {
  isLoading: boolean;
  error: string | null;
  stats: AnalyticsStats | null;
  recentTransactions: Transaction[];
  activeAlerts: FraudAlert[];
  isRealTimeEnabled: boolean;
}

export interface ChartDataPoint {
  x: string | number;
  y: number;
  label?: string;
}

export interface AlertNotification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  timestamp: Date;
  autoClose?: boolean;
}

// Navigation Types
export interface NavItem {
  id: string;
  label: string;
  path: string;
  icon: string;
  badge?: number;
}

// Filter Types
export interface TransactionFilters {
  user_id?: string;
  merchant_id?: string;
  amount_min?: number;
  amount_max?: number;
  start_date?: string;
  end_date?: string;
  is_fraud?: boolean;
  transaction_type?: string;
  country?: string;
}

export interface AlertFilters {
  status?: string;
  severity?: string;
  alert_type?: string;
  start_date?: string;
  end_date?: string;
}

// API Client Types
export interface ApiResponse<T> {
  data: T;
  status: number;
  message?: string;
}

export interface ApiError {
  message: string;
  status: number;
  details?: any;
}

// Theme Types
export interface Theme {
  colors: {
    primary: string;
    secondary: string;
    success: string;
    warning: string;
    error: string;
    info: string;
    background: string;
    surface: string;
    text: {
      primary: string;
      secondary: string;
      disabled: string;
    };
    border: string;
  };
  spacing: {
    xs: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  borderRadius: string;
  shadows: {
    sm: string;
    md: string;
    lg: string;
  };
}

// Utility Types
export type SeverityColor = 'green' | 'yellow' | 'orange' | 'red';
export type TransactionStatus = 'legitimate' | 'suspicious' | 'fraud' | 'under_review';
export type ChartTimeframe = '1h' | '24h' | '7d' | '30d' | '90d';
export type SortDirection = 'asc' | 'desc';
export type ViewMode = 'grid' | 'list' | 'table';
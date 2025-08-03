from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class TransactionType(str, Enum):
    PURCHASE = "purchase"
    WITHDRAWAL = "withdrawal"
    TRANSFER = "transfer"
    DEPOSIT = "deposit"
    REFUND = "refund"


class PaymentMethod(str, Enum):
    CREDIT_CARD = "credit_card"
    DEBIT_CARD = "debit_card"
    BANK_TRANSFER = "bank_transfer"
    DIGITAL_WALLET = "digital_wallet"
    CRYPTOCURRENCY = "cryptocurrency"
    CASH = "cash"


class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    ACTIVE = "active"
    INVESTIGATED = "investigated"
    RESOLVED = "resolved"
    FALSE_POSITIVE = "false_positive"


# Request Models
class TransactionRequest(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    merchant_id: str = Field(..., description="Merchant identifier")
    amount: float = Field(..., gt=0, description="Transaction amount")
    currency: str = Field(default="USD", description="Currency code")
    transaction_type: TransactionType = Field(..., description="Type of transaction")
    
    # Location data (optional)
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    country: Optional[str] = None
    city: Optional[str] = None
    
    # Device and session info
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    session_id: Optional[str] = None
    
    # Transaction details
    description: Optional[str] = None
    merchant_category: Optional[str] = None
    payment_method: Optional[PaymentMethod] = None
    
    @validator('amount')
    def validate_amount(cls, v):
        if v <= 0:
            raise ValueError('Amount must be positive')
        if v > 1000000:  # $1M limit
            raise ValueError('Amount exceeds maximum limit')
        return v


class BulkTransactionRequest(BaseModel):
    transactions: List[TransactionRequest] = Field(..., max_items=1000)


# Response Models
class FraudDetectionResult(BaseModel):
    is_fraud: bool = Field(..., description="Whether transaction is classified as fraud")
    fraud_score: float = Field(..., ge=0, le=1, description="Fraud probability score (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence level")
    fraud_reasons: List[str] = Field(default=[], description="Reasons for fraud classification")
    model_version: str = Field(..., description="ML model version used")
    processing_time_ms: float = Field(..., description="Processing time in milliseconds")


class TransactionResponse(BaseModel):
    id: int
    user_id: str
    merchant_id: str
    amount: float
    currency: str
    transaction_type: str
    timestamp: datetime
    
    # Location data
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    country: Optional[str] = None
    city: Optional[str] = None
    
    # Fraud detection results
    is_fraud: bool
    fraud_score: float
    fraud_reasons: Optional[List[str]] = None
    model_version: Optional[str] = None
    
    # Metadata
    processed_at: datetime
    processing_time_ms: Optional[float] = None
    
    class Config:
        from_attributes = True


class FraudAlertResponse(BaseModel):
    id: int
    transaction_id: int
    alert_type: str
    severity: AlertSeverity
    message: str
    details: Optional[Dict[str, Any]] = None
    status: AlertStatus
    created_at: datetime
    
    class Config:
        from_attributes = True


class AnalyticsStats(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_rate: float
    total_volume: float
    fraud_volume: float
    avg_transaction_amount: float
    avg_fraud_amount: float
    top_fraud_reasons: List[Dict[str, Any]]
    hourly_stats: List[Dict[str, Any]]
    daily_stats: List[Dict[str, Any]]


class ModelMetricsResponse(BaseModel):
    model_name: str
    model_version: str
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    auc_roc: Optional[float] = None
    training_samples: Optional[int] = None
    fraud_samples: Optional[int] = None
    legitimate_samples: Optional[int] = None
    deployed_at: datetime
    is_active: bool
    
    class Config:
        from_attributes = True


class UserRiskProfile(BaseModel):
    user_id: str
    risk_score: float = Field(..., ge=0, le=1)
    transaction_count: int
    avg_amount: float
    countries_count: int
    merchants_count: int
    fraud_count: int
    last_transaction: Optional[datetime] = None
    risk_factors: List[str] = Field(default=[])


class RealTimeAlert(BaseModel):
    alert_id: str
    transaction_id: int
    user_id: str
    alert_type: str
    severity: AlertSeverity
    message: str
    fraud_score: float
    amount: float
    merchant_id: str
    timestamp: datetime
    requires_action: bool = False


# WebSocket Models
class WSMessage(BaseModel):
    type: str  # alert, stats, heartbeat
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)


class WSAlert(BaseModel):
    type: str = "alert"
    alert: RealTimeAlert


class WSStats(BaseModel):
    type: str = "stats"
    stats: AnalyticsStats


# Health Check
class HealthCheck(BaseModel):
    status: str = "healthy"
    timestamp: datetime = Field(default_factory=datetime.now)
    version: str
    database_status: str
    redis_status: str
    model_status: str
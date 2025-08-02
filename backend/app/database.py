from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import func
from datetime import datetime
from typing import Generator
import redis

from .config import settings

# SQLAlchemy setup
engine = create_engine(
    settings.database_url,
    echo=settings.database_echo,
    connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Redis setup
redis_client = redis.Redis(
    host=settings.redis_host,
    port=settings.redis_port,
    db=settings.redis_db,
    password=settings.redis_password,
    decode_responses=True
)


class Transaction(Base):
    __tablename__ = "transactions"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String(255), index=True)
    merchant_id = Column(String(255), index=True)
    amount = Column(Float, nullable=False)
    currency = Column(String(10), default="USD")
    transaction_type = Column(String(50), nullable=False)
    timestamp = Column(DateTime, default=func.now())
    
    # Location data
    latitude = Column(Float)
    longitude = Column(Float)
    country = Column(String(100))
    city = Column(String(100))
    
    # Device and session info
    device_id = Column(String(255))
    ip_address = Column(String(45))
    user_agent = Column(Text)
    session_id = Column(String(255))
    
    # Transaction details
    description = Column(Text)
    merchant_category = Column(String(100))
    payment_method = Column(String(50))
    
    # Fraud detection results
    is_fraud = Column(Boolean, default=False)
    fraud_score = Column(Float, default=0.0)
    fraud_reasons = Column(JSON)
    model_version = Column(String(50))
    
    # Processing metadata
    processed_at = Column(DateTime, default=func.now())
    processing_time_ms = Column(Float)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class FraudAlert(Base):
    __tablename__ = "fraud_alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    transaction_id = Column(Integer, index=True)
    alert_type = Column(String(50), nullable=False)
    severity = Column(String(20), default="medium")  # low, medium, high, critical
    message = Column(Text, nullable=False)
    details = Column(JSON)
    
    # Status tracking
    status = Column(String(20), default="active")  # active, investigated, resolved, false_positive
    assigned_to = Column(String(255))
    resolution_notes = Column(Text)
    
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    resolved_at = Column(DateTime)


class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    
    # Performance metrics
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    auc_roc = Column(Float)
    
    # Training data info
    training_samples = Column(Integer)
    fraud_samples = Column(Integer)
    legitimate_samples = Column(Integer)
    
    # Deployment info
    deployed_at = Column(DateTime, default=func.now())
    is_active = Column(Boolean, default=True)
    
    created_at = Column(DateTime, default=func.now())


# Database dependency
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Redis utilities
class RedisCache:
    @staticmethod
    def set_transaction_score(transaction_id: str, score: float, ttl: int = 3600):
        """Cache transaction fraud score"""
        redis_client.setex(f"fraud_score:{transaction_id}", ttl, str(score))
    
    @staticmethod
    def get_transaction_score(transaction_id: str) -> float:
        """Get cached transaction fraud score"""
        score = redis_client.get(f"fraud_score:{transaction_id}")
        return float(score) if score else None
    
    @staticmethod
    def set_user_stats(user_id: str, stats: dict, ttl: int = 1800):
        """Cache user transaction statistics"""
        redis_client.setex(f"user_stats:{user_id}", ttl, str(stats))
    
    @staticmethod
    def get_user_stats(user_id: str) -> dict:
        """Get cached user statistics"""
        stats = redis_client.get(f"user_stats:{user_id}")
        return eval(stats) if stats else None
    
    @staticmethod
    def increment_daily_transactions(date_str: str) -> int:
        """Increment daily transaction counter"""
        key = f"daily_transactions:{date_str}"
        return redis_client.incr(key)
    
    @staticmethod
    def increment_daily_fraud(date_str: str) -> int:
        """Increment daily fraud counter"""
        key = f"daily_fraud:{date_str}"
        return redis_client.incr(key)


# Create tables
def create_tables():
    Base.metadata.create_all(bind=engine)
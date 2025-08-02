from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import pandas as pd
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
import uvicorn
from contextlib import asynccontextmanager

from app.config import settings
from app.database import get_db, create_tables, Transaction, FraudAlert, ModelMetrics, RedisCache
from app.models import (
    TransactionRequest, BulkTransactionRequest, TransactionResponse,
    FraudDetectionResult, AnalyticsStats, FraudAlertResponse,
    ModelMetricsResponse, UserRiskProfile, HealthCheck, WSMessage, RealTimeAlert
)
from app.ml_engine import fraud_detection_model
from app.services import TransactionService, AnalyticsService, AlertService, WebSocketManager


# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Starting VigilantBytes AI...")
    create_tables()
    
    # Try to load existing models
    if not fraud_detection_model.load_models("data/models"):
        print("⚠️ No pre-trained models found. Will train on first transactions.")
    else:
        print("✅ Loaded pre-trained fraud detection models")
    
    yield
    
    # Shutdown
    print("🛑 Shutting down VigilantBytes AI...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Real-time fraud detection system using machine learning",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
transaction_service = TransactionService()
analytics_service = AnalyticsService()
alert_service = AlertService()
websocket_manager = WebSocketManager()


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check(db: Session = Depends(get_db)):
    """Health check endpoint"""
    try:
        # Check database
        db.execute("SELECT 1")
        db_status = "healthy"
    except Exception:
        db_status = "unhealthy"
    
    try:
        # Check Redis
        RedisCache.increment_daily_transactions("health_check")
        redis_status = "healthy"
    except Exception:
        redis_status = "unhealthy"
    
    model_status = "healthy" if fraud_detection_model.is_trained else "not_trained"
    
    return HealthCheck(
        version=settings.app_version,
        database_status=db_status,
        redis_status=redis_status,
        model_status=model_status
    )


# Transaction endpoints
@app.post("/api/transactions/check", response_model=FraudDetectionResult)
async def check_transaction(
    transaction: TransactionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Check a single transaction for fraud"""
    try:
        # Get historical data for the user
        historical_data = transaction_service.get_user_history(transaction.user_id, db)
        
        # If model is not trained and we have enough data, train it
        if not fraud_detection_model.is_trained:
            all_transactions = transaction_service.get_training_data(db)
            if len(all_transactions) >= 100:  # Minimum training samples
                background_tasks.add_task(train_models, all_transactions)
                # For now, return a simple rule-based result
                return simple_fraud_check(transaction)
        
        # Predict fraud
        result = fraud_detection_model.predict(transaction, historical_data)
        
        # Store transaction in database
        background_tasks.add_task(
            store_transaction_result, 
            transaction, result, db
        )
        
        # Send real-time alert if fraud detected
        if result.is_fraud:
            background_tasks.add_task(
                send_fraud_alert,
                transaction, result
            )
        
        # Update daily counters
        today = datetime.now().strftime("%Y-%m-%d")
        RedisCache.increment_daily_transactions(today)
        if result.is_fraud:
            RedisCache.increment_daily_fraud(today)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transaction: {str(e)}")


@app.post("/api/transactions/bulk", response_model=List[FraudDetectionResult])
async def check_bulk_transactions(
    bulk_request: BulkTransactionRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Check multiple transactions for fraud"""
    results = []
    
    for transaction in bulk_request.transactions:
        try:
            historical_data = transaction_service.get_user_history(transaction.user_id, db)
            
            if fraud_detection_model.is_trained:
                result = fraud_detection_model.predict(transaction, historical_data)
            else:
                result = simple_fraud_check(transaction)
            
            results.append(result)
            
            # Store in background
            background_tasks.add_task(
                store_transaction_result,
                transaction, result, db
            )
            
        except Exception as e:
            # Continue with other transactions if one fails
            results.append(FraudDetectionResult(
                is_fraud=False,
                fraud_score=0.0,
                confidence=0.0,
                fraud_reasons=[f"Error: {str(e)}"],
                model_version="error",
                processing_time_ms=0.0
            ))
    
    return results


@app.get("/api/transactions/history", response_model=List[TransactionResponse])
async def get_transaction_history(
    user_id: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get transaction history"""
    query = db.query(Transaction)
    
    if user_id:
        query = query.filter(Transaction.user_id == user_id)
    
    transactions = query.order_by(desc(Transaction.timestamp)).offset(offset).limit(limit).all()
    
    return [TransactionResponse.from_orm(t) for t in transactions]


@app.get("/api/transactions/{transaction_id}", response_model=TransactionResponse)
async def get_transaction(transaction_id: int, db: Session = Depends(get_db)):
    """Get a specific transaction"""
    transaction = db.query(Transaction).filter(Transaction.id == transaction_id).first()
    
    if not transaction:
        raise HTTPException(status_code=404, detail="Transaction not found")
    
    return TransactionResponse.from_orm(transaction)


# Analytics endpoints
@app.get("/api/analytics/stats", response_model=AnalyticsStats)
async def get_analytics_stats(
    days: int = 7,
    db: Session = Depends(get_db)
):
    """Get fraud detection analytics"""
    return analytics_service.get_analytics_stats(days, db)


@app.get("/api/analytics/user-risk/{user_id}", response_model=UserRiskProfile)
async def get_user_risk_profile(user_id: str, db: Session = Depends(get_db)):
    """Get user risk profile"""
    return analytics_service.get_user_risk_profile(user_id, db)


# Alert endpoints
@app.get("/api/alerts", response_model=List[FraudAlertResponse])
async def get_alerts(
    status: Optional[str] = None,
    severity: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get fraud alerts"""
    return alert_service.get_alerts(status, severity, limit, db)


@app.put("/api/alerts/{alert_id}/status")
async def update_alert_status(
    alert_id: int,
    status: str,
    notes: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """Update alert status"""
    return alert_service.update_alert_status(alert_id, status, notes, db)


# Model endpoints
@app.get("/api/models/metrics", response_model=List[ModelMetricsResponse])
async def get_model_metrics(db: Session = Depends(get_db)):
    """Get model performance metrics"""
    metrics = db.query(ModelMetrics).filter(ModelMetrics.is_active == True).all()
    return [ModelMetricsResponse.from_orm(m) for m in metrics]


@app.post("/api/models/retrain")
async def retrain_models(background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    """Trigger model retraining"""
    training_data = transaction_service.get_training_data(db)
    
    if len(training_data) < 100:
        raise HTTPException(
            status_code=400, 
            detail="Insufficient training data. Need at least 100 transactions."
        )
    
    background_tasks.add_task(train_models, training_data)
    
    return {"message": "Model retraining started", "training_samples": len(training_data)}


# WebSocket endpoint for real-time updates
@app.websocket("/ws/alerts")
async def websocket_alerts(websocket: WebSocket):
    """WebSocket endpoint for real-time fraud alerts"""
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Send heartbeat every 30 seconds
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": datetime.now().isoformat()
            })
            await asyncio.sleep(30)
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


# Background tasks
async def store_transaction_result(
    transaction: TransactionRequest,
    result: FraudDetectionResult,
    db: Session
):
    """Store transaction and fraud detection result in database"""
    try:
        db_transaction = Transaction(
            user_id=transaction.user_id,
            merchant_id=transaction.merchant_id,
            amount=transaction.amount,
            currency=transaction.currency,
            transaction_type=transaction.transaction_type,
            latitude=transaction.latitude,
            longitude=transaction.longitude,
            country=transaction.country,
            city=transaction.city,
            device_id=transaction.device_id,
            ip_address=transaction.ip_address,
            user_agent=transaction.user_agent,
            session_id=transaction.session_id,
            description=transaction.description,
            merchant_category=transaction.merchant_category,
            payment_method=transaction.payment_method,
            is_fraud=result.is_fraud,
            fraud_score=result.fraud_score,
            fraud_reasons=result.fraud_reasons,
            model_version=result.model_version,
            processing_time_ms=result.processing_time_ms
        )
        
        db.add(db_transaction)
        db.commit()
        
    except Exception as e:
        print(f"Error storing transaction: {e}")
        db.rollback()


async def send_fraud_alert(transaction: TransactionRequest, result: FraudDetectionResult):
    """Send real-time fraud alert"""
    try:
        alert = RealTimeAlert(
            alert_id=f"alert_{datetime.now().timestamp()}",
            transaction_id=0,  # Will be updated when transaction is stored
            user_id=transaction.user_id,
            alert_type="fraud_detected",
            severity="high" if result.fraud_score > 0.8 else "medium",
            message=f"Potential fraud detected for user {transaction.user_id}",
            fraud_score=result.fraud_score,
            amount=transaction.amount,
            merchant_id=transaction.merchant_id,
            timestamp=datetime.now(),
            requires_action=result.fraud_score > 0.8
        )
        
        # Send to all connected WebSocket clients
        await websocket_manager.broadcast_alert(alert)
        
        # Store alert in database
        # This would be handled by AlertService
        
    except Exception as e:
        print(f"Error sending fraud alert: {e}")


async def train_models(training_data: pd.DataFrame):
    """Train fraud detection models in background"""
    try:
        print(f"🤖 Training models with {len(training_data)} samples...")
        
        # Train the ensemble model
        metrics = fraud_detection_model.train(training_data)
        
        # Save trained models
        fraud_detection_model.save_models("data/models")
        
        print(f"✅ Model training completed. Metrics: {metrics}")
        
        # Store metrics in database (would need database session)
        # This could be improved by passing db session to background task
        
    except Exception as e:
        print(f"❌ Error training models: {e}")


def simple_fraud_check(transaction: TransactionRequest) -> FraudDetectionResult:
    """Simple rule-based fraud check when ML models are not available"""
    fraud_score = 0.0
    fraud_reasons = []
    
    # High amount transactions
    if transaction.amount > 10000:
        fraud_score += 0.3
        fraud_reasons.append("High transaction amount")
    
    # Weekend transactions
    if datetime.now().weekday() >= 5:
        fraud_score += 0.1
        fraud_reasons.append("Weekend transaction")
    
    # Late night transactions
    if datetime.now().hour < 6 or datetime.now().hour > 22:
        fraud_score += 0.2
        fraud_reasons.append("Unusual transaction time")
    
    # Cryptocurrency transactions
    if transaction.payment_method == "cryptocurrency":
        fraud_score += 0.4
        fraud_reasons.append("Cryptocurrency payment")
    
    is_fraud = fraud_score > settings.fraud_threshold
    
    return FraudDetectionResult(
        is_fraud=is_fraud,
        fraud_score=min(fraud_score, 1.0),
        confidence=0.6,  # Lower confidence for rule-based
        fraud_reasons=fraud_reasons,
        model_version="rule_based_v1.0",
        processing_time_ms=1.0
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )
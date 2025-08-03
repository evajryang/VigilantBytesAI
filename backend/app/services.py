from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_
import pandas as pd
from datetime import datetime, timedelta
from fastapi import WebSocket
import json

from .database import Transaction, FraudAlert, ModelMetrics, RedisCache
from .models import (
    AnalyticsStats, UserRiskProfile, FraudAlertResponse,
    RealTimeAlert, AlertSeverity, AlertStatus
)


class TransactionService:
    """Service for transaction-related operations"""
    
    def get_user_history(self, user_id: str, db: Session, days: int = 30) -> pd.DataFrame:
        """Get user's transaction history as DataFrame"""
        since_date = datetime.now() - timedelta(days=days)
        
        transactions = db.query(Transaction).filter(
            and_(
                Transaction.user_id == user_id,
                Transaction.timestamp >= since_date
            )
        ).all()
        
        if not transactions:
            return pd.DataFrame()
        
        # Convert to DataFrame
        data = []
        for t in transactions:
            data.append({
                'user_id': t.user_id,
                'merchant_id': t.merchant_id,
                'amount': t.amount,
                'currency': t.currency,
                'transaction_type': t.transaction_type,
                'timestamp': t.timestamp,
                'latitude': t.latitude,
                'longitude': t.longitude,
                'country': t.country,
                'city': t.city,
                'device_id': t.device_id,
                'ip_address': t.ip_address,
                'payment_method': t.payment_method,
                'is_fraud': t.is_fraud,
                'fraud_score': t.fraud_score
            })
        
        return pd.DataFrame(data)
    
    def get_training_data(self, db: Session, limit: int = 10000) -> pd.DataFrame:
        """Get transaction data for model training"""
        transactions = db.query(Transaction).order_by(
            desc(Transaction.timestamp)
        ).limit(limit).all()
        
        if not transactions:
            return pd.DataFrame()
        
        # Convert to DataFrame for ML training
        data = []
        for t in transactions:
            data.append({
                'user_id': t.user_id,
                'merchant_id': t.merchant_id,
                'amount': t.amount,
                'currency': t.currency,
                'transaction_type': t.transaction_type,
                'timestamp': t.timestamp,
                'latitude': t.latitude,
                'longitude': t.longitude,
                'country': t.country,
                'city': t.city,
                'device_id': t.device_id,
                'ip_address': t.ip_address,
                'payment_method': t.payment_method,
                'merchant_category': t.merchant_category,
                'is_fraud': t.is_fraud,
                'fraud_score': t.fraud_score
            })
        
        return pd.DataFrame(data)


class AnalyticsService:
    """Service for analytics and reporting"""
    
    def get_analytics_stats(self, days: int, db: Session) -> AnalyticsStats:
        """Get comprehensive analytics statistics"""
        since_date = datetime.now() - timedelta(days=days)
        
        # Basic transaction stats
        total_transactions = db.query(func.count(Transaction.id)).filter(
            Transaction.timestamp >= since_date
        ).scalar() or 0
        
        fraud_transactions = db.query(func.count(Transaction.id)).filter(
            and_(
                Transaction.timestamp >= since_date,
                Transaction.is_fraud == True
            )
        ).scalar() or 0
        
        fraud_rate = fraud_transactions / total_transactions if total_transactions > 0 else 0.0
        
        # Volume stats
        total_volume = db.query(func.sum(Transaction.amount)).filter(
            Transaction.timestamp >= since_date
        ).scalar() or 0.0
        
        fraud_volume = db.query(func.sum(Transaction.amount)).filter(
            and_(
                Transaction.timestamp >= since_date,
                Transaction.is_fraud == True
            )
        ).scalar() or 0.0
        
        avg_transaction_amount = db.query(func.avg(Transaction.amount)).filter(
            Transaction.timestamp >= since_date
        ).scalar() or 0.0
        
        avg_fraud_amount = db.query(func.avg(Transaction.amount)).filter(
            and_(
                Transaction.timestamp >= since_date,
                Transaction.is_fraud == True
            )
        ).scalar() or 0.0
        
        # Top fraud reasons
        fraud_reasons_query = db.query(Transaction.fraud_reasons).filter(
            and_(
                Transaction.timestamp >= since_date,
                Transaction.is_fraud == True,
                Transaction.fraud_reasons.isnot(None)
            )
        ).all()
        
        reason_counts = {}
        for reasons_row in fraud_reasons_query:
            if reasons_row[0]:
                for reason in reasons_row[0]:
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        top_fraud_reasons = [
            {"reason": reason, "count": count}
            for reason, count in sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        ]
        
        # Hourly stats for the last 24 hours
        hourly_stats = self._get_hourly_stats(db)
        
        # Daily stats
        daily_stats = self._get_daily_stats(days, db)
        
        return AnalyticsStats(
            total_transactions=total_transactions,
            fraud_transactions=fraud_transactions,
            fraud_rate=fraud_rate,
            total_volume=total_volume,
            fraud_volume=fraud_volume,
            avg_transaction_amount=avg_transaction_amount,
            avg_fraud_amount=avg_fraud_amount,
            top_fraud_reasons=top_fraud_reasons,
            hourly_stats=hourly_stats,
            daily_stats=daily_stats
        )
    
    def _get_hourly_stats(self, db: Session) -> List[Dict[str, Any]]:
        """Get hourly transaction statistics for the last 24 hours"""
        since_date = datetime.now() - timedelta(hours=24)
        
        # Group by hour
        hourly_data = db.query(
            func.extract('hour', Transaction.timestamp).label('hour'),
            func.count(Transaction.id).label('total'),
            func.sum(func.case([(Transaction.is_fraud == True, 1)], else_=0)).label('fraud'),
            func.sum(Transaction.amount).label('volume')
        ).filter(
            Transaction.timestamp >= since_date
        ).group_by(
            func.extract('hour', Transaction.timestamp)
        ).all()
        
        stats = []
        for hour_data in hourly_data:
            stats.append({
                "hour": int(hour_data.hour),
                "total_transactions": hour_data.total,
                "fraud_transactions": hour_data.fraud,
                "fraud_rate": hour_data.fraud / hour_data.total if hour_data.total > 0 else 0,
                "volume": float(hour_data.volume or 0)
            })
        
        return stats
    
    def _get_daily_stats(self, days: int, db: Session) -> List[Dict[str, Any]]:
        """Get daily transaction statistics"""
        since_date = datetime.now() - timedelta(days=days)
        
        daily_data = db.query(
            func.date(Transaction.timestamp).label('date'),
            func.count(Transaction.id).label('total'),
            func.sum(func.case([(Transaction.is_fraud == True, 1)], else_=0)).label('fraud'),
            func.sum(Transaction.amount).label('volume')
        ).filter(
            Transaction.timestamp >= since_date
        ).group_by(
            func.date(Transaction.timestamp)
        ).order_by(
            func.date(Transaction.timestamp)
        ).all()
        
        stats = []
        for day_data in daily_data:
            stats.append({
                "date": day_data.date.isoformat() if day_data.date else None,
                "total_transactions": day_data.total,
                "fraud_transactions": day_data.fraud,
                "fraud_rate": day_data.fraud / day_data.total if day_data.total > 0 else 0,
                "volume": float(day_data.volume or 0)
            })
        
        return stats
    
    def get_user_risk_profile(self, user_id: str, db: Session) -> UserRiskProfile:
        """Get comprehensive user risk profile"""
        # Get user's transaction history
        user_transactions = db.query(Transaction).filter(
            Transaction.user_id == user_id
        ).all()
        
        if not user_transactions:
            return UserRiskProfile(
                user_id=user_id,
                risk_score=0.5,  # Neutral risk for new users
                transaction_count=0,
                avg_amount=0.0,
                countries_count=0,
                merchants_count=0,
                fraud_count=0,
                risk_factors=["New user - limited history"]
            )
        
        # Calculate risk metrics
        transaction_count = len(user_transactions)
        avg_amount = sum(t.amount for t in user_transactions) / transaction_count
        countries = set(t.country for t in user_transactions if t.country)
        merchants = set(t.merchant_id for t in user_transactions)
        fraud_count = sum(1 for t in user_transactions if t.is_fraud)
        last_transaction = max(t.timestamp for t in user_transactions)
        
        # Calculate risk score
        risk_score = 0.0
        risk_factors = []
        
        # Fraud history factor
        fraud_rate = fraud_count / transaction_count
        if fraud_rate > 0.1:
            risk_score += 0.4
            risk_factors.append(f"High fraud rate: {fraud_rate:.1%}")
        elif fraud_rate > 0.05:
            risk_score += 0.2
            risk_factors.append(f"Elevated fraud rate: {fraud_rate:.1%}")
        
        # High average amount
        if avg_amount > 5000:
            risk_score += 0.2
            risk_factors.append("High average transaction amount")
        
        # Multiple countries
        if len(countries) > 5:
            risk_score += 0.2
            risk_factors.append("Transactions from many countries")
        
        # High transaction frequency
        if transaction_count > 100:
            recent_transactions = [
                t for t in user_transactions 
                if t.timestamp >= datetime.now() - timedelta(days=30)
            ]
            if len(recent_transactions) > 50:
                risk_score += 0.2
                risk_factors.append("High transaction frequency")
        
        # Inactive user
        if (datetime.now() - last_transaction).days > 90:
            risk_score += 0.1
            risk_factors.append("Long period of inactivity")
        
        # New user with high activity
        if transaction_count < 10 and avg_amount > 1000:
            risk_score += 0.3
            risk_factors.append("New user with high-value transactions")
        
        risk_score = min(risk_score, 1.0)  # Cap at 1.0
        
        if not risk_factors:
            risk_factors = ["Normal transaction patterns"]
        
        return UserRiskProfile(
            user_id=user_id,
            risk_score=risk_score,
            transaction_count=transaction_count,
            avg_amount=avg_amount,
            countries_count=len(countries),
            merchants_count=len(merchants),
            fraud_count=fraud_count,
            last_transaction=last_transaction,
            risk_factors=risk_factors
        )


class AlertService:
    """Service for fraud alert management"""
    
    def get_alerts(
        self, 
        status: Optional[str], 
        severity: Optional[str], 
        limit: int, 
        db: Session
    ) -> List[FraudAlertResponse]:
        """Get fraud alerts with optional filtering"""
        query = db.query(FraudAlert)
        
        if status:
            query = query.filter(FraudAlert.status == status)
        
        if severity:
            query = query.filter(FraudAlert.severity == severity)
        
        alerts = query.order_by(desc(FraudAlert.created_at)).limit(limit).all()
        
        return [FraudAlertResponse.from_orm(alert) for alert in alerts]
    
    def update_alert_status(
        self, 
        alert_id: int, 
        status: str, 
        notes: Optional[str], 
        db: Session
    ) -> Dict[str, Any]:
        """Update alert status and resolution notes"""
        alert = db.query(FraudAlert).filter(FraudAlert.id == alert_id).first()
        
        if not alert:
            raise ValueError("Alert not found")
        
        alert.status = status
        if notes:
            alert.resolution_notes = notes
        
        if status in ["resolved", "false_positive"]:
            alert.resolved_at = datetime.now()
        
        db.commit()
        
        return {
            "alert_id": alert_id,
            "status": status,
            "updated_at": datetime.now().isoformat()
        }
    
    def create_alert(
        self, 
        transaction_id: int, 
        alert_type: str, 
        severity: str, 
        message: str, 
        details: Optional[Dict[str, Any]], 
        db: Session
    ) -> FraudAlert:
        """Create a new fraud alert"""
        alert = FraudAlert(
            transaction_id=transaction_id,
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
            status="active"
        )
        
        db.add(alert)
        db.commit()
        db.refresh(alert)
        
        return alert


class WebSocketManager:
    """Manager for WebSocket connections and real-time alerts"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"📡 New WebSocket connection. Total: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            print(f"📡 WebSocket disconnected. Total: {len(self.active_connections)}")
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        """Send a message to a specific WebSocket"""
        try:
            await websocket.send_text(message)
        except Exception as e:
            print(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def broadcast(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSockets"""
        if not self.active_connections:
            return
        
        message_str = json.dumps(message, default=str)
        disconnected = []
        
        for connection in self.active_connections:
            try:
                await connection.send_text(message_str)
            except Exception as e:
                print(f"Error broadcasting to WebSocket: {e}")
                disconnected.append(connection)
        
        # Remove disconnected connections
        for connection in disconnected:
            self.disconnect(connection)
    
    async def broadcast_alert(self, alert: RealTimeAlert):
        """Broadcast a fraud alert to all connected clients"""
        message = {
            "type": "fraud_alert",
            "data": {
                "alert_id": alert.alert_id,
                "transaction_id": alert.transaction_id,
                "user_id": alert.user_id,
                "alert_type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "fraud_score": alert.fraud_score,
                "amount": alert.amount,
                "merchant_id": alert.merchant_id,
                "timestamp": alert.timestamp.isoformat(),
                "requires_action": alert.requires_action
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message)
        print(f"🚨 Fraud alert broadcasted: {alert.alert_id}")
    
    async def broadcast_stats_update(self, stats: AnalyticsStats):
        """Broadcast updated statistics to all connected clients"""
        message = {
            "type": "stats_update",
            "data": {
                "total_transactions": stats.total_transactions,
                "fraud_transactions": stats.fraud_transactions,
                "fraud_rate": stats.fraud_rate,
                "total_volume": stats.total_volume,
                "fraud_volume": stats.fraud_volume
            },
            "timestamp": datetime.now().isoformat()
        }
        
        await self.broadcast(message)


class ReportService:
    """Service for generating reports and exports"""
    
    def generate_fraud_report(
        self, 
        start_date: datetime, 
        end_date: datetime, 
        db: Session
    ) -> Dict[str, Any]:
        """Generate comprehensive fraud report for date range"""
        
        # Get transactions in date range
        transactions = db.query(Transaction).filter(
            and_(
                Transaction.timestamp >= start_date,
                Transaction.timestamp <= end_date
            )
        ).all()
        
        if not transactions:
            return {
                "period": f"{start_date.date()} to {end_date.date()}",
                "summary": "No transactions in this period",
                "total_transactions": 0,
                "fraud_transactions": 0
            }
        
        # Calculate metrics
        total_transactions = len(transactions)
        fraud_transactions = sum(1 for t in transactions if t.is_fraud)
        fraud_rate = fraud_transactions / total_transactions
        
        total_volume = sum(t.amount for t in transactions)
        fraud_volume = sum(t.amount for t in transactions if t.is_fraud)
        
        # Top fraud patterns
        fraud_reasons = {}
        for t in transactions:
            if t.is_fraud and t.fraud_reasons:
                for reason in t.fraud_reasons:
                    fraud_reasons[reason] = fraud_reasons.get(reason, 0) + 1
        
        # Risk by merchant
        merchant_risks = {}
        for t in transactions:
            merchant_id = t.merchant_id
            if merchant_id not in merchant_risks:
                merchant_risks[merchant_id] = {"total": 0, "fraud": 0}
            
            merchant_risks[merchant_id]["total"] += 1
            if t.is_fraud:
                merchant_risks[merchant_id]["fraud"] += 1
        
        # Calculate merchant risk scores
        for merchant_id in merchant_risks:
            data = merchant_risks[merchant_id]
            data["fraud_rate"] = data["fraud"] / data["total"] if data["total"] > 0 else 0
        
        # Sort merchants by fraud rate
        high_risk_merchants = sorted(
            [(mid, data) for mid, data in merchant_risks.items() if data["total"] >= 5],
            key=lambda x: x[1]["fraud_rate"],
            reverse=True
        )[:10]
        
        return {
            "period": f"{start_date.date()} to {end_date.date()}",
            "summary": {
                "total_transactions": total_transactions,
                "fraud_transactions": fraud_transactions,
                "fraud_rate": fraud_rate,
                "total_volume": total_volume,
                "fraud_volume": fraud_volume,
                "avg_fraud_amount": fraud_volume / fraud_transactions if fraud_transactions > 0 else 0
            },
            "top_fraud_reasons": [
                {"reason": reason, "count": count}
                for reason, count in sorted(fraud_reasons.items(), key=lambda x: x[1], reverse=True)[:10]
            ],
            "high_risk_merchants": [
                {
                    "merchant_id": merchant_id,
                    "total_transactions": data["total"],
                    "fraud_transactions": data["fraud"],
                    "fraud_rate": data["fraud_rate"]
                }
                for merchant_id, data in high_risk_merchants
            ],
            "generated_at": datetime.now().isoformat()
        }
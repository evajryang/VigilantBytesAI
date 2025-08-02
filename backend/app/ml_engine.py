import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow import keras
import joblib
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

from .config import settings
from .database import Session, Transaction, ModelMetrics
from .models import TransactionRequest, FraudDetectionResult


class FeatureExtractor:
    """Extract and engineer features from transaction data"""
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.user_stats = {}
        
    def extract_features(self, transaction: TransactionRequest, historical_data: pd.DataFrame = None) -> Dict[str, float]:
        """Extract features from a single transaction"""
        features = {}
        
        # Basic transaction features
        features['amount'] = float(transaction.amount)
        features['hour'] = datetime.now().hour
        features['day_of_week'] = datetime.now().weekday()
        features['is_weekend'] = 1.0 if datetime.now().weekday() >= 5 else 0.0
        
        # Transaction type encoding
        transaction_types = {'purchase': 1, 'withdrawal': 2, 'transfer': 3, 'deposit': 4, 'refund': 5}
        features['transaction_type'] = transaction_types.get(transaction.transaction_type, 0)
        
        # Payment method encoding
        payment_methods = {'credit_card': 1, 'debit_card': 2, 'bank_transfer': 3, 'digital_wallet': 4, 'cryptocurrency': 5, 'cash': 6}
        features['payment_method'] = payment_methods.get(transaction.payment_method, 0) if transaction.payment_method else 0
        
        # Location features
        features['has_location'] = 1.0 if transaction.latitude and transaction.longitude else 0.0
        features['latitude'] = transaction.latitude or 0.0
        features['longitude'] = transaction.longitude or 0.0
        
        # Device features
        features['has_device_id'] = 1.0 if transaction.device_id else 0.0
        features['has_session_id'] = 1.0 if transaction.session_id else 0.0
        
        # User behavioral features (if historical data available)
        if historical_data is not None and not historical_data.empty:
            user_data = historical_data[historical_data['user_id'] == transaction.user_id]
            if not user_data.empty:
                # User transaction patterns
                features['user_avg_amount'] = user_data['amount'].mean()
                features['user_std_amount'] = user_data['amount'].std() or 0.0
                features['user_transaction_count'] = len(user_data)
                features['user_merchant_count'] = user_data['merchant_id'].nunique()
                features['amount_zscore'] = (transaction.amount - features['user_avg_amount']) / (features['user_std_amount'] + 1e-6)
                
                # Time-based features
                recent_transactions = user_data[user_data['timestamp'] >= datetime.now() - timedelta(hours=24)]
                features['transactions_last_24h'] = len(recent_transactions)
                features['volume_last_24h'] = recent_transactions['amount'].sum() if not recent_transactions.empty else 0.0
                
                # Merchant patterns
                features['new_merchant'] = 1.0 if transaction.merchant_id not in user_data['merchant_id'].values else 0.0
                
                # Location patterns
                if transaction.country:
                    features['new_country'] = 1.0 if transaction.country not in user_data['country'].values else 0.0
                else:
                    features['new_country'] = 0.0
            else:
                # First-time user
                features.update({
                    'user_avg_amount': transaction.amount,
                    'user_std_amount': 0.0,
                    'user_transaction_count': 1,
                    'user_merchant_count': 1,
                    'amount_zscore': 0.0,
                    'transactions_last_24h': 1,
                    'volume_last_24h': transaction.amount,
                    'new_merchant': 1.0,
                    'new_country': 1.0
                })
        else:
            # No historical data available
            features.update({
                'user_avg_amount': transaction.amount,
                'user_std_amount': 0.0,
                'user_transaction_count': 1,
                'user_merchant_count': 1,
                'amount_zscore': 0.0,
                'transactions_last_24h': 1,
                'volume_last_24h': transaction.amount,
                'new_merchant': 1.0,
                'new_country': 1.0
            })
        
        return features
    
    def prepare_dataset(self, transactions_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare dataset for training"""
        features_list = []
        
        for _, transaction in transactions_df.iterrows():
            # Convert row to TransactionRequest-like object
            features = self.extract_features_from_row(transaction, transactions_df)
            features_list.append(list(features.values()))
        
        X = np.array(features_list)
        y = transactions_df['is_fraud'].values
        
        return X, y
    
    def extract_features_from_row(self, row: pd.Series, full_df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from a DataFrame row"""
        features = {}
        
        # Basic features
        features['amount'] = float(row['amount'])
        features['hour'] = row['timestamp'].hour if hasattr(row['timestamp'], 'hour') else 12
        features['day_of_week'] = row['timestamp'].weekday() if hasattr(row['timestamp'], 'weekday') else 1
        features['is_weekend'] = 1.0 if features['day_of_week'] >= 5 else 0.0
        
        # Transaction type encoding
        transaction_types = {'purchase': 1, 'withdrawal': 2, 'transfer': 3, 'deposit': 4, 'refund': 5}
        features['transaction_type'] = transaction_types.get(row['transaction_type'], 0)
        
        # Payment method encoding
        payment_methods = {'credit_card': 1, 'debit_card': 2, 'bank_transfer': 3, 'digital_wallet': 4, 'cryptocurrency': 5, 'cash': 6}
        features['payment_method'] = payment_methods.get(row.get('payment_method'), 0)
        
        # Location features
        features['has_location'] = 1.0 if pd.notna(row.get('latitude')) and pd.notna(row.get('longitude')) else 0.0
        features['latitude'] = row.get('latitude', 0.0) if pd.notna(row.get('latitude')) else 0.0
        features['longitude'] = row.get('longitude', 0.0) if pd.notna(row.get('longitude')) else 0.0
        
        # Device features
        features['has_device_id'] = 1.0 if pd.notna(row.get('device_id')) else 0.0
        features['has_session_id'] = 1.0 if pd.notna(row.get('session_id')) else 0.0
        
        # User behavioral features
        user_data = full_df[full_df['user_id'] == row['user_id']]
        user_data = user_data[user_data.index < row.name]  # Only past transactions
        
        if not user_data.empty:
            features['user_avg_amount'] = user_data['amount'].mean()
            features['user_std_amount'] = user_data['amount'].std() or 0.0
            features['user_transaction_count'] = len(user_data)
            features['user_merchant_count'] = user_data['merchant_id'].nunique()
            features['amount_zscore'] = (row['amount'] - features['user_avg_amount']) / (features['user_std_amount'] + 1e-6)
            
            # Time-based features
            recent_transactions = user_data[user_data['timestamp'] >= row['timestamp'] - timedelta(hours=24)]
            features['transactions_last_24h'] = len(recent_transactions)
            features['volume_last_24h'] = recent_transactions['amount'].sum() if not recent_transactions.empty else 0.0
            
            # Merchant and location patterns
            features['new_merchant'] = 1.0 if row['merchant_id'] not in user_data['merchant_id'].values else 0.0
            features['new_country'] = 1.0 if row.get('country') and row['country'] not in user_data['country'].dropna().values else 0.0
        else:
            # First transaction for user
            features.update({
                'user_avg_amount': row['amount'],
                'user_std_amount': 0.0,
                'user_transaction_count': 1,
                'user_merchant_count': 1,
                'amount_zscore': 0.0,
                'transactions_last_24h': 1,
                'volume_last_24h': row['amount'],
                'new_merchant': 1.0,
                'new_country': 1.0
            })
        
        return features


class IsolationForestModel:
    """Isolation Forest model for anomaly detection"""
    
    def __init__(self):
        self.model = IsolationForest(
            contamination=0.1,  # Expected fraud rate
            random_state=42,
            n_estimators=100
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
    
    def train(self, X: np.ndarray, y: np.ndarray = None) -> Dict[str, float]:
        """Train the Isolation Forest model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model (unsupervised)
        self.model.fit(X_scaled)
        self.is_trained = True
        
        # Evaluate on training data
        predictions = self.model.predict(X_scaled)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert to binary predictions (1 = normal, -1 = anomaly)
        fraud_predictions = (predictions == -1).astype(int)
        
        metrics = {}
        if y is not None:
            from sklearn.metrics import precision_score, recall_score, f1_score
            metrics['precision'] = precision_score(y, fraud_predictions)
            metrics['recall'] = recall_score(y, fraud_predictions)
            metrics['f1_score'] = f1_score(y, fraud_predictions)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fraud probability"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly scores (more negative = more anomalous)
        anomaly_scores = self.model.decision_function(X_scaled)
        
        # Convert scores to probabilities (0-1 range)
        # Normalize scores to 0-1 where higher = more likely fraud
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        probabilities = 1 - (anomaly_scores - min_score) / (max_score - min_score + 1e-8)
        
        # Binary predictions
        predictions = (anomaly_scores < 0).astype(int)
        
        return predictions, probabilities


class NeuralNetworkModel:
    """Neural Network model for fraud detection"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.input_dim = None
    
    def build_model(self, input_dim: int) -> keras.Model:
        """Build the neural network architecture"""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def train(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Train the neural network model"""
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Build model
        self.input_dim = X_scaled.shape[1]
        self.model = self.build_model(self.input_dim)
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=50,
            batch_size=128,
            verbose=0,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )
        
        self.is_trained = True
        
        # Evaluate model
        val_loss, val_accuracy, val_precision, val_recall = self.model.evaluate(X_val, y_val, verbose=0)
        f1_score = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-8)
        
        return {
            'accuracy': val_accuracy,
            'precision': val_precision,
            'recall': val_recall,
            'f1_score': f1_score,
            'loss': val_loss
        }
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict fraud probability"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled, verbose=0).flatten()
        predictions = (probabilities > 0.5).astype(int)
        
        return predictions, probabilities


class EnsembleModel:
    """Ensemble of multiple fraud detection models"""
    
    def __init__(self):
        self.isolation_forest = IsolationForestModel()
        self.neural_network = NeuralNetworkModel()
        self.feature_extractor = FeatureExtractor()
        self.is_trained = False
        self.weights = {'isolation_forest': 0.3, 'neural_network': 0.7}
    
    def train(self, transactions_df: pd.DataFrame) -> Dict[str, Any]:
        """Train all models in the ensemble"""
        # Extract features
        X, y = self.feature_extractor.prepare_dataset(transactions_df)
        
        # Train individual models
        if_metrics = self.isolation_forest.train(X, y)
        nn_metrics = self.neural_network.train(X, y)
        
        self.is_trained = True
        
        return {
            'isolation_forest': if_metrics,
            'neural_network': nn_metrics,
            'training_samples': len(transactions_df),
            'fraud_samples': y.sum(),
            'legitimate_samples': len(y) - y.sum()
        }
    
    def predict(self, transaction: TransactionRequest, historical_data: pd.DataFrame = None) -> FraudDetectionResult:
        """Predict fraud for a single transaction"""
        if not self.is_trained:
            raise ValueError("Models must be trained before prediction")
        
        start_time = datetime.now()
        
        # Extract features
        features = self.feature_extractor.extract_features(transaction, historical_data)
        X = np.array([list(features.values())])
        
        # Get predictions from individual models
        if_pred, if_prob = self.isolation_forest.predict(X)
        nn_pred, nn_prob = self.neural_network.predict(X)
        
        # Ensemble prediction (weighted average)
        ensemble_prob = (
            self.weights['isolation_forest'] * if_prob[0] + 
            self.weights['neural_network'] * nn_prob[0]
        )
        
        # Final prediction
        is_fraud = ensemble_prob > settings.fraud_threshold
        
        # Generate fraud reasons
        fraud_reasons = []
        if is_fraud:
            if ensemble_prob > 0.8:
                fraud_reasons.append("High fraud probability")
            if features.get('amount_zscore', 0) > 3:
                fraud_reasons.append("Unusually high transaction amount")
            if features.get('new_merchant', 0) == 1:
                fraud_reasons.append("Transaction with new merchant")
            if features.get('new_country', 0) == 1:
                fraud_reasons.append("Transaction from new country")
            if features.get('transactions_last_24h', 0) > 10:
                fraud_reasons.append("High transaction frequency")
        
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return FraudDetectionResult(
            is_fraud=bool(is_fraud),
            fraud_score=float(ensemble_prob),
            confidence=float(max(ensemble_prob, 1 - ensemble_prob)),
            fraud_reasons=fraud_reasons,
            model_version="ensemble_v1.0",
            processing_time_ms=processing_time
        )
    
    def save_models(self, path: str):
        """Save trained models to disk"""
        os.makedirs(path, exist_ok=True)
        
        # Save Isolation Forest
        joblib.dump(self.isolation_forest, os.path.join(path, 'isolation_forest.pkl'))
        
        # Save Neural Network
        if self.neural_network.model:
            self.neural_network.model.save(os.path.join(path, 'neural_network.h5'))
            joblib.dump(self.neural_network.scaler, os.path.join(path, 'nn_scaler.pkl'))
        
        # Save feature extractor
        joblib.dump(self.feature_extractor, os.path.join(path, 'feature_extractor.pkl'))
    
    def load_models(self, path: str):
        """Load trained models from disk"""
        try:
            # Load Isolation Forest
            self.isolation_forest = joblib.load(os.path.join(path, 'isolation_forest.pkl'))
            
            # Load Neural Network
            self.neural_network.model = keras.models.load_model(os.path.join(path, 'neural_network.h5'))
            self.neural_network.scaler = joblib.load(os.path.join(path, 'nn_scaler.pkl'))
            self.neural_network.is_trained = True
            
            # Load feature extractor
            self.feature_extractor = joblib.load(os.path.join(path, 'feature_extractor.pkl'))
            
            self.is_trained = True
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False


# Global model instance
fraud_detection_model = EnsembleModel()
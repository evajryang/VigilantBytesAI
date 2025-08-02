#!/usr/bin/env python3
"""
Generate synthetic transaction data for training and testing
VigilantBytes AI fraud detection system
"""

import random
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
import os
from typing import List, Dict, Any

# Ensure we can import from the app
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.database import SessionLocal, Transaction, create_tables
from app.models import TransactionType, PaymentMethod


class SyntheticDataGenerator:
    """Generate realistic synthetic transaction data"""
    
    def __init__(self):
        self.merchants = [
            "Amazon", "Walmart", "Target", "Best Buy", "Home Depot",
            "McDonald's", "Starbucks", "Uber", "Netflix", "Apple",
            "Google", "Microsoft", "Tesla", "Nike", "Adidas",
            "Zara", "H&M", "Spotify", "PayPal", "Square",
            "Steam", "PlayStation", "Xbox", "Nintendo", "Epic Games"
        ]
        
        self.merchant_categories = [
            "retail", "grocery", "restaurant", "gas_station", "pharmacy",
            "electronics", "clothing", "entertainment", "transportation",
            "digital_services", "gaming", "travel", "hotel", "airline"
        ]
        
        self.countries = [
            "US", "CA", "GB", "DE", "FR", "IT", "ES", "NL", "BE", "CH",
            "AU", "NZ", "JP", "KR", "SG", "HK", "CN", "IN", "BR", "MX"
        ]
        
        self.cities = {
            "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
            "CA": ["Toronto", "Vancouver", "Montreal", "Calgary", "Ottawa"],
            "GB": ["London", "Manchester", "Birmingham", "Liverpool", "Leeds"],
            "DE": ["Berlin", "Munich", "Hamburg", "Frankfurt", "Cologne"],
            "FR": ["Paris", "Lyon", "Marseille", "Toulouse", "Nice"]
        }
        
        # User behavior patterns
        self.user_profiles = {
            "normal_user": {"fraud_rate": 0.01, "avg_amount": 150, "std_amount": 100},
            "high_value_user": {"fraud_rate": 0.02, "avg_amount": 2000, "std_amount": 1500},
            "frequent_user": {"fraud_rate": 0.015, "avg_amount": 75, "std_amount": 50},
            "suspicious_user": {"fraud_rate": 0.25, "avg_amount": 500, "std_amount": 400}
        }
    
    def generate_user_id(self) -> str:
        """Generate a realistic user ID"""
        return f"user_{random.randint(10000, 99999)}"
    
    def generate_device_id(self) -> str:
        """Generate a device ID"""
        return f"device_{random.randint(100000, 999999)}"
    
    def generate_ip_address(self) -> str:
        """Generate a realistic IP address"""
        return f"{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
    
    def generate_session_id(self) -> str:
        """Generate a session ID"""
        return f"session_{random.randint(1000000, 9999999)}"
    
    def get_user_profile_type(self, user_id: str) -> str:
        """Assign a profile type to a user based on user_id hash"""
        hash_val = hash(user_id) % 100
        if hash_val < 70:
            return "normal_user"
        elif hash_val < 85:
            return "high_value_user"
        elif hash_val < 95:
            return "frequent_user"
        else:
            return "suspicious_user"
    
    def generate_location(self) -> Dict[str, Any]:
        """Generate realistic location data"""
        country = random.choice(self.countries)
        city = random.choice(self.cities.get(country, ["Unknown City"]))
        
        # Generate realistic coordinates (rough approximation)
        lat_ranges = {
            "US": (25.0, 49.0), "CA": (42.0, 83.0), "GB": (50.0, 61.0),
            "DE": (47.0, 55.0), "FR": (41.0, 51.0)
        }
        lng_ranges = {
            "US": (-125.0, -66.0), "CA": (-140.0, -52.0), "GB": (-8.0, 2.0),
            "DE": (5.0, 15.0), "FR": (-5.0, 10.0)
        }
        
        lat_range = lat_ranges.get(country, (0.0, 0.0))
        lng_range = lng_ranges.get(country, (0.0, 0.0))
        
        return {
            "country": country,
            "city": city,
            "latitude": round(random.uniform(lat_range[0], lat_range[1]), 6),
            "longitude": round(random.uniform(lng_range[0], lng_range[1]), 6)
        }
    
    def generate_transaction_amount(self, profile_type: str, is_fraud: bool = False) -> float:
        """Generate realistic transaction amount based on user profile and fraud status"""
        profile = self.user_profiles[profile_type]
        
        if is_fraud:
            # Fraudulent transactions tend to be higher amounts
            base_amount = profile["avg_amount"] * random.uniform(2.0, 10.0)
            amount = max(10, np.random.normal(base_amount, profile["std_amount"] * 2))
        else:
            # Normal transactions
            amount = max(1, np.random.normal(profile["avg_amount"], profile["std_amount"]))
        
        return round(amount, 2)
    
    def generate_transaction_time(self, base_time: datetime, user_behavior: str) -> datetime:
        """Generate realistic transaction timestamp"""
        if user_behavior == "suspicious_user":
            # Fraudulent users more likely to transact at odd hours
            if random.random() < 0.3:
                hour = random.choice([2, 3, 4, 23, 0, 1])  # Late night/early morning
            else:
                hour = random.randint(6, 22)
        else:
            # Normal users mostly transact during business hours
            hour = random.choices(
                range(24),
                weights=[1, 1, 1, 1, 1, 2, 4, 6, 8, 10, 12, 12, 12, 12, 10, 8, 6, 4, 2, 1, 1, 1, 1, 1]
            )[0]
        
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        
        return base_time.replace(hour=hour, minute=minute, second=second)
    
    def should_be_fraud(self, profile_type: str, user_transactions: List[Dict]) -> bool:
        """Determine if this transaction should be fraudulent"""
        profile = self.user_profiles[profile_type]
        base_fraud_rate = profile["fraud_rate"]
        
        # Increase fraud probability for certain patterns
        recent_transactions = [t for t in user_transactions if 
                             (datetime.now() - t["timestamp"]).days < 1]
        
        if len(recent_transactions) > 10:  # High frequency
            base_fraud_rate *= 2
        
        if len(user_transactions) < 5:  # New user with limited history
            base_fraud_rate *= 1.5
        
        return random.random() < base_fraud_rate
    
    def generate_fraud_indicators(self, transaction: Dict[str, Any], user_transactions: List[Dict]) -> List[str]:
        """Generate realistic fraud indicators"""
        reasons = []
        
        # High amount
        if transaction["amount"] > 5000:
            reasons.append("High transaction amount")
        
        # Unusual time
        hour = transaction["timestamp"].hour
        if hour < 6 or hour > 22:
            reasons.append("Unusual transaction time")
        
        # New country for user
        user_countries = set(t.get("country") for t in user_transactions if t.get("country"))
        if transaction["country"] not in user_countries and len(user_countries) > 0:
            reasons.append("Transaction from new country")
        
        # High frequency
        recent_transactions = [t for t in user_transactions if 
                             (transaction["timestamp"] - t["timestamp"]).total_seconds() < 3600]
        if len(recent_transactions) > 5:
            reasons.append("High transaction frequency")
        
        # New merchant
        user_merchants = set(t.get("merchant_id") for t in user_transactions if t.get("merchant_id"))
        if transaction["merchant_id"] not in user_merchants and len(user_merchants) > 0:
            reasons.append("Transaction with new merchant")
        
        return reasons
    
    def generate_transactions(self, num_transactions: int = 10000, num_users: int = 1000) -> List[Dict[str, Any]]:
        """Generate a dataset of synthetic transactions"""
        transactions = []
        user_transaction_history = {}
        
        # Generate users
        users = [self.generate_user_id() for _ in range(num_users)]
        
        print(f"Generating {num_transactions} transactions for {num_users} users...")
        
        for i in range(num_transactions):
            if i % 1000 == 0:
                print(f"Generated {i} transactions...")
            
            # Select user (some users will have more transactions)
            user_id = np.random.choice(users, p=self._get_user_weights(num_users))
            profile_type = self.get_user_profile_type(user_id)
            
            # Get user's transaction history
            if user_id not in user_transaction_history:
                user_transaction_history[user_id] = []
            
            user_history = user_transaction_history[user_id]
            
            # Generate transaction timestamp (within last 90 days)
            base_time = datetime.now() - timedelta(days=random.randint(0, 90))
            timestamp = self.generate_transaction_time(base_time, profile_type)
            
            # Determine if fraud
            is_fraud = self.should_be_fraud(profile_type, user_history)
            
            # Generate location
            location = self.generate_location()
            
            # Generate transaction details
            transaction = {
                "user_id": user_id,
                "merchant_id": random.choice(self.merchants),
                "amount": self.generate_transaction_amount(profile_type, is_fraud),
                "currency": "USD",
                "transaction_type": random.choice(list(TransactionType)).value,
                "timestamp": timestamp,
                "latitude": location["latitude"],
                "longitude": location["longitude"],
                "country": location["country"],
                "city": location["city"],
                "device_id": self.generate_device_id(),
                "ip_address": self.generate_ip_address(),
                "user_agent": f"Mozilla/5.0 (User Agent {random.randint(1000, 9999)})",
                "session_id": self.generate_session_id(),
                "description": f"Transaction at {random.choice(self.merchants)}",
                "merchant_category": random.choice(self.merchant_categories),
                "payment_method": random.choice(list(PaymentMethod)).value,
                "is_fraud": is_fraud,
                "fraud_score": random.uniform(0.7, 1.0) if is_fraud else random.uniform(0.0, 0.3),
                "fraud_reasons": self.generate_fraud_indicators({"amount": 0, "timestamp": timestamp, "country": location["country"], "merchant_id": random.choice(self.merchants)}, user_history) if is_fraud else [],
                "model_version": "synthetic_v1.0",
                "processing_time_ms": random.uniform(10, 100)
            }
            
            transactions.append(transaction)
            user_transaction_history[user_id].append(transaction)
        
        print(f"Generated {len(transactions)} transactions")
        fraud_count = sum(1 for t in transactions if t["is_fraud"])
        print(f"Fraud rate: {fraud_count / len(transactions):.2%} ({fraud_count} fraudulent transactions)")
        
        return transactions
    
    def _get_user_weights(self, num_users: int) -> np.ndarray:
        """Generate user selection weights (some users more active than others)"""
        # Create a power law distribution for user activity
        weights = np.random.pareto(1.16, num_users) + 1
        return weights / weights.sum()
    
    def save_to_database(self, transactions: List[Dict[str, Any]]):
        """Save transactions to database"""
        print("Saving transactions to database...")
        
        create_tables()
        db = SessionLocal()
        
        try:
            for i, transaction_data in enumerate(transactions):
                if i % 500 == 0:
                    print(f"Saved {i} transactions...")
                
                transaction = Transaction(
                    user_id=transaction_data["user_id"],
                    merchant_id=transaction_data["merchant_id"],
                    amount=transaction_data["amount"],
                    currency=transaction_data["currency"],
                    transaction_type=transaction_data["transaction_type"],
                    timestamp=transaction_data["timestamp"],
                    latitude=transaction_data["latitude"],
                    longitude=transaction_data["longitude"],
                    country=transaction_data["country"],
                    city=transaction_data["city"],
                    device_id=transaction_data["device_id"],
                    ip_address=transaction_data["ip_address"],
                    user_agent=transaction_data["user_agent"],
                    session_id=transaction_data["session_id"],
                    description=transaction_data["description"],
                    merchant_category=transaction_data["merchant_category"],
                    payment_method=transaction_data["payment_method"],
                    is_fraud=transaction_data["is_fraud"],
                    fraud_score=transaction_data["fraud_score"],
                    fraud_reasons=transaction_data["fraud_reasons"],
                    model_version=transaction_data["model_version"],
                    processing_time_ms=transaction_data["processing_time_ms"]
                )
                
                db.add(transaction)
                
                # Commit in batches
                if i % 500 == 0:
                    db.commit()
            
            # Final commit
            db.commit()
            print(f"Successfully saved {len(transactions)} transactions to database")
            
        except Exception as e:
            print(f"Error saving to database: {e}")
            db.rollback()
        finally:
            db.close()
    
    def save_to_csv(self, transactions: List[Dict[str, Any]], filename: str = "synthetic_transactions.csv"):
        """Save transactions to CSV file"""
        df = pd.DataFrame(transactions)
        
        # Ensure data directory exists
        os.makedirs("data/raw", exist_ok=True)
        filepath = os.path.join("data/raw", filename)
        
        df.to_csv(filepath, index=False)
        print(f"Saved {len(transactions)} transactions to {filepath}")


def main():
    """Main function to generate and save synthetic data"""
    generator = SyntheticDataGenerator()
    
    # Generate transactions
    transactions = generator.generate_transactions(
        num_transactions=10000,
        num_users=500
    )
    
    # Save to CSV
    generator.save_to_csv(transactions)
    
    # Save to database
    generator.save_to_database(transactions)
    
    print("✅ Synthetic data generation completed!")


if __name__ == "__main__":
    main()
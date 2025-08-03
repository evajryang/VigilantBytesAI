from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    # Application settings
    app_name: str = "VigilantBytes AI"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Database settings
    database_url: str = "sqlite:///./vigilantbytes.db"
    database_echo: bool = False
    
    # Redis settings
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: Optional[str] = None
    
    # Security settings
    secret_key: str = "vigilantbytes-super-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # Machine Learning settings
    model_update_interval: int = 3600  # seconds
    fraud_threshold: float = 0.7
    max_training_samples: int = 100000
    
    # API settings
    cors_origins: list = ["http://localhost:3000", "http://localhost:8080"]
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Logging settings
    log_level: str = "INFO"
    log_file: str = "vigilantbytes.log"
    
    # Real-time processing
    stream_batch_size: int = 100
    stream_timeout: float = 5.0
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()
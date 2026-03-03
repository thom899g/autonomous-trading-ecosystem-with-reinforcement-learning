"""
Central configuration management for the trading ecosystem.
Pydantic validation ensures type safety and environment variable parsing.
"""
import os
from typing import Dict, Any, List
from pydantic import BaseSettings, Field, validator
from loguru import logger
import json

class TradingSettings(BaseSettings):
    """Core trading configuration with validation."""
    
    # Environment
    ENV: str = Field(default="development", env="ENV")
    DEBUG: bool = Field(default=False, env="DEBUG")
    
    # Exchange Configuration
    EXCHANGE_ID: str = Field(default="binance", env="EXCHANGE_ID")
    API_KEY: str = Field(default="", env="API_KEY")
    API_SECRET: str = Field(default="", env="API_SECRET")
    
    # Trading Parameters
    TRADING_PAIR: str = Field(default="BTC/USDT", env="TRADING_PAIR")
    TIMEFRAME: str = Field(default="1h", env="TIMEFRAME")
    INITIAL_CAPITAL: float = Field(default=10000.0, env="INITIAL_CAPITAL")
    
    # RL Model Configuration
    RL_MODEL_TYPE: str = Field(default="PPO", env="RL_MODEL_TYPE")
    TRAINING_EPISODES: int = Field(default=1000, env="TRAINING_EPISODES")
    
    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH: str = Field(
        default="config/firebase_credentials.json",
        env="FIREBASE_CREDENTIALS_PATH"
    )
    
    # Risk Management
    MAX_POSITION_SIZE: float = Field(default=0.1, env="MAX_POSITION_SIZE")
    MAX_DRAWDOWN_PCT: float = Field(default=0.25, env="MAX_DRAWDOWN_PCT")
    STOP_LOSS_PCT: float = Field(default=0.02, env="STOP_LOSS_PCT")
    
    # Data Collection
    HISTORICAL_DAYS: int = Field(default=365, env="HISTORICAL_DAYS")
    REAL_TIME_UPDATE_SECONDS: int = Field(default=60, env="REAL_TIME_UPDATE_SECONDS")
    
    @validator("ENV")
    def validate_env(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"ENV must be one of {allowed}")
        return v
    
    @validator("TIMEFRAME")
    def validate_timeframe(cls, v):
        allowed = ["1m", "5m", "15m", "1h", "4h", "1d"]
        if v not in allowed:
            raise ValueError(f"TIMEFRAME must be one of {allowed}")
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Initialize settings with error handling
try:
    SETTINGS = TradingSettings()
    logger.info(f"Configuration loaded for {SETTINGS.ENV} environment")
except Exception as e:
    logger.error(f"Configuration error: {e}")
    # Fallback to minimal safe defaults
    SETTINGS = TradingSettings(_env_file=None)
    SETTINGS.DEBUG = True
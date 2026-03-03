"""
Robust market data engine with multiple data sources, error handling, and caching.
Prioritizes reliability with fallback mechanisms and real-time data streaming.
"""
import asyncio
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
from loguru import logger
import ccxt
import yfinance as yf
from abc import ABC, abstractmethod
import time
from dataclasses import dataclass

@dataclass
class MarketData:
    """Type-safe market data container."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    symbol: str
    source: str

class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    async def fetch_historical(self, symbol: str, days: int,
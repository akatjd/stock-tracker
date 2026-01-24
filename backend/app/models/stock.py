from pydantic import BaseModel
from typing import List, Optional, Literal


class StockRSI(BaseModel):
    """주식 RSI 정보"""
    symbol: str
    name: str
    market: Literal["US", "KR"]
    price: float
    rsi: float
    change_percent: float


class OversoldScanRequest(BaseModel):
    """과매도 스캔 요청"""
    market: Literal["us", "kr", "all"] = "all"
    rsi_threshold: float = 30
    limit: int = 500  # 시장당 최대 스캔 종목 수


class OversoldScanResponse(BaseModel):
    """과매도 스캔 응답"""
    stocks: List[StockRSI]
    total_count: int
    market: str
    rsi_threshold: float

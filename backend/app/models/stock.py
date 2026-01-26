from pydantic import BaseModel
from typing import List, Optional, Literal


# 세분화된 시장 타입
MarketType = Literal[
    "all",           # 전체
    "kr",            # 한국 전체 (KOSPI + KOSDAQ)
    "us",            # 미국 전체 (NASDAQ + NYSE)
    "kospi",         # 한국 KOSPI
    "kosdaq",        # 한국 KOSDAQ
    "nasdaq",        # 미국 NASDAQ
    "dow"            # 미국 다우존스 30
]

# 시가총액 타입
MarketCapType = Literal[
    "all",           # 전체
    "large",         # 대형주: KR 10조+, US $10B+
    "mid",           # 중형주: KR 1조~10조, US $2B~$10B
    "small"          # 소형주: KR 1조 미만, US $2B 미만
]

# 봉 타입 (일봉, 주봉, 월봉)
PeriodType = Literal[
    "day",           # 일봉
    "week",          # 주봉
    "month"          # 월봉
]

# 섹터 타입
SectorType = Literal[
    "all",           # 전체
    "technology",    # 기술
    "finance",       # 금융
    "healthcare",    # 헬스케어
    "consumer",      # 소비재
    "industrial",    # 산업재
    "energy",        # 에너지
    "utilities",     # 유틸리티
    "materials",     # 소재
    "realestate",    # 부동산
    "communication"  # 통신
]

# 섹터 매핑 (yfinance 섹터명 -> 우리 섹터명)
SECTOR_MAPPING = {
    # US (yfinance)
    "Technology": "technology",
    "Financial Services": "finance",
    "Financials": "finance",
    "Healthcare": "healthcare",
    "Consumer Cyclical": "consumer",
    "Consumer Defensive": "consumer",
    "Consumer Discretionary": "consumer",
    "Consumer Staples": "consumer",
    "Industrials": "industrial",
    "Energy": "energy",
    "Utilities": "utilities",
    "Basic Materials": "materials",
    "Materials": "materials",
    "Real Estate": "realestate",
    "Communication Services": "communication",
    # KR (FinanceDataReader)
    "IT": "technology",
    "금융": "finance",
    "의료정밀": "healthcare",
    "의약품": "healthcare",
    "서비스업": "consumer",
    "유통업": "consumer",
    "음식료품": "consumer",
    "화학": "materials",
    "철강및금속": "materials",
    "기계": "industrial",
    "전기전자": "technology",
    "운수장비": "industrial",
    "건설업": "industrial",
    "전기가스업": "utilities",
    "통신업": "communication",
}


class StockRSI(BaseModel):
    """주식 RSI 정보"""
    symbol: str
    name: str
    market: str  # KOSPI, KOSDAQ, NASDAQ, DOW 등
    price: float
    rsi: float
    change_percent: float
    sector: Optional[str] = None  # 섹터
    market_cap: Optional[float] = None  # 시가총액
    market_cap_label: Optional[str] = None  # 대형주/중형주/소형주


class OversoldScanRequest(BaseModel):
    """과매도 스캔 요청"""
    market: MarketType = "all"
    rsi_threshold: float = 30
    limit: int = 500  # 시장당 최대 스캔 종목 수


class OversoldScanResponse(BaseModel):
    """과매도 스캔 응답"""
    stocks: List[StockRSI]
    total_count: int
    market: str
    rsi_threshold: float

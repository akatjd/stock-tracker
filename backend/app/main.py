from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging
from typing import Literal, Optional
import asyncio
from concurrent.futures import ThreadPoolExecutor

from app.config import settings
from app.services.stock_data import stock_service
from app.models.stock import StockRSI, OversoldScanResponse, MarketType, MarketCapType, SectorType

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title="Stock Tracker API",
    description="주식 추적 및 모니터링 API - RSI 과매도 종목 스캐너",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 스캔 결과 캐시
scan_cache = {
    "results": [],
    "is_scanning": False,
    "last_scan": None,
    "progress": 0
}


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "Stock Tracker API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "scan_oversold": "/api/v1/scan/oversold",
            "scan_status": "/api/v1/scan/status"
        }
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {"status": "healthy"}


@app.get("/api/v1/scan/oversold", response_model=OversoldScanResponse)
async def scan_oversold_stocks(
    market: MarketType = Query("all", description="스캔할 시장 (all, us, kr, kospi, kosdaq, nasdaq, dow)"),
    rsi_threshold: float = Query(30, description="RSI 기준값 (이하인 종목 필터링)"),
    limit: int = Query(500, description="시장당 최대 스캔 종목 수"),
    market_cap: MarketCapType = Query("all", description="시가총액 필터 (all, large, mid, small)"),
    sector: SectorType = Query("all", description="섹터 필터")
):
    """
    RSI 과매도 종목 스캔

    - **market**:
        - all: 전체 시장
        - us: 미국 전체 (NASDAQ + DOW)
        - kr: 한국 전체 (KOSPI + KOSDAQ)
        - kospi: 한국 KOSPI
        - kosdaq: 한국 KOSDAQ
        - nasdaq: 미국 NASDAQ
        - dow: 미국 다우존스 30
    - **rsi_threshold**: RSI 기준값 (기본 30)
    - **limit**: 시장당 최대 스캔 종목 수 (기본 500)
    - **market_cap**: 시가총액 필터
        - all: 전체
        - large: 대형주 (KR 10조+, US $10B+)
        - mid: 중형주 (KR 1조~10조, US $2B~$10B)
        - small: 소형주 (KR 1조 미만, US $2B 미만)
    - **sector**: 섹터 필터
        - all, technology, finance, healthcare, consumer, industrial, energy, utilities, materials, realestate, communication

    주의: 전체 시장 스캔 시 시간이 오래 걸릴 수 있습니다.
    """
    logger.info(f"Starting oversold scan: market={market}, threshold={rsi_threshold}, limit={limit}, market_cap={market_cap}, sector={sector}")

    # 동기 함수를 비동기로 실행
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        results = await loop.run_in_executor(
            executor,
            lambda: stock_service.scan_oversold_stocks(
                market=market,
                rsi_threshold=rsi_threshold,
                limit=limit,
                market_cap_filter=market_cap,
                sector_filter=sector
            )
        )

    return OversoldScanResponse(
        stocks=results,
        total_count=len(results),
        market=market,
        rsi_threshold=rsi_threshold
    )


@app.get("/api/v1/symbols/us")
async def get_us_symbols():
    """미국 주식 심볼 목록 조회"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        symbols = await loop.run_in_executor(
            executor,
            stock_service.get_us_symbols
        )

    return {
        "market": "US",
        "count": len(symbols),
        "symbols": symbols[:100]  # 처음 100개만 반환
    }


@app.get("/api/v1/symbols/kr")
async def get_kr_symbols():
    """한국 주식 심볼 목록 조회"""
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        symbols = await loop.run_in_executor(
            executor,
            stock_service.get_kr_symbols
        )

    return {
        "market": "KR",
        "count": len(symbols),
        "symbols": dict(list(symbols.items())[:100])  # 처음 100개만 반환
    }


@app.get("/api/v1/stock/{symbol}/rsi")
async def get_stock_rsi(
    symbol: str,
    market: Literal["us", "kr"] = Query("us", description="시장 (us 또는 kr)")
):
    """개별 종목 RSI 조회"""
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        if market == "us":
            result = await loop.run_in_executor(
                executor,
                lambda: stock_service.get_us_stock_rsi(symbol.upper())
            )
        else:
            # 한국 주식은 코드만으로 조회
            kr_symbols = stock_service.get_kr_symbols()
            name = kr_symbols.get(symbol, symbol)
            result = await loop.run_in_executor(
                executor,
                lambda: stock_service.get_kr_stock_rsi(symbol, name)
            )

    if not result:
        return {"error": f"Failed to get RSI for {symbol}"}

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

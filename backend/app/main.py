from fastapi import FastAPI, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import logging
from typing import Literal, Optional
import asyncio
import json
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


@app.get("/api/v1/scan/oversold/stream")
async def scan_oversold_stocks_stream(
    market: MarketType = Query("all", description="스캔할 시장"),
    rsi_threshold: float = Query(30, description="RSI 기준값"),
    limit: int = Query(500, description="시장당 최대 스캔 종목 수"),
    market_cap: MarketCapType = Query("all", description="시가총액 필터"),
    sector: SectorType = Query("all", description="섹터 필터")
):
    """
    RSI 과매도 종목 스캔 (실시간 스트리밍)

    Server-Sent Events를 통해 스캔 진행 상황을 실시간으로 전송합니다.
    """
    async def generate():
        loop = asyncio.get_event_loop()

        # 진행 상황을 저장할 큐
        progress_queue = asyncio.Queue()

        def progress_callback(current, total, symbol, market_name, found_count):
            """진행 상황 콜백"""
            asyncio.run_coroutine_threadsafe(
                progress_queue.put({
                    "type": "progress",
                    "current": current,
                    "total": total,
                    "symbol": symbol,
                    "market": market_name,
                    "found": found_count,
                    "percent": round((current / total) * 100, 1) if total > 0 else 0
                }),
                loop
            )

        def result_callback(result):
            """결과 발견 콜백"""
            asyncio.run_coroutine_threadsafe(
                progress_queue.put({
                    "type": "found",
                    "stock": result
                }),
                loop
            )

        # 백그라운드에서 스캔 실행
        async def run_scan():
            with ThreadPoolExecutor() as executor:
                results = await loop.run_in_executor(
                    executor,
                    lambda: stock_service.scan_oversold_stocks_with_progress(
                        market=market,
                        rsi_threshold=rsi_threshold,
                        limit=limit,
                        market_cap_filter=market_cap,
                        sector_filter=sector,
                        progress_callback=progress_callback,
                        result_callback=result_callback
                    )
                )
                await progress_queue.put({
                    "type": "complete",
                    "results": results,
                    "total_count": len(results)
                })

        # 스캔 태스크 시작
        scan_task = asyncio.create_task(run_scan())

        try:
            while True:
                try:
                    # 큐에서 진행 상황 가져오기
                    data = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                    yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

                    if data.get("type") == "complete":
                        break
                except asyncio.TimeoutError:
                    # 연결 유지를 위한 heartbeat
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            if not scan_task.done():
                scan_task.cancel()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
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

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
from app.models.stock import StockRSI, OversoldScanResponse, MarketType, MarketCapType, SectorType, PeriodType

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
    sector: SectorType = Query("all", description="섹터 필터"),
    period: PeriodType = Query("day", description="봉 타입 (day: 일봉, week: 주봉, month: 월봉)"),
    custom_stocks: Optional[str] = Query(None, description="커스텀 종목 리스트 (JSON)")
):
    """
    RSI 과매도 종목 스캔 (실시간 스트리밍)

    Server-Sent Events를 통해 스캔 진행 상황을 실시간으로 전송합니다.

    - **period**: 봉 타입
        - day: 일봉 (기본값)
        - week: 주봉
        - month: 월봉
    - **custom_stocks**: 추가로 스캔할 커스텀 종목 (JSON 형식)
    """
    # 커스텀 종목 파싱
    parsed_custom_stocks = []
    if custom_stocks:
        try:
            parsed_custom_stocks = json.loads(custom_stocks)
            logger.info(f"Custom stocks to scan: {len(parsed_custom_stocks)}")
        except json.JSONDecodeError:
            logger.warning("Failed to parse custom_stocks JSON")
    async def generate():
        logger.info(f"SSE stream started for market={market}")
        loop = asyncio.get_event_loop()

        # 진행 상황을 저장할 큐
        progress_queue = asyncio.Queue()

        # 취소 플래그
        import threading
        cancelled = threading.Event()

        def cancel_check():
            return cancelled.is_set()

        # 연결 시작 이벤트 즉시 전송
        logger.info("Sending connected event")
        yield f"data: {json.dumps({'type': 'connected', 'message': 'Scan started'})}\n\n"

        def progress_callback(current, total, symbol, market_name, found_count):
            """진행 상황 콜백"""
            if current <= 3:  # 처음 3개만 로그
                logger.info(f"Progress: {current}/{total} - {symbol} ({market_name})")
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

        def start_callback(total):
            """스캔 시작 콜백"""
            logger.info(f"Start callback: total={total}")
            asyncio.run_coroutine_threadsafe(
                progress_queue.put({
                    "type": "start",
                    "total": total
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
                        candle_period=period,
                        progress_callback=progress_callback,
                        result_callback=result_callback,
                        start_callback=start_callback,
                        cancel_check=cancel_check,
                        custom_stocks=parsed_custom_stocks
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
        except asyncio.CancelledError:
            logger.info("SSE stream cancelled by client")
            cancelled.set()
        except GeneratorExit:
            logger.info("Client disconnected (GeneratorExit)")
            cancelled.set()
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            # 스캔 취소 신호 전송
            cancelled.set()
            if not scan_task.done():
                scan_task.cancel()
                try:
                    await asyncio.wait_for(scan_task, timeout=1.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass
            logger.info("SSE stream closed")

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/api/v1/scan/preview")
async def preview_scan_stocks(
    market: MarketType = Query("all", description="스캔할 시장"),
    limit: int = Query(500, description="시장당 최대 스캔 종목 수"),
    market_cap: MarketCapType = Query("all", description="시가총액 필터"),
    sector: SectorType = Query("all", description="섹터 필터"),
    page: int = Query(1, description="페이지 번호", ge=1),
    page_size: int = Query(50, description="페이지당 항목 수", ge=10, le=100),
    search: Optional[str] = Query(None, description="종목코드 또는 종목명 검색")
):
    """
    스캔 대상 종목 미리보기 (페이징 및 검색 지원)

    실제 RSI를 계산하지 않고 스캔 대상이 될 종목 목록만 반환합니다.
    """
    loop = asyncio.get_event_loop()

    def get_preview_stocks():
        all_stocks = []

        # NASDAQ
        if market in ["nasdaq", "us", "all"]:
            nasdaq_symbols = stock_service.get_nasdaq_symbols()[:limit]
            all_stocks.extend([{"symbol": s, "name": s, "market": "NASDAQ"} for s in nasdaq_symbols])

        # DOW
        if market in ["dow", "us", "all"]:
            dow_symbols = stock_service.get_dow_symbols()[:limit]
            all_stocks.extend([{"symbol": s, "name": s, "market": "DOW"} for s in dow_symbols])

        # KOSPI
        if market in ["kospi", "kr", "all"]:
            kospi_stocks = stock_service.get_kospi_symbols_detailed()[:limit]
            all_stocks.extend([{
                "symbol": s['code'],
                "name": s['name'],
                "market": "KOSPI",
                "sector": s.get('sector'),
                "market_cap": s.get('market_cap')
            } for s in kospi_stocks])

        # KOSDAQ
        if market in ["kosdaq", "kr", "all"]:
            kosdaq_stocks = stock_service.get_kosdaq_symbols_detailed()[:limit]
            all_stocks.extend([{
                "symbol": s['code'],
                "name": s['name'],
                "market": "KOSDAQ",
                "sector": s.get('sector'),
                "market_cap": s.get('market_cap')
            } for s in kosdaq_stocks])

        return all_stocks

    with ThreadPoolExecutor() as executor:
        stocks = await loop.run_in_executor(executor, get_preview_stocks)

    # 시장별 개수 계산 (검색 전 전체 기준)
    market_counts = {}
    for stock in stocks:
        m = stock['market']
        market_counts[m] = market_counts.get(m, 0) + 1

    total_count = len(stocks)

    # 검색 필터 적용
    if search:
        search_lower = search.lower()
        stocks = [
            s for s in stocks
            if search_lower in s['symbol'].lower() or search_lower in s['name'].lower()
        ]

    filtered_count = len(stocks)

    # 페이징 적용
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_stocks = stocks[start_idx:end_idx]
    total_pages = (filtered_count + page_size - 1) // page_size

    return {
        "total_count": total_count,
        "filtered_count": filtered_count,
        "market_counts": market_counts,
        "stocks": paged_stocks,
        "page": page,
        "page_size": page_size,
        "total_pages": total_pages,
        "has_prev": page > 1,
        "has_next": page < total_pages
    }


@app.get("/api/v1/stock/validate")
async def validate_stock(
    symbol: str = Query(..., description="종목코드 또는 심볼"),
    market: str = Query(..., description="시장 (KOSPI, KOSDAQ, NASDAQ, NYSE, DOW)")
):
    """
    종목이 실제로 존재하는지 검증

    - **symbol**: 종목코드 또는 심볼 (예: 005930, AAPL)
    - **market**: 시장 (KOSPI, KOSDAQ, NASDAQ, NYSE, DOW)

    Returns:
        valid: 유효 여부
        symbol: 종목코드
        name: 종목명
        market: 시장
        message: 결과 메시지
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: stock_service.validate_stock(symbol, market)
        )

    return result


@app.get("/api/v1/stock/detail/{symbol}")
async def get_stock_detail(
    symbol: str,
    market: str = Query(..., description="시장 (KOSPI, KOSDAQ, NASDAQ, NYSE)"),
    period: str = Query("6mo", description="기간 (1mo, 3mo, 6mo, 1y, 2y, 5y)"),
    interval: str = Query("1d", description="봉 타입 (1h, 4h, 1d, 1wk, 1mo)")
):
    """
    종목 상세 정보 조회 (차트 데이터 + 재무제표)

    - **symbol**: 종목코드 또는 심볼
    - **market**: 시장
    - **period**: 데이터 기간 (1mo, 3mo, 6mo, 1y, 2y, 5y)
    - **interval**: 봉 타입 (1h, 4h, 1d, 1wk, 1mo)

    Returns:
        - 기본 정보 (가격, 등락률, 52주 최고/최저 등)
        - 차트 데이터
        - 재무제표 정보
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: stock_service.get_stock_detail(symbol, market, period, interval)
        )

    return result


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


@app.get("/api/v1/stock/search")
async def search_stocks(
    query: str = Query(..., description="검색어 (종목코드, 종목명)"),
    limit: int = Query(10, description="최대 결과 수", ge=1, le=50)
):
    """
    종목 검색 API

    한글명, 영문명, 종목코드로 검색 가능
    예: 삼성전자, APPLE, 005930, AAPL
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        results = await loop.run_in_executor(
            executor,
            lambda: stock_service.search_stocks(query, limit)
        )

    return {
        "query": query,
        "count": len(results),
        "results": results
    }


@app.get("/api/v1/stock/{symbol}/quote")
async def get_stock_quote(
    symbol: str,
    market: str = Query(..., description="시장 (KOSPI, KOSDAQ, NASDAQ, NYSE)")
):
    """
    실시간 시세 경량 조회 (자동갱신용)

    전체 상세 정보 대신 현재가, 등락, 거래량만 빠르게 반환합니다.
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: stock_service.get_stock_quote(symbol, market)
        )

    return result


@app.get("/api/v1/stock/{symbol}/backtest")
async def run_backtest(
    symbol: str,
    market: str = Query(..., description="시장 (KOSPI, KOSDAQ, NASDAQ, NYSE)"),
    strategy: str = Query("rsi", description="전략 (rsi, ma_cross, rsi_ma)"),
    buy_rsi: float = Query(30, description="RSI 매수 기준"),
    sell_rsi: float = Query(70, description="RSI 매도 기준"),
    period: str = Query("2y", description="백테스트 기간 (1y, 2y, 3y, 5y)"),
    initial_capital: float = Query(10000000, description="초기 투자금")
):
    """
    백테스트 시뮬레이터

    - **symbol**: 종목코드/심볼
    - **market**: 시장
    - **strategy**: 전략 타입
        - rsi: RSI 기반 매매 (RSI ≤ buy_rsi 매수, RSI ≥ sell_rsi 매도)
        - ma_cross: 이동평균 교차 (MA5/MA20 골든크로스 매수, 데드크로스 매도)
        - rsi_ma: RSI + 이동평균 복합 전략
    - **buy_rsi**: RSI 매수 기준값
    - **sell_rsi**: RSI 매도 기준값
    - **period**: 백테스트 기간
    - **initial_capital**: 초기 투자금
    """
    loop = asyncio.get_event_loop()

    with ThreadPoolExecutor() as executor:
        result = await loop.run_in_executor(
            executor,
            lambda: stock_service.run_backtest(
                symbol=symbol,
                market=market,
                strategy=strategy,
                buy_rsi=buy_rsi,
                sell_rsi=sell_rsi,
                period=period,
                initial_capital=initial_capital
            )
        )

    return result


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

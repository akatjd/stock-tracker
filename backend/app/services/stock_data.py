import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.rsi_calculator import calculate_rsi

logger = logging.getLogger(__name__)


class StockDataService:
    """주식 데이터 서비스"""

    def __init__(self):
        self._us_symbols_cache: Optional[List[str]] = None
        self._kr_symbols_cache: Optional[Dict[str, str]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=24)

    def get_us_symbols(self) -> List[str]:
        """미국 전체 주식 심볼 가져오기 (NYSE + NASDAQ)"""
        if self._us_symbols_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._us_symbols_cache

        try:
            # NYSE
            nyse = fdr.StockListing('NYSE')
            nyse_symbols = nyse['Symbol'].tolist() if 'Symbol' in nyse.columns else []

            # NASDAQ
            nasdaq = fdr.StockListing('NASDAQ')
            nasdaq_symbols = nasdaq['Symbol'].tolist() if 'Symbol' in nasdaq.columns else []

            # 합치고 중복 제거, 특수문자 포함 심볼 제외
            all_symbols = list(set(nyse_symbols + nasdaq_symbols))
            all_symbols = [s for s in all_symbols if isinstance(s, str) and s.isalpha() and len(s) <= 5]

            self._us_symbols_cache = all_symbols
            self._cache_time = datetime.now()

            logger.info(f"Loaded {len(all_symbols)} US symbols")
            return all_symbols

        except Exception as e:
            logger.error(f"Failed to get US symbols: {e}")
            return []

    def get_kr_symbols(self) -> Dict[str, str]:
        """한국 전체 주식 심볼 가져오기 (KOSPI + KOSDAQ) - {코드: 종목명}"""
        if self._kr_symbols_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._kr_symbols_cache

        try:
            # KOSPI
            kospi = fdr.StockListing('KOSPI')
            kospi_dict = dict(zip(kospi['Code'], kospi['Name'])) if 'Code' in kospi.columns else {}

            # KOSDAQ
            kosdaq = fdr.StockListing('KOSDAQ')
            kosdaq_dict = dict(zip(kosdaq['Code'], kosdaq['Name'])) if 'Code' in kosdaq.columns else {}

            # 합치기
            all_symbols = {**kospi_dict, **kosdaq_dict}

            self._kr_symbols_cache = all_symbols
            self._cache_time = datetime.now()

            logger.info(f"Loaded {len(all_symbols)} KR symbols")
            return all_symbols

        except Exception as e:
            logger.error(f"Failed to get KR symbols: {e}")
            return {}

    def get_us_stock_rsi(self, symbol: str, period: int = 14) -> Optional[Dict]:
        """미국 주식 RSI 조회"""
        try:
            ticker = yf.Ticker(symbol)
            # 최근 50일 데이터 (RSI 계산에 충분한 양)
            hist = ticker.history(period="3mo")

            if hist.empty or len(hist) < period + 1:
                return None

            rsi = calculate_rsi(hist['Close'], period)

            if rsi is None:
                return None

            info = ticker.info
            return {
                "symbol": symbol,
                "name": info.get('shortName', symbol),
                "market": "US",
                "price": round(hist['Close'].iloc[-1], 2),
                "rsi": rsi,
                "change_percent": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100, 2) if len(hist) > 1 else 0
            }

        except Exception as e:
            logger.debug(f"Failed to get RSI for {symbol}: {e}")
            return None

    def get_kr_stock_rsi(self, code: str, name: str, period: int = 14) -> Optional[Dict]:
        """한국 주식 RSI 조회"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=100)

            df = fdr.DataReader(code, start_date, end_date)

            if df.empty or len(df) < period + 1:
                return None

            rsi = calculate_rsi(df['Close'], period)

            if rsi is None:
                return None

            return {
                "symbol": code,
                "name": name,
                "market": "KR",
                "price": int(df['Close'].iloc[-1]),
                "rsi": rsi,
                "change_percent": round(((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100, 2) if len(df) > 1 else 0
            }

        except Exception as e:
            logger.debug(f"Failed to get RSI for {code}: {e}")
            return None

    def scan_oversold_stocks(
        self,
        market: str = "all",
        rsi_threshold: float = 30,
        max_workers: int = 10,
        limit: int = 100
    ) -> List[Dict]:
        """
        RSI 과매도 종목 스캔

        Args:
            market: "us", "kr", or "all"
            rsi_threshold: RSI 기준값 (이하인 종목 필터링)
            max_workers: 병렬 처리 스레드 수
            limit: 각 시장당 최대 스캔 종목 수 (전체 스캔 시 시간 제한)

        Returns:
            과매도 종목 리스트
        """
        results = []

        # 미국 주식 스캔
        if market in ["us", "all"]:
            us_symbols = self.get_us_symbols()[:limit]
            logger.info(f"Scanning {len(us_symbols)} US stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.get_us_stock_rsi, symbol): symbol for symbol in us_symbols}

                for future in as_completed(futures):
                    result = future.result()
                    if result and result['rsi'] <= rsi_threshold:
                        results.append(result)

        # 한국 주식 스캔
        if market in ["kr", "all"]:
            kr_symbols = self.get_kr_symbols()
            kr_items = list(kr_symbols.items())[:limit]
            logger.info(f"Scanning {len(kr_items)} KR stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.get_kr_stock_rsi, code, name): code for code, name in kr_items}

                for future in as_completed(futures):
                    result = future.result()
                    if result and result['rsi'] <= rsi_threshold:
                        results.append(result)

        # RSI 오름차순 정렬 (가장 과매도인 종목 먼저)
        results.sort(key=lambda x: x['rsi'])

        logger.info(f"Found {len(results)} oversold stocks")
        return results


# 싱글톤 인스턴스
stock_service = StockDataService()

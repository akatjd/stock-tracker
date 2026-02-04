import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import math
import logging
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.services.rsi_calculator import calculate_rsi
from app.models.stock import SECTOR_MAPPING

logger = logging.getLogger(__name__)


def translate_to_korean(text: str) -> str:
    """영어 텍스트를 한국어로 번역"""
    if not text:
        return text
    try:
        from deep_translator import GoogleTranslator
        translator = GoogleTranslator(source='en', target='ko')
        # 텍스트가 너무 길면 분할 번역 (Google Translate 제한)
        if len(text) > 4500:
            text = text[:4500]
        translated = translator.translate(text)
        return translated
    except Exception as e:
        logger.warning(f"Translation failed: {e}")
        return text  # 번역 실패 시 원문 반환


def calculate_bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0):
    """볼린저 밴드 계산"""
    sma = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    return sma, upper, lower


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD 계산"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_rsi_series(prices: pd.Series, period: int = 14):
    """RSI 시리즈 계산 (차트용)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_support_resistance(df: pd.DataFrame, window: int = 20):
    """지지선/저항선 계산 (피봇 포인트 기반)"""
    recent = df.tail(window)

    # 최근 고점/저점 기반
    resistance_levels = []
    support_levels = []

    # 로컬 최고점/최저점 찾기
    highs = recent['High'].values
    lows = recent['Low'].values

    for i in range(2, len(highs) - 2):
        # 로컬 최고점
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(float(highs[i]))
        # 로컬 최저점
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(float(lows[i]))

    # 중복 제거 및 클러스터링 (비슷한 가격대 병합)
    def cluster_levels(levels, threshold=0.02):
        if not levels:
            return []
        levels = sorted(levels)
        clustered = [levels[0]]
        for level in levels[1:]:
            if (level - clustered[-1]) / clustered[-1] > threshold:
                clustered.append(level)
        return clustered[-3:]  # 최근 3개만

    return {
        "resistance": cluster_levels(resistance_levels),
        "support": cluster_levels(support_levels)
    }


def resample_to_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    """
    일봉 데이터를 주봉 또는 월봉으로 변환

    Args:
        df: OHLC 데이터프레임 (Close 컬럼 필수)
        period: 'day', 'week', 'month'

    Returns:
        리샘플링된 데이터프레임
    """
    if period == 'day':
        return df

    # 리샘플링 규칙
    rule = 'W' if period == 'week' else 'ME'  # ME = Month End

    # OHLC 리샘플링
    resampled = df.resample(rule).agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()

    return resampled


def get_market_cap_label(market_cap: float, is_korean: bool = False) -> str:
    """시가총액을 기준으로 대형/중형/소형주 라벨 반환"""
    if market_cap is None:
        return "unknown"

    if is_korean:
        # 한국: 10조 이상 대형, 1조~10조 중형, 1조 미만 소형
        if market_cap >= 10_000_000_000_000:  # 10조
            return "large"
        elif market_cap >= 1_000_000_000_000:  # 1조
            return "mid"
        else:
            return "small"
    else:
        # 미국: $10B 이상 대형, $2B~$10B 중형, $2B 미만 소형
        if market_cap >= 10_000_000_000:  # $10B
            return "large"
        elif market_cap >= 2_000_000_000:  # $2B
            return "mid"
        else:
            return "small"


def normalize_sector(sector: str) -> Optional[str]:
    """섹터명을 정규화"""
    if not sector:
        return None
    return SECTOR_MAPPING.get(sector, None)

# 다우존스 30 종목 리스트
DOW_30_SYMBOLS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "DOW",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PG", "TRV", "UNH", "V", "VZ", "WBA", "WMT"
]


class StockDataService:
    """주식 데이터 서비스"""

    def __init__(self):
        self._us_symbols_cache: Optional[List[str]] = None
        self._kr_symbols_cache: Optional[Dict[str, str]] = None
        self._kospi_cache: Optional[Dict[str, str]] = None
        self._kosdaq_cache: Optional[Dict[str, str]] = None
        self._nasdaq_cache: Optional[List[str]] = None
        self._nyse_cache: Optional[List[str]] = None
        self._cache_time: Optional[datetime] = None
        self._cache_duration = timedelta(hours=24)
        self._exchange_rate_cache: Optional[float] = None
        self._exchange_rate_time: Optional[datetime] = None

    def get_usd_krw_rate(self) -> Optional[float]:
        """USD/KRW 환율 조회 (1시간 캐싱)"""
        if self._exchange_rate_cache and self._exchange_rate_time:
            if datetime.now() - self._exchange_rate_time < timedelta(hours=1):
                return self._exchange_rate_cache

        try:
            ticker = yf.Ticker("USDKRW=X")
            hist = ticker.history(period="1d")
            if not hist.empty:
                rate = round(float(hist['Close'].iloc[-1]), 2)
                self._exchange_rate_cache = rate
                self._exchange_rate_time = datetime.now()
                logger.info(f"USD/KRW exchange rate: {rate}")
                return rate
        except Exception as e:
            logger.error(f"Failed to get USD/KRW rate: {e}")

        return self._exchange_rate_cache or 1350.0  # 실패 시 기본값

    def get_stock_news(self, symbol: str, limit: int = 10) -> List[Dict]:
        """종목 관련 뉴스 조회"""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news
            if not news:
                return []

            result = []
            for item in news[:limit]:
                content = item.get('content', item)
                title = content.get('title', '')
                if not title:
                    continue

                # 링크
                link = ''
                canonical = content.get('canonicalUrl') or content.get('clickThroughUrl')
                if canonical:
                    link = canonical.get('url', '')
                elif content.get('link'):
                    link = content.get('link', '')

                # 매체명
                provider = content.get('provider', {})
                publisher = provider.get('displayName', '') if isinstance(provider, dict) else content.get('publisher', '')

                # 날짜
                pub_date = content.get('pubDate', '')

                # 썸네일
                thumbnail = ''
                thumb_data = content.get('thumbnail')
                if thumb_data and thumb_data.get('resolutions'):
                    resolutions = thumb_data['resolutions']
                    thumbnail = resolutions[-1].get('url', '') if len(resolutions) > 1 else resolutions[0].get('url', '')

                result.append({
                    'title': title,
                    'publisher': publisher,
                    'link': link,
                    'pubDate': pub_date,
                    'thumbnail': thumbnail,
                })
            return result
        except Exception as e:
            logger.error(f"Failed to get news for {symbol}: {e}")
            return []

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

    def get_nasdaq_symbols(self) -> List[str]:
        """NASDAQ 주식 심볼 가져오기"""
        if self._nasdaq_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._nasdaq_cache

        try:
            nasdaq = fdr.StockListing('NASDAQ')
            nasdaq_symbols = nasdaq['Symbol'].tolist() if 'Symbol' in nasdaq.columns else []
            nasdaq_symbols = [s for s in nasdaq_symbols if isinstance(s, str) and s.isalpha() and len(s) <= 5]

            self._nasdaq_cache = nasdaq_symbols
            self._cache_time = datetime.now()

            logger.info(f"Loaded {len(nasdaq_symbols)} NASDAQ symbols")
            return nasdaq_symbols

        except Exception as e:
            logger.error(f"Failed to get NASDAQ symbols: {e}")
            return []

    def get_dow_symbols(self) -> List[str]:
        """다우존스 30 심볼 가져오기"""
        return DOW_30_SYMBOLS.copy()

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

    def get_kospi_symbols_detailed(self) -> List[Dict]:
        """KOSPI 주식 상세 정보 가져오기 - [{code, name, sector, market_cap}, ...]"""
        try:
            kospi = fdr.StockListing('KOSPI')
            result = []
            for _, row in kospi.iterrows():
                code = row.get('Code', '')
                name = row.get('Name', '')
                sector = row.get('Sector', row.get('Industry', ''))
                market_cap = row.get('Marcap', row.get('MarketCap', None))
                if code and name:
                    result.append({
                        'code': code,
                        'name': name,
                        'sector': sector,
                        'market_cap': market_cap
                    })
            logger.info(f"Loaded {len(result)} KOSPI symbols with details")
            return result
        except Exception as e:
            logger.error(f"Failed to get KOSPI symbols: {e}")
            return []

    def get_kospi_symbols(self) -> Dict[str, str]:
        """KOSPI 주식 심볼 가져오기 - {코드: 종목명}"""
        if self._kospi_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._kospi_cache

        try:
            kospi = fdr.StockListing('KOSPI')
            kospi_dict = dict(zip(kospi['Code'], kospi['Name'])) if 'Code' in kospi.columns else {}

            self._kospi_cache = kospi_dict
            self._cache_time = datetime.now()

            logger.info(f"Loaded {len(kospi_dict)} KOSPI symbols")
            return kospi_dict

        except Exception as e:
            logger.error(f"Failed to get KOSPI symbols: {e}")
            return {}

    def get_kosdaq_symbols_detailed(self) -> List[Dict]:
        """KOSDAQ 주식 상세 정보 가져오기 - [{code, name, sector, market_cap}, ...]"""
        try:
            kosdaq = fdr.StockListing('KOSDAQ')
            result = []
            for _, row in kosdaq.iterrows():
                code = row.get('Code', '')
                name = row.get('Name', '')
                sector = row.get('Sector', row.get('Industry', ''))
                market_cap = row.get('Marcap', row.get('MarketCap', None))
                if code and name:
                    result.append({
                        'code': code,
                        'name': name,
                        'sector': sector,
                        'market_cap': market_cap
                    })
            logger.info(f"Loaded {len(result)} KOSDAQ symbols with details")
            return result
        except Exception as e:
            logger.error(f"Failed to get KOSDAQ symbols: {e}")
            return []

    def get_kosdaq_symbols(self) -> Dict[str, str]:
        """KOSDAQ 주식 심볼 가져오기 - {코드: 종목명}"""
        if self._kosdaq_cache and self._cache_time:
            if datetime.now() - self._cache_time < self._cache_duration:
                return self._kosdaq_cache

        try:
            kosdaq = fdr.StockListing('KOSDAQ')
            kosdaq_dict = dict(zip(kosdaq['Code'], kosdaq['Name'])) if 'Code' in kosdaq.columns else {}

            self._kosdaq_cache = kosdaq_dict
            self._cache_time = datetime.now()

            logger.info(f"Loaded {len(kosdaq_dict)} KOSDAQ symbols")
            return kosdaq_dict

        except Exception as e:
            logger.error(f"Failed to get KOSDAQ symbols: {e}")
            return {}

    def get_us_stock_rsi(self, symbol: str, rsi_period: int = 14, market_label: str = "US",
                          candle_period: str = "day") -> Optional[Dict]:
        """
        미국 주식 RSI 조회

        Args:
            symbol: 주식 심볼
            rsi_period: RSI 계산 기간 (기본 14)
            market_label: 시장 라벨
            candle_period: 봉 타입 (day, week, month)
        """
        try:
            ticker = yf.Ticker(symbol)
            # 주봉/월봉의 경우 더 긴 기간 필요
            if candle_period == 'month':
                hist = ticker.history(period="2y")
            elif candle_period == 'week':
                hist = ticker.history(period="1y")
            else:
                hist = ticker.history(period="3mo")

            if hist.empty:
                return None

            # 주봉/월봉 리샘플링
            if candle_period in ['week', 'month']:
                hist = resample_to_period(hist, candle_period)

            if len(hist) < rsi_period + 1:
                return None

            rsi = calculate_rsi(hist['Close'], rsi_period)

            if rsi is None:
                return None

            info = ticker.info
            market_cap = info.get('marketCap')
            sector = info.get('sector')

            return {
                "symbol": symbol,
                "name": info.get('shortName', symbol),
                "market": market_label,
                "price": round(hist['Close'].iloc[-1], 2),
                "rsi": rsi,
                "change_percent": round(((hist['Close'].iloc[-1] / hist['Close'].iloc[-2]) - 1) * 100, 2) if len(hist) > 1 else 0,
                "sector": normalize_sector(sector),
                "market_cap": market_cap,
                "market_cap_label": get_market_cap_label(market_cap, is_korean=False)
            }

        except Exception as e:
            logger.debug(f"Failed to get RSI for {symbol}: {e}")
            return None

    def get_kr_stock_rsi(self, code: str, name: str, rsi_period: int = 14, market_label: str = "KR",
                         sector: str = None, market_cap: float = None,
                         candle_period: str = "day") -> Optional[Dict]:
        """
        한국 주식 RSI 조회

        Args:
            code: 주식 코드
            name: 종목명
            rsi_period: RSI 계산 기간 (기본 14)
            market_label: 시장 라벨
            sector: 섹터
            market_cap: 시가총액
            candle_period: 봉 타입 (day, week, month)
        """
        try:
            end_date = datetime.now()
            # 주봉/월봉의 경우 더 긴 기간 필요
            if candle_period == 'month':
                start_date = end_date - timedelta(days=730)  # 2년
            elif candle_period == 'week':
                start_date = end_date - timedelta(days=365)  # 1년
            else:
                start_date = end_date - timedelta(days=100)

            df = fdr.DataReader(code, start_date, end_date)

            if df.empty:
                return None

            # 주봉/월봉 리샘플링
            if candle_period in ['week', 'month']:
                df = resample_to_period(df, candle_period)

            if len(df) < rsi_period + 1:
                return None

            rsi = calculate_rsi(df['Close'], rsi_period)

            if rsi is None:
                return None

            return {
                "symbol": code,
                "name": name,
                "market": market_label,
                "price": int(df['Close'].iloc[-1]),
                "rsi": rsi,
                "change_percent": round(((df['Close'].iloc[-1] / df['Close'].iloc[-2]) - 1) * 100, 2) if len(df) > 1 else 0,
                "sector": normalize_sector(sector) if sector else None,
                "market_cap": market_cap,
                "market_cap_label": get_market_cap_label(market_cap, is_korean=True) if market_cap else "unknown"
            }

        except Exception as e:
            logger.debug(f"Failed to get RSI for {code}: {e}")
            return None

    def _filter_result(self, result: Dict, rsi_threshold: float,
                        market_cap_filter: str, sector_filter: str,
                        apply_rsi_filter: bool = True) -> bool:
        """결과 필터링"""
        if not result:
            return False

        # RSI 필터 (옵션)
        if apply_rsi_filter and result['rsi'] > rsi_threshold:
            return False

        # 시가총액 필터
        if market_cap_filter != "all":
            if result.get('market_cap_label') != market_cap_filter:
                return False

        # 섹터 필터
        if sector_filter != "all":
            if result.get('sector') != sector_filter:
                return False

        return True

    def scan_oversold_stocks(
        self,
        market: str = "all",
        rsi_threshold: float = 30,
        max_workers: int = 10,
        limit: int = 100,
        market_cap_filter: str = "all",
        sector_filter: str = "all"
    ) -> List[Dict]:
        """
        RSI 과매도 종목 스캔

        Args:
            market: "all", "us", "kr", "kospi", "kosdaq", "nasdaq", "dow"
            rsi_threshold: RSI 기준값 (이하인 종목 필터링)
            max_workers: 병렬 처리 스레드 수
            limit: 각 시장당 최대 스캔 종목 수 (전체 스캔 시 시간 제한)
            market_cap_filter: "all", "large", "mid", "small"
            sector_filter: "all", "technology", "finance", etc.

        Returns:
            과매도 종목 리스트
        """
        results = []

        # NASDAQ 스캔
        if market in ["nasdaq", "us", "all"]:
            nasdaq_symbols = self.get_nasdaq_symbols()[:limit]
            logger.info(f"Scanning {len(nasdaq_symbols)} NASDAQ stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.get_us_stock_rsi, symbol, 14, "NASDAQ"): symbol for symbol in nasdaq_symbols}

                for future in as_completed(futures):
                    result = future.result()
                    if self._filter_result(result, rsi_threshold, market_cap_filter, sector_filter):
                        results.append(result)

        # DOW 30 스캔
        if market in ["dow", "us", "all"]:
            dow_symbols = self.get_dow_symbols()
            logger.info(f"Scanning {len(dow_symbols)} DOW stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {executor.submit(self.get_us_stock_rsi, symbol, 14, "DOW"): symbol for symbol in dow_symbols}

                for future in as_completed(futures):
                    result = future.result()
                    if self._filter_result(result, rsi_threshold, market_cap_filter, sector_filter):
                        results.append(result)

        # KOSPI 스캔
        if market in ["kospi", "kr", "all"]:
            kospi_stocks = self.get_kospi_symbols_detailed()[:limit]
            logger.info(f"Scanning {len(kospi_stocks)} KOSPI stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_kr_stock_rsi,
                        stock['code'], stock['name'], 14, "KOSPI",
                        stock.get('sector'), stock.get('market_cap')
                    ): stock['code'] for stock in kospi_stocks
                }

                for future in as_completed(futures):
                    result = future.result()
                    if self._filter_result(result, rsi_threshold, market_cap_filter, sector_filter):
                        results.append(result)

        # KOSDAQ 스캔
        if market in ["kosdaq", "kr", "all"]:
            kosdaq_stocks = self.get_kosdaq_symbols_detailed()[:limit]
            logger.info(f"Scanning {len(kosdaq_stocks)} KOSDAQ stocks...")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        self.get_kr_stock_rsi,
                        stock['code'], stock['name'], 14, "KOSDAQ",
                        stock.get('sector'), stock.get('market_cap')
                    ): stock['code'] for stock in kosdaq_stocks
                }

                for future in as_completed(futures):
                    result = future.result()
                    if self._filter_result(result, rsi_threshold, market_cap_filter, sector_filter):
                        results.append(result)

        # RSI 오름차순 정렬 (가장 과매도인 종목 먼저)
        results.sort(key=lambda x: x['rsi'])

        logger.info(f"Found {len(results)} oversold stocks")
        return results

    def scan_oversold_stocks_with_progress(
        self,
        market: str = "all",
        rsi_threshold: float = 30,
        max_workers: int = 10,
        limit: int = 100,
        market_cap_filter: str = "all",
        sector_filter: str = "all",
        candle_period: str = "day",
        progress_callback=None,
        result_callback=None,
        start_callback=None,
        cancel_check=None,
        custom_stocks: List[Dict] = None
    ) -> List[Dict]:
        """
        RSI 과매도 종목 스캔 (진행 상황 콜백 지원)

        Args:
            candle_period: 봉 타입 (day, week, month)
            custom_stocks: 추가로 스캔할 커스텀 종목 리스트
        """
        results = []
        all_stocks = []

        # 스캔할 종목 목록 수집
        if market in ["nasdaq", "us", "all"]:
            nasdaq_symbols = self.get_nasdaq_symbols()[:limit]
            all_stocks.extend([{"symbol": s, "market": "NASDAQ", "type": "us"} for s in nasdaq_symbols])

        if market in ["dow", "us", "all"]:
            dow_symbols = self.get_dow_symbols()[:limit]
            all_stocks.extend([{"symbol": s, "market": "DOW", "type": "us"} for s in dow_symbols])

        if market in ["kospi", "kr", "all"]:
            kospi_stocks = self.get_kospi_symbols_detailed()[:limit]
            all_stocks.extend([{
                "symbol": s['code'],
                "name": s['name'],
                "market": "KOSPI",
                "type": "kr",
                "sector": s.get('sector'),
                "market_cap": s.get('market_cap')
            } for s in kospi_stocks])

        if market in ["kosdaq", "kr", "all"]:
            kosdaq_stocks = self.get_kosdaq_symbols_detailed()[:limit]
            all_stocks.extend([{
                "symbol": s['code'],
                "name": s['name'],
                "market": "KOSDAQ",
                "type": "kr",
                "sector": s.get('sector'),
                "market_cap": s.get('market_cap')
            } for s in kosdaq_stocks])

        # 커스텀 종목 추가
        if custom_stocks:
            for cs in custom_stocks:
                # 이미 목록에 있는지 확인
                symbol = cs.get('symbol', '')
                market_name = cs.get('market', '')
                if not any(s['symbol'] == symbol and s['market'] == market_name for s in all_stocks):
                    stock_type = 'kr' if market_name in ['KOSPI', 'KOSDAQ'] else 'us'
                    all_stocks.append({
                        "symbol": symbol,
                        "name": cs.get('name', symbol),
                        "market": market_name,
                        "type": stock_type,
                        "isCustom": True
                    })
            logger.info(f"Added {len(custom_stocks)} custom stocks")

        total = len(all_stocks)
        logger.info(f"Total stocks to scan: {total}")

        # 스캔 시작 알림
        if start_callback:
            start_callback(total)

        # 순차적으로 스캔하면서 진행 상황 보고
        found_count = 0
        for idx, stock in enumerate(all_stocks, 1):
            # 취소 확인
            if cancel_check and cancel_check():
                logger.info("Scan cancelled by client")
                break

            symbol = stock["symbol"]
            market_name = stock["market"]

            # 진행 상황 콜백
            if progress_callback:
                progress_callback(idx, total, symbol, market_name, found_count)

            # RSI 조회
            if stock["type"] == "us":
                result = self.get_us_stock_rsi(symbol, 14, market_name, candle_period)
            else:
                result = self.get_kr_stock_rsi(
                    symbol,
                    stock.get("name", symbol),
                    14,
                    market_name,
                    stock.get("sector"),
                    stock.get("market_cap"),
                    candle_period
                )

            # 필터링 및 결과 추가 (시총/섹터 필터만 적용, RSI는 프론트에서 처리)
            if self._filter_result(result, rsi_threshold, market_cap_filter, sector_filter, apply_rsi_filter=False):
                # RSI 기준 이하인 경우 과매도 표시
                result['is_oversold'] = result['rsi'] <= rsi_threshold
                results.append(result)

                # 과매도 종목 발견 시 콜백
                if result['is_oversold']:
                    found_count += 1
                    if result_callback:
                        result_callback(result)

        # RSI 오름차순 정렬
        results.sort(key=lambda x: x['rsi'])
        logger.info(f"Scanned {len(results)} stocks, {found_count} oversold")
        return results


    def validate_stock(self, symbol: str, market: str) -> Dict:
        """
        종목이 실제로 존재하는지 검증

        Args:
            symbol: 종목 코드/심볼
            market: 시장 (KOSPI, KOSDAQ, NASDAQ, NYSE, DOW)

        Returns:
            {
                "valid": bool,
                "symbol": str,
                "name": str (종목명),
                "market": str,
                "message": str (오류 메시지)
            }
        """
        try:
            symbol = symbol.strip().upper()

            if market in ['KOSPI', 'KOSDAQ']:
                # 한국 주식: FinanceDataReader로 검증
                if market == 'KOSPI':
                    stocks = self.get_kospi_symbols_detailed()
                else:
                    stocks = self.get_kosdaq_symbols_detailed()

                # 종목코드로 검색
                for stock in stocks:
                    if stock['code'] == symbol:
                        return {
                            "valid": True,
                            "symbol": stock['code'],
                            "name": stock['name'],
                            "market": market,
                            "message": "종목이 확인되었습니다."
                        }

                # 종목명으로 검색 (부분 일치)
                for stock in stocks:
                    if symbol in stock['name']:
                        return {
                            "valid": True,
                            "symbol": stock['code'],
                            "name": stock['name'],
                            "market": market,
                            "message": f"'{stock['name']}' 종목이 확인되었습니다."
                        }

                return {
                    "valid": False,
                    "symbol": symbol,
                    "name": None,
                    "market": market,
                    "message": f"{market}에서 '{symbol}'을(를) 찾을 수 없습니다."
                }

            else:
                # 미국 주식: yfinance로 검증
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info

                    # info가 비어있거나 유효하지 않은 경우
                    if not info or info.get('regularMarketPrice') is None:
                        # 히스토리 데이터로 한번 더 확인
                        hist = ticker.history(period="5d")
                        if hist.empty:
                            return {
                                "valid": False,
                                "symbol": symbol,
                                "name": None,
                                "market": market,
                                "message": f"'{symbol}' 종목을 찾을 수 없습니다."
                            }

                    name = info.get('shortName') or info.get('longName') or symbol

                    return {
                        "valid": True,
                        "symbol": symbol,
                        "name": name,
                        "market": market,
                        "message": "종목이 확인되었습니다."
                    }

                except Exception as e:
                    logger.debug(f"Failed to validate {symbol}: {e}")
                    return {
                        "valid": False,
                        "symbol": symbol,
                        "name": None,
                        "market": market,
                        "message": f"'{symbol}' 종목을 찾을 수 없습니다."
                    }

        except Exception as e:
            logger.error(f"Validation error for {symbol}: {e}")
            return {
                "valid": False,
                "symbol": symbol,
                "name": None,
                "market": market,
                "message": f"검증 중 오류가 발생했습니다: {str(e)}"
            }

    def get_stock_detail(self, symbol: str, market: str, period: str = "6mo", interval: str = "1d") -> Dict:
        """
        종목 상세 정보 조회 (차트 데이터 + 재무제표)

        Args:
            symbol: 종목 코드/심볼
            market: 시장 (KOSPI, KOSDAQ, NASDAQ, NYSE)
            period: 데이터 기간 (1mo, 3mo, 6mo, 1y, 2y, 5y)
            interval: 봉 타입 (1h, 4h, 1d, 1wk, 1mo)

        Returns:
            상세 정보 딕셔너리
        """
        try:
            symbol = symbol.strip().upper()
            is_korean = market in ['KOSPI', 'KOSDAQ']

            if is_korean:
                return self._get_kr_stock_detail(symbol, market, period, interval)
            else:
                return self._get_us_stock_detail(symbol, market, period, interval)

        except Exception as e:
            logger.error(f"Failed to get stock detail for {symbol}: {e}")
            return {"error": str(e)}

    def _get_kr_stock_detail(self, symbol: str, market: str, period: str = "6mo", interval: str = "1d") -> Dict:
        """한국 주식 상세 정보"""
        try:
            # 기간 설정
            period_days = {"1mo": 30, "3mo": 90, "6mo": 180, "1y": 365, "2y": 730, "5y": 1825}
            days = period_days.get(period, 180)

            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            # 가격 데이터 조회 (한국 주식은 일봉만 지원)
            df = fdr.DataReader(symbol, start_date, end_date)
            if df.empty:
                return {"error": "가격 데이터를 찾을 수 없습니다."}

            # 주봉/월봉으로 리샘플링
            if interval == '1wk':
                df = resample_to_period(df, 'week')
            elif interval == '1mo':
                df = resample_to_period(df, 'month')
            elif interval in ['1h', '4h']:
                # 한국 주식은 시간봉 미지원 - 일봉으로 대체
                pass

            # 종목명 조회
            stocks = self.get_kospi_symbols_detailed() if market == 'KOSPI' else self.get_kosdaq_symbols_detailed()
            stock_info = next((s for s in stocks if s['code'] == symbol), None)
            name = stock_info['name'] if stock_info else symbol

            # 기술적 지표 계산
            close_prices = df['Close']

            # 볼린저 밴드
            bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(close_prices)

            # MACD
            macd_line, signal_line, macd_histogram = calculate_macd(close_prices)

            # RSI 시리즈
            rsi_series = calculate_rsi_series(close_prices)

            # 이동평균 시리즈
            ma5_series = close_prices.rolling(window=5).mean()
            ma20_series = close_prices.rolling(window=20).mean()
            ma60_series = close_prices.rolling(window=60).mean()
            ma120_series = close_prices.rolling(window=120).mean()

            # 지지선/저항선
            support_resistance = calculate_support_resistance(df)

            # 차트 데이터 - 기술적 지표 포함
            chart_data = []
            for idx, (date, row) in enumerate(df.iterrows()):
                date_str = date.strftime("%Y-%m-%d %H:%M") if interval in ['1h', '4h'] else date.strftime("%Y-%m-%d")
                data_point = {
                    "date": date_str,
                    "open": int(row['Open']),
                    "high": int(row['High']),
                    "low": int(row['Low']),
                    "close": int(row['Close']),
                    "volume": int(row['Volume']),
                    # 볼린저 밴드
                    "bb_upper": int(bb_upper.loc[date]) if pd.notna(bb_upper.loc[date]) else None,
                    "bb_middle": int(bb_middle.loc[date]) if pd.notna(bb_middle.loc[date]) else None,
                    "bb_lower": int(bb_lower.loc[date]) if pd.notna(bb_lower.loc[date]) else None,
                    # MACD
                    "macd": round(float(macd_line.loc[date]), 2) if pd.notna(macd_line.loc[date]) else None,
                    "macd_signal": round(float(signal_line.loc[date]), 2) if pd.notna(signal_line.loc[date]) else None,
                    "macd_histogram": round(float(macd_histogram.loc[date]), 2) if pd.notna(macd_histogram.loc[date]) else None,
                    # RSI
                    "rsi": round(float(rsi_series.loc[date]), 2) if pd.notna(rsi_series.loc[date]) else None,
                    # 이동평균
                    "ma5": int(ma5_series.loc[date]) if pd.notna(ma5_series.loc[date]) else None,
                    "ma20": int(ma20_series.loc[date]) if pd.notna(ma20_series.loc[date]) else None,
                    "ma60": int(ma60_series.loc[date]) if pd.notna(ma60_series.loc[date]) else None,
                    "ma120": int(ma120_series.loc[date]) if pd.notna(ma120_series.loc[date]) else None,
                }
                chart_data.append(data_point)

            # 기본 정보
            current_price = int(df['Close'].iloc[-1])
            prev_close = int(df['Close'].iloc[-2]) if len(df) > 1 else current_price
            change = current_price - prev_close
            change_percent = round((change / prev_close) * 100, 2) if prev_close > 0 else 0

            # 52주 최고/최저 (원본 데이터 기준)
            df_1y = fdr.DataReader(symbol, end_date - timedelta(days=365), end_date)
            high_52w = int(df_1y['High'].max()) if not df_1y.empty else int(df['High'].max())
            low_52w = int(df_1y['Low'].min()) if not df_1y.empty else int(df['Low'].min())

            # 현재 RSI
            from app.services.rsi_calculator import calculate_rsi
            rsi = calculate_rsi(df['Close'], 14)

            # 현재 이동평균
            ma5 = int(df['Close'].tail(5).mean())
            ma20 = int(df['Close'].tail(20).mean())
            ma60 = int(df['Close'].tail(60).mean()) if len(df) >= 60 else None
            ma120 = int(df['Close'].tail(120).mean()) if len(df) >= 120 else None

            # 시가총액 정보
            market_cap = stock_info.get('market_cap') if stock_info else None

            # 뉴스 (한국 주식은 yfinance 티커 형식으로 변환)
            kr_yf_symbol = f"{symbol}.KS" if market == 'KOSPI' else f"{symbol}.KQ"
            news = self.get_stock_news(kr_yf_symbol)

            return {
                "symbol": symbol,
                "name": name,
                "market": market,
                "period": period,
                "interval": interval if interval not in ['1h', '4h'] else '1d',
                "current_price": current_price,
                "change": change,
                "change_percent": change_percent,
                "high_52w": high_52w,
                "low_52w": low_52w,
                "rsi": rsi,
                "market_cap": market_cap,
                "moving_averages": {
                    "ma5": ma5,
                    "ma20": ma20,
                    "ma60": ma60,
                    "ma120": ma120
                },
                "support_resistance": support_resistance,
                "chart_data": chart_data,
                "financials": self._get_kr_financials(symbol),
                "news": news
            }

        except Exception as e:
            logger.error(f"Failed to get KR stock detail: {e}")
            return {"error": str(e)}

    def _get_kr_financials(self, symbol: str) -> Dict:
        """한국 주식 재무제표 (yfinance 사용) - 5개년 데이터"""
        try:
            # 한국 주식은 .KS (KOSPI) 또는 .KQ (KOSDAQ) 접미사 사용
            # 먼저 KOSPI로 시도
            ticker_symbol = f"{symbol}.KS"
            ticker = yf.Ticker(ticker_symbol)

            # 기본 정보 확인
            info = ticker.info
            if not info or info.get('regularMarketPrice') is None:
                # KOSDAQ으로 재시도
                ticker_symbol = f"{symbol}.KQ"
                ticker = yf.Ticker(ticker_symbol)
                info = ticker.info

            if not info or info.get('regularMarketPrice') is None:
                return {
                    "available": False,
                    "message": "재무제표를 찾을 수 없습니다."
                }

            # 손익계산서
            income_stmt = ticker.financials
            # 대차대조표
            balance_sheet = ticker.balance_sheet
            # 현금흐름표
            cashflow = ticker.cashflow

            def format_number(val):
                """숫자를 읽기 쉬운 형태로 변환 (억원 단위)"""
                if val is None:
                    return None
                try:
                    val = float(val)
                    if abs(val) >= 100000000:  # 1억 이상
                        return f"{val / 100000000:.1f}억"
                    elif abs(val) >= 10000:  # 1만 이상
                        return f"{val / 10000:.1f}만"
                    return str(int(val))
                except:
                    return None

            def get_yearly_data(df, keys):
                """DataFrame에서 연도별 데이터 추출"""
                if df is None or df.empty:
                    return []

                yearly_data = []
                # 컬럼은 날짜 (최근 연도부터)
                for col in df.columns[:5]:  # 최대 5개년
                    year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)[:4]
                    year_data = {"year": year}

                    for key, display_name in keys.items():
                        try:
                            if key in df.index:
                                val = df.loc[key, col]
                                if pd.notna(val):
                                    year_data[display_name] = int(val) if abs(val) > 1 else round(float(val), 4)
                                    year_data[f"{display_name}Formatted"] = format_number(val)
                                else:
                                    year_data[display_name] = None
                                    year_data[f"{display_name}Formatted"] = None
                            else:
                                year_data[display_name] = None
                                year_data[f"{display_name}Formatted"] = None
                        except:
                            year_data[display_name] = None
                            year_data[f"{display_name}Formatted"] = None

                    yearly_data.append(year_data)

                return yearly_data

            # 손익계산서 키 매핑
            income_keys = {
                'Total Revenue': 'totalRevenue',
                'Gross Profit': 'grossProfit',
                'Operating Income': 'operatingIncome',
                'Net Income': 'netIncome',
                'EBITDA': 'ebitda'
            }

            # 대차대조표 키 매핑
            balance_keys = {
                'Total Assets': 'totalAssets',
                'Total Liabilities Net Minority Interest': 'totalLiabilities',
                'Stockholders Equity': 'totalEquity',
                'Cash And Cash Equivalents': 'cash',
                'Total Debt': 'totalDebt'
            }

            # 현금흐름표 키 매핑
            cashflow_keys = {
                'Operating Cash Flow': 'operatingCashFlow',
                'Investing Cash Flow': 'investingCashFlow',
                'Financing Cash Flow': 'financingCashFlow',
                'Free Cash Flow': 'freeCashFlow'
            }

            # 배당수익률 처리 (yfinance 값이 불일치할 수 있음)
            # yfinance는 dividendYield를 퍼센트로 반환하기도 하고 (0.4 = 0.4%)
            # 소수로 반환하기도 함 (0.004 = 0.4%)
            div_yield = info.get('dividendYield', 0)
            if div_yield:
                # 100을 곱했을 때 20%를 초과하면 이미 퍼센트 형태로 간주
                if div_yield * 100 > 20:
                    div_yield_pct = round(div_yield, 2)
                else:
                    div_yield_pct = round(div_yield * 100, 2)
            else:
                div_yield_pct = None

            financials_data = {
                "available": True,
                "currency": "KRW",
                # 기본 정보
                "basic": {
                    "marketCap": info.get('marketCap'),
                    "marketCapFormatted": format_number(info.get('marketCap')),
                    "enterpriseValue": info.get('enterpriseValue'),
                    "trailingPE": round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else None,
                    "forwardPE": round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else None,
                    "priceToBook": round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else None,
                    "dividendYield": div_yield_pct,
                },
                # 5개년 손익계산서
                "incomeStatementYearly": get_yearly_data(income_stmt, income_keys),
                # 5개년 대차대조표
                "balanceSheetYearly": get_yearly_data(balance_sheet, balance_keys),
                # 5개년 현금흐름표
                "cashFlowYearly": get_yearly_data(cashflow, cashflow_keys),
                # 수익성 지표 (현재)
                "profitability": {
                    "grossMargin": round(info.get('grossMargins', 0) * 100, 2) if info.get('grossMargins') else None,
                    "operatingMargin": round(info.get('operatingMargins', 0) * 100, 2) if info.get('operatingMargins') else None,
                    "profitMargin": round(info.get('profitMargins', 0) * 100, 2) if info.get('profitMargins') else None,
                    "returnOnAssets": round(info.get('returnOnAssets', 0) * 100, 2) if info.get('returnOnAssets') else None,
                    "returnOnEquity": round(info.get('returnOnEquity', 0) * 100, 2) if info.get('returnOnEquity') else None,
                }
            }

            return financials_data

        except Exception as e:
            logger.error(f"Failed to get KR financials for {symbol}: {e}")
            return {
                "available": False,
                "message": f"재무제표 조회 실패: {str(e)}"
            }

    def _get_us_stock_detail(self, symbol: str, market: str, period: str = "6mo", interval: str = "1d") -> Dict:
        """미국 주식 상세 정보"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # 시간봉의 경우 yfinance 제한 고려
            # 1h: 최대 730일, 4h: 없음(1h로 대체 후 리샘플링)
            yf_interval = interval
            if interval == '4h':
                yf_interval = '1h'  # 4시간봉은 1시간봉을 리샘플링

            # 시간봉의 경우 기간 제한
            yf_period = period
            if interval in ['1h', '4h']:
                # 시간봉은 최대 60일
                if period in ['1y', '2y', '5y']:
                    yf_period = '60d'
                elif period == '6mo':
                    yf_period = '60d'
                elif period == '3mo':
                    yf_period = '60d'

            # 가격 데이터 조회
            hist = ticker.history(period=yf_period, interval=yf_interval)
            if hist.empty:
                return {"error": "가격 데이터를 찾을 수 없습니다."}

            # 4시간봉 리샘플링
            if interval == '4h' and yf_interval == '1h':
                hist = hist.resample('4h').agg({
                    'Open': 'first',
                    'High': 'max',
                    'Low': 'min',
                    'Close': 'last',
                    'Volume': 'sum'
                }).dropna()

            # 기술적 지표 계산
            close_prices = hist['Close']

            # 볼린저 밴드
            bb_middle, bb_upper, bb_lower = calculate_bollinger_bands(close_prices)

            # MACD
            macd_line, signal_line, macd_histogram = calculate_macd(close_prices)

            # RSI 시리즈
            rsi_series = calculate_rsi_series(close_prices)

            # 이동평균 시리즈
            ma5_series = close_prices.rolling(window=5).mean()
            ma20_series = close_prices.rolling(window=20).mean()
            ma60_series = close_prices.rolling(window=60).mean()
            ma120_series = close_prices.rolling(window=120).mean()

            # 지지선/저항선
            support_resistance = calculate_support_resistance(hist)

            # 차트 데이터 - 기술적 지표 포함
            chart_data = []
            for idx, (date, row) in enumerate(hist.iterrows()):
                # 시간봉은 시간 포함, 일봉 이상은 날짜만
                if interval in ['1h', '4h']:
                    date_str = date.strftime("%m/%d %H:%M")
                else:
                    date_str = date.strftime("%Y-%m-%d")

                data_point = {
                    "date": date_str,
                    "open": round(row['Open'], 2),
                    "high": round(row['High'], 2),
                    "low": round(row['Low'], 2),
                    "close": round(row['Close'], 2),
                    "volume": int(row['Volume']),
                    # 볼린저 밴드
                    "bb_upper": round(float(bb_upper.loc[date]), 2) if pd.notna(bb_upper.loc[date]) else None,
                    "bb_middle": round(float(bb_middle.loc[date]), 2) if pd.notna(bb_middle.loc[date]) else None,
                    "bb_lower": round(float(bb_lower.loc[date]), 2) if pd.notna(bb_lower.loc[date]) else None,
                    # MACD
                    "macd": round(float(macd_line.loc[date]), 2) if pd.notna(macd_line.loc[date]) else None,
                    "macd_signal": round(float(signal_line.loc[date]), 2) if pd.notna(signal_line.loc[date]) else None,
                    "macd_histogram": round(float(macd_histogram.loc[date]), 2) if pd.notna(macd_histogram.loc[date]) else None,
                    # RSI
                    "rsi": round(float(rsi_series.loc[date]), 2) if pd.notna(rsi_series.loc[date]) else None,
                    # 이동평균
                    "ma5": round(float(ma5_series.loc[date]), 2) if pd.notna(ma5_series.loc[date]) else None,
                    "ma20": round(float(ma20_series.loc[date]), 2) if pd.notna(ma20_series.loc[date]) else None,
                    "ma60": round(float(ma60_series.loc[date]), 2) if pd.notna(ma60_series.loc[date]) else None,
                    "ma120": round(float(ma120_series.loc[date]), 2) if pd.notna(ma120_series.loc[date]) else None,
                }
                chart_data.append(data_point)

            # 기본 정보
            current_price = round(hist['Close'].iloc[-1], 2)
            prev_close = round(hist['Close'].iloc[-2], 2) if len(hist) > 1 else current_price
            change = round(current_price - prev_close, 2)
            change_percent = round((change / prev_close) * 100, 2) if prev_close > 0 else 0

            # 52주 최고/최저 (항상 일봉 기준)
            hist_1y = ticker.history(period="1y")
            high_52w = round(hist_1y['High'].max(), 2) if not hist_1y.empty else round(hist['High'].max(), 2)
            low_52w = round(hist_1y['Low'].min(), 2) if not hist_1y.empty else round(hist['Low'].min(), 2)

            # 현재 RSI
            from app.services.rsi_calculator import calculate_rsi
            rsi = calculate_rsi(hist['Close'], 14)

            # 현재 이동평균
            ma5 = round(hist['Close'].tail(5).mean(), 2)
            ma20 = round(hist['Close'].tail(20).mean(), 2)
            ma60 = round(hist['Close'].tail(60).mean(), 2) if len(hist) >= 60 else None
            ma120 = round(hist['Close'].tail(120).mean(), 2) if len(hist) >= 120 else None

            # 환율 정보
            exchange_rate = self.get_usd_krw_rate()

            # 뉴스
            news = self.get_stock_news(symbol)

            return {
                "symbol": symbol,
                "name": info.get('shortName') or info.get('longName') or symbol,
                "market": market,
                "period": period,
                "interval": interval,
                "current_price": current_price,
                "change": change,
                "change_percent": change_percent,
                "high_52w": high_52w,
                "low_52w": low_52w,
                "rsi": rsi,
                "market_cap": info.get('marketCap'),
                "exchange_rate": exchange_rate,
                "current_price_krw": round(current_price * exchange_rate) if exchange_rate else None,
                "moving_averages": {
                    "ma5": ma5,
                    "ma20": ma20,
                    "ma60": ma60,
                    "ma120": ma120
                },
                "support_resistance": support_resistance,
                "chart_data": chart_data,
                "financials": self._get_us_financials(ticker, info),
                "news": news
            }

        except Exception as e:
            logger.error(f"Failed to get US stock detail: {e}")
            return {"error": str(e)}

    def _get_us_financials(self, ticker, info: Dict) -> Dict:
        """미국 주식 재무제표 정보 - 5개년 데이터"""
        try:
            # 기업 소개 번역
            description_en = info.get('longBusinessSummary', '')
            description_kr = ''
            if description_en:
                # 500자로 제한 후 번역
                description_kr = translate_to_korean(description_en[:500])

            # 손익계산서
            income_stmt = ticker.financials
            # 대차대조표
            balance_sheet = ticker.balance_sheet
            # 현금흐름표
            cashflow = ticker.cashflow

            def format_number_usd(val):
                """숫자를 읽기 쉬운 형태로 변환 (USD - B/M 단위)"""
                if val is None:
                    return None
                try:
                    val = float(val)
                    if abs(val) >= 1000000000:  # 10억 이상 (Billion)
                        return f"${val / 1000000000:.1f}B"
                    elif abs(val) >= 1000000:  # 100만 이상 (Million)
                        return f"${val / 1000000:.1f}M"
                    elif abs(val) >= 1000:  # 1000 이상 (Thousand)
                        return f"${val / 1000:.1f}K"
                    return f"${int(val)}"
                except:
                    return None

            def get_yearly_data(df, keys):
                """DataFrame에서 연도별 데이터 추출"""
                if df is None or df.empty:
                    return []

                yearly_data = []
                # 컬럼은 날짜 (최근 연도부터)
                for col in df.columns[:5]:  # 최대 5개년
                    year = col.strftime('%Y') if hasattr(col, 'strftime') else str(col)[:4]
                    year_data = {"year": year}

                    for key, display_name in keys.items():
                        try:
                            if key in df.index:
                                val = df.loc[key, col]
                                if pd.notna(val):
                                    year_data[display_name] = int(val) if abs(val) > 1 else round(float(val), 4)
                                    year_data[f"{display_name}Formatted"] = format_number_usd(val)
                                else:
                                    year_data[display_name] = None
                                    year_data[f"{display_name}Formatted"] = None
                            else:
                                year_data[display_name] = None
                                year_data[f"{display_name}Formatted"] = None
                        except:
                            year_data[display_name] = None
                            year_data[f"{display_name}Formatted"] = None

                    yearly_data.append(year_data)

                return yearly_data

            # 손익계산서 키 매핑
            income_keys = {
                'Total Revenue': 'totalRevenue',
                'Gross Profit': 'grossProfit',
                'Operating Income': 'operatingIncome',
                'Net Income': 'netIncome',
                'EBITDA': 'ebitda'
            }

            # 대차대조표 키 매핑
            balance_keys = {
                'Total Assets': 'totalAssets',
                'Total Liabilities Net Minority Interest': 'totalLiabilities',
                'Stockholders Equity': 'totalEquity',
                'Cash And Cash Equivalents': 'cash',
                'Total Debt': 'totalDebt'
            }

            # 현금흐름표 키 매핑
            cashflow_keys = {
                'Operating Cash Flow': 'operatingCashFlow',
                'Investing Cash Flow': 'investingCashFlow',
                'Financing Cash Flow': 'financingCashFlow',
                'Free Cash Flow': 'freeCashFlow'
            }

            # 배당수익률 처리 (yfinance 값이 불일치할 수 있음)
            # yfinance는 dividendYield를 퍼센트로 반환하기도 하고 (0.4 = 0.4%)
            # 소수로 반환하기도 함 (0.004 = 0.4%)
            div_yield = info.get('dividendYield', 0)
            if div_yield:
                # 100을 곱했을 때 20%를 초과하면 이미 퍼센트 형태로 간주
                if div_yield * 100 > 20:
                    div_yield_pct = round(div_yield, 2)
                else:
                    div_yield_pct = round(div_yield * 100, 2)
            else:
                div_yield_pct = None

            return {
                "available": True,
                "currency": "USD",
                # 기본 정보
                "basic": {
                    "marketCap": info.get('marketCap'),
                    "marketCapFormatted": format_number_usd(info.get('marketCap')),
                    "enterpriseValue": info.get('enterpriseValue'),
                    "trailingPE": round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else None,
                    "forwardPE": round(info.get('forwardPE', 0), 2) if info.get('forwardPE') else None,
                    "priceToBook": round(info.get('priceToBook', 0), 2) if info.get('priceToBook') else None,
                    "dividendYield": div_yield_pct,
                },
                # 5개년 손익계산서
                "incomeStatementYearly": get_yearly_data(income_stmt, income_keys),
                # 5개년 대차대조표
                "balanceSheetYearly": get_yearly_data(balance_sheet, balance_keys),
                # 5개년 현금흐름표
                "cashFlowYearly": get_yearly_data(cashflow, cashflow_keys),
                # 수익성 지표 (현재)
                "profitability": {
                    "grossMargin": round(info.get('grossMargins', 0) * 100, 2) if info.get('grossMargins') else None,
                    "operatingMargin": round(info.get('operatingMargins', 0) * 100, 2) if info.get('operatingMargins') else None,
                    "profitMargin": round(info.get('profitMargins', 0) * 100, 2) if info.get('profitMargins') else None,
                    "returnOnAssets": round(info.get('returnOnAssets', 0) * 100, 2) if info.get('returnOnAssets') else None,
                    "returnOnEquity": round(info.get('returnOnEquity', 0) * 100, 2) if info.get('returnOnEquity') else None,
                },
                # 기존 정보도 유지
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "description": description_kr
            }
        except Exception as e:
            logger.error(f"Failed to get US financials: {e}")
            return {"available": False, "message": str(e)}

    def search_stocks(self, query: str, limit: int = 10) -> List[Dict]:
        """
        종목 검색 (한글명, 영문명, 심볼로 검색)

        Args:
            query: 검색어
            limit: 최대 결과 수

        Returns:
            검색 결과 리스트 [{symbol, name, market}, ...]
        """
        if not query or len(query) < 1:
            return []

        query_upper = query.upper()
        query_lower = query.lower()
        results = []

        try:
            # 한국 주식 검색 (KOSPI)
            kospi_stocks = self.get_kospi_symbols_detailed()
            for stock in kospi_stocks:
                code = stock.get('code', '')
                name = stock.get('name', '')
                # 코드나 이름에 검색어가 포함되면 추가
                if query in code or query in name or query_lower in name.lower():
                    results.append({
                        'symbol': code,
                        'name': name,
                        'market': 'KOSPI'
                    })

            # 한국 주식 검색 (KOSDAQ)
            kosdaq_stocks = self.get_kosdaq_symbols_detailed()
            for stock in kosdaq_stocks:
                code = stock.get('code', '')
                name = stock.get('name', '')
                if query in code or query in name or query_lower in name.lower():
                    results.append({
                        'symbol': code,
                        'name': name,
                        'market': 'KOSDAQ'
                    })

            # 미국 주식 검색 (NASDAQ + NYSE)
            us_symbols = self.get_us_symbols()
            for symbol in us_symbols:
                if query_upper in symbol:
                    # 심볼이 매칭되면 추가 (이름은 나중에 가져옴)
                    results.append({
                        'symbol': symbol,
                        'name': symbol,  # 일단 심볼을 이름으로
                        'market': 'US'
                    })

            # 미국 주식 영문명 검색을 위한 일부 인기 종목 매핑
            us_stock_names = {
                'AAPL': 'Apple Inc.',
                'MSFT': 'Microsoft Corporation',
                'GOOGL': 'Alphabet Inc.',
                'GOOG': 'Alphabet Inc.',
                'AMZN': 'Amazon.com Inc.',
                'META': 'Meta Platforms Inc.',
                'TSLA': 'Tesla Inc.',
                'NVDA': 'NVIDIA Corporation',
                'NFLX': 'Netflix Inc.',
                'AMD': 'Advanced Micro Devices',
                'INTC': 'Intel Corporation',
                'CSCO': 'Cisco Systems',
                'ADBE': 'Adobe Inc.',
                'PYPL': 'PayPal Holdings',
                'CRM': 'Salesforce Inc.',
                'ORCL': 'Oracle Corporation',
                'IBM': 'IBM Corporation',
                'QCOM': 'Qualcomm Inc.',
                'TXN': 'Texas Instruments',
                'AVGO': 'Broadcom Inc.',
                'COST': 'Costco Wholesale',
                'PEP': 'PepsiCo Inc.',
                'KO': 'Coca-Cola Company',
                'WMT': 'Walmart Inc.',
                'DIS': 'Walt Disney Company',
                'V': 'Visa Inc.',
                'MA': 'Mastercard Inc.',
                'JPM': 'JPMorgan Chase',
                'BAC': 'Bank of America',
                'GS': 'Goldman Sachs',
                'MS': 'Morgan Stanley',
                'UBER': 'Uber Technologies',
                'LYFT': 'Lyft Inc.',
                'ABNB': 'Airbnb Inc.',
                'SQ': 'Block Inc.',
                'SHOP': 'Shopify Inc.',
                'SPOT': 'Spotify Technology',
                'ZM': 'Zoom Video',
                'COIN': 'Coinbase Global',
                'NOW': 'ServiceNow Inc.',
                'COP': 'ConocoPhillips',
                'CVX': 'Chevron Corporation',
                'XOM': 'Exxon Mobil',
                'JNJ': 'Johnson & Johnson',
                'PFE': 'Pfizer Inc.',
                'UNH': 'UnitedHealth Group',
                'HD': 'Home Depot',
                'NKE': 'Nike Inc.',
                'MCD': 'McDonald\'s Corporation',
                'BA': 'Boeing Company',
                'CAT': 'Caterpillar Inc.',
            }

            # 영문 이름으로 검색
            for symbol, name in us_stock_names.items():
                if query_lower in name.lower() or query_upper in symbol:
                    # 이미 결과에 있는지 확인
                    exists = any(r['symbol'] == symbol for r in results)
                    if not exists:
                        results.append({
                            'symbol': symbol,
                            'name': name,
                            'market': 'US'
                        })
                    else:
                        # 이름 업데이트
                        for r in results:
                            if r['symbol'] == symbol:
                                r['name'] = name
                                break

            # 정확히 일치하는 것을 앞으로
            def sort_key(item):
                symbol = item['symbol']
                name = item['name']
                # 정확히 일치하면 0, 시작하면 1, 포함하면 2
                if symbol == query_upper or name == query:
                    return 0
                if symbol.startswith(query_upper) or name.startswith(query):
                    return 1
                return 2

            results.sort(key=sort_key)

            return results[:limit]

        except Exception as e:
            logger.error(f"Failed to search stocks: {e}")
            return []

    def run_backtest(self, symbol: str, market: str, strategy: str = "rsi",
                     buy_rsi: float = 30, sell_rsi: float = 70,
                     period: str = "2y", initial_capital: float = 10000000,
                     buy_ma: int = 0, sell_ma: int = 0) -> Dict:
        """
        백테스트 시뮬레이터

        Args:
            symbol: 종목 코드/심볼
            market: 시장 (KOSPI, KOSDAQ, NASDAQ, NYSE 등)
            strategy: 전략 타입 (rsi, ma_cross, rsi_ma)
            buy_rsi: RSI 매수 기준 (이하일 때 매수)
            sell_rsi: RSI 매도 기준 (이상일 때 매도)
            period: 백테스트 기간 (1y, 2y, 3y, 5y)
            initial_capital: 초기 투자금
            buy_ma: 이동평균 매수 기준 (단기선)
            sell_ma: 이동평균 매도 기준 (장기선)

        Returns:
            백테스트 결과 딕셔너리
        """
        try:
            is_korean = market in ['KOSPI', 'KOSDAQ']

            # 데이터 가져오기
            if is_korean:
                yf_symbol = f"{symbol}.KS" if market == 'KOSPI' else f"{symbol}.KQ"
            else:
                yf_symbol = symbol

            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period)

            if df.empty or len(df) < 30:
                return {"error": "데이터가 부족합니다. 다른 기간을 선택해주세요."}

            # 미국 주식: 원화 -> 달러 변환
            exchange_rate = None
            if not is_korean:
                exchange_rate = self.get_usd_krw_rate()
                capital_usd = initial_capital / exchange_rate
                initial_capital_display = initial_capital  # 원화 표시용
                initial_capital = capital_usd  # 실제 거래는 달러로

            # 기술적 지표 계산
            df['RSI'] = calculate_rsi_series(df['Close'], period=14)
            df['MA5'] = df['Close'].rolling(window=5).mean()
            df['MA20'] = df['Close'].rolling(window=20).mean()
            df['MA60'] = df['Close'].rolling(window=60).mean()

            # NaN 제거
            df = df.dropna(subset=['RSI', 'Close'])

            if len(df) < 10:
                return {"error": "유효한 데이터가 부족합니다. 다른 기간을 선택해주세요."}

            # numpy -> python 변환 헬퍼
            def safe_float(val):
                if val is None:
                    return 0.0
                v = float(val)
                if math.isnan(v) or math.isinf(v):
                    return 0.0
                return v

            # 시뮬레이션
            capital = initial_capital
            position = 0  # 보유 수량
            buy_price = 0
            trades = []
            portfolio_values = []
            in_position = False

            for i, (date, row) in enumerate(df.iterrows()):
                current_price = safe_float(row['Close'])
                current_rsi = safe_float(row['RSI'])
                if current_price <= 0:
                    continue
                portfolio_value = capital + (position * current_price)

                # 포트폴리오 가치 기록 (차트용)
                date_str = date.strftime('%Y-%m-%d')
                portfolio_values.append({
                    'date': date_str,
                    'value': round(safe_float(portfolio_value), 2),
                    'price': round(safe_float(current_price), 2),
                    'rsi': round(current_rsi, 2)
                })

                # 매수 신호
                buy_signal = False
                sell_signal = False

                if strategy == 'rsi':
                    buy_signal = current_rsi <= buy_rsi and not in_position
                    sell_signal = current_rsi >= sell_rsi and in_position
                elif strategy == 'ma_cross':
                    if i > 0:
                        prev_ma5 = safe_float(df['MA5'].iloc[i - 1])
                        prev_ma20 = safe_float(df['MA20'].iloc[i - 1])
                        curr_ma5 = safe_float(row['MA5'])
                        curr_ma20 = safe_float(row['MA20'])
                        if prev_ma5 > 0 and prev_ma20 > 0 and curr_ma5 > 0 and curr_ma20 > 0:
                            # 골든크로스 매수
                            buy_signal = prev_ma5 <= prev_ma20 and curr_ma5 > curr_ma20 and not in_position
                            # 데드크로스 매도
                            sell_signal = prev_ma5 >= prev_ma20 and curr_ma5 < curr_ma20 and in_position
                elif strategy == 'rsi_ma':
                    curr_ma5 = safe_float(row['MA5'])
                    curr_ma20 = safe_float(row['MA20'])
                    if curr_ma5 > 0 and curr_ma20 > 0:
                        buy_signal = current_rsi <= buy_rsi and curr_ma5 > curr_ma20 and not in_position
                        sell_signal = (current_rsi >= sell_rsi or (curr_ma5 < curr_ma20)) and in_position

                # 매수 실행
                if buy_signal and capital > 0:
                    shares = int(capital / current_price)
                    if shares > 0:
                        cost = shares * current_price
                        capital -= cost
                        position = shares
                        buy_price = current_price
                        in_position = True
                        trades.append({
                            'type': 'BUY',
                            'date': date_str,
                            'price': round(current_price, 2),
                            'shares': shares,
                            'amount': round(cost, 2),
                            'rsi': round(current_rsi, 2),
                            'profit': None,
                            'profit_percent': None
                        })

                # 매도 실행
                elif sell_signal and position > 0:
                    revenue = position * current_price
                    profit = revenue - (position * buy_price)
                    profit_percent = ((current_price - buy_price) / buy_price) * 100
                    capital += revenue
                    trades.append({
                        'type': 'SELL',
                        'date': date_str,
                        'price': round(current_price, 2),
                        'shares': position,
                        'amount': round(revenue, 2),
                        'rsi': round(current_rsi, 2),
                        'profit': round(profit, 2),
                        'profit_percent': round(profit_percent, 2)
                    })
                    position = 0
                    buy_price = 0
                    in_position = False

            # 최종 포트폴리오 가치
            final_price = safe_float(df['Close'].iloc[-1])
            final_value = capital + (position * final_price)
            total_return = final_value - initial_capital
            total_return_percent = (total_return / initial_capital) * 100 if initial_capital > 0 else 0

            # Buy & Hold 수익률 비교
            first_price = safe_float(df['Close'].iloc[0])
            buy_hold_return = ((final_price - first_price) / first_price) * 100 if first_price > 0 else 0

            # 거래 통계
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            winning_trades = [t for t in sell_trades if t['profit'] and t['profit'] > 0]
            losing_trades = [t for t in sell_trades if t['profit'] and t['profit'] <= 0]

            win_rate = (len(winning_trades) / len(sell_trades) * 100) if sell_trades else 0
            avg_profit = safe_float(sum(t['profit'] for t in sell_trades) / len(sell_trades)) if sell_trades else 0
            avg_win = safe_float(sum(t['profit'] for t in winning_trades) / len(winning_trades)) if winning_trades else 0
            avg_loss = safe_float(sum(t['profit'] for t in losing_trades) / len(losing_trades)) if losing_trades else 0

            # 최대 낙폭 (MDD) 계산
            peak = initial_capital
            max_drawdown = 0
            for pv in portfolio_values:
                if pv['value'] > peak:
                    peak = pv['value']
                drawdown = ((peak - pv['value']) / peak) * 100
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

            # 미국 주식: 달러 -> 원화 변환 (표시용)
            currency = 'KRW'
            display_initial = initial_capital
            display_final = final_value
            display_return = total_return
            display_holding_value = (position * final_price) if position > 0 else 0

            if not is_korean and exchange_rate:
                currency = 'USD'
                display_initial = round(initial_capital_display)  # 원래 원화 입력값
                display_final = round(final_value * exchange_rate)
                display_return = round(total_return * exchange_rate)
                display_holding_value = round(display_holding_value * exchange_rate)
                # 거래 금액도 원화로 변환
                for t in trades:
                    t['amount_krw'] = round(t['amount'] * exchange_rate)
                    if t['profit'] is not None:
                        t['profit_krw'] = round(t['profit'] * exchange_rate)
                # 포트폴리오 가치도 원화로 변환
                for pv in portfolio_values:
                    pv['value_krw'] = round(pv['value'] * exchange_rate)

            # 포트폴리오 가치 데이터 간소화 (차트용, 최대 200포인트)
            step = max(1, len(portfolio_values) // 200)
            chart_data = portfolio_values[::step]
            if portfolio_values[-1] not in chart_data:
                chart_data.append(portfolio_values[-1])

            return {
                "symbol": symbol,
                "market": market,
                "strategy": strategy,
                "period": period,
                "currency": currency,
                "exchange_rate": exchange_rate,
                "is_korean": is_korean,
                "initial_capital": round(display_initial),
                "final_value": round(display_final),
                "total_return": round(display_return),
                "total_return_percent": round(total_return_percent, 2),
                "buy_hold_return_percent": round(buy_hold_return, 2),
                "total_trades": len(trades),
                "sell_trades": len(sell_trades),
                "winning_trades": len(winning_trades),
                "losing_trades": len(losing_trades),
                "win_rate": round(win_rate, 2),
                "avg_profit": round(avg_profit, 2),
                "avg_win": round(avg_win, 2),
                "avg_loss": round(avg_loss, 2),
                "max_drawdown": round(max_drawdown, 2),
                "still_holding": position > 0,
                "holding_shares": position,
                "holding_value": round(display_holding_value),
                "trades": trades,
                "chart_data": chart_data,
                "data_start": df.index[0].strftime('%Y-%m-%d'),
                "data_end": df.index[-1].strftime('%Y-%m-%d'),
                "parameters": {
                    "buy_rsi": buy_rsi,
                    "sell_rsi": sell_rsi,
                    "strategy": strategy
                }
            }

        except Exception as e:
            logger.error(f"Backtest error for {symbol}: {e}")
            return {"error": f"백테스트 실행 중 오류가 발생했습니다: {str(e)}"}

    def get_stock_quote(self, symbol: str, market: str) -> Dict:
        """
        실시간 시세 경량 조회 (가격, 등락, 거래량만)

        Args:
            symbol: 종목 코드/심볼
            market: 시장 (KOSPI, KOSDAQ, NASDAQ, NYSE)

        Returns:
            현재가, 변동, 변동률, 거래량
        """
        try:
            symbol = symbol.strip().upper()
            is_korean = market in ['KOSPI', 'KOSDAQ']

            if is_korean:
                ticker_symbol = f"{symbol}.KS" if market == 'KOSPI' else f"{symbol}.KQ"
            else:
                ticker_symbol = symbol

            ticker = yf.Ticker(ticker_symbol)
            info = ticker.fast_info

            current_price = float(info.get('lastPrice', 0) or info.get('last_price', 0))
            previous_close = float(info.get('previousClose', 0) or info.get('previous_close', 0))

            if current_price == 0:
                # fast_info 실패 시 히스토리에서 가져오기
                hist = ticker.history(period='2d')
                if len(hist) >= 1:
                    current_price = float(hist['Close'].iloc[-1])
                    if len(hist) >= 2:
                        previous_close = float(hist['Close'].iloc[-2])

            change = current_price - previous_close if previous_close else 0
            change_percent = (change / previous_close * 100) if previous_close else 0

            volume = int(info.get('lastVolume', 0) or info.get('last_volume', 0))

            result = {
                "symbol": symbol,
                "market": market,
                "current_price": current_price,
                "change": change,
                "change_percent": round(change_percent, 2),
                "volume": volume,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # 미국 주식: 원화 환산
            if not is_korean:
                rate = self.get_usd_krw_rate()
                if rate:
                    result["current_price_krw"] = round(current_price * rate)
                    result["exchange_rate"] = rate

            return result

        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            return {"error": str(e)}


# 싱글톤 인스턴스
stock_service = StockDataService()

import yfinance as yf
import FinanceDataReader as fdr
import pandas as pd
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
                "financials": self._get_kr_financials(symbol)
            }

        except Exception as e:
            logger.error(f"Failed to get KR stock detail: {e}")
            return {"error": str(e)}

    def _get_kr_financials(self, symbol: str) -> Dict:
        """한국 주식 재무제표 (기본 정보만)"""
        # FinanceDataReader는 재무제표를 직접 제공하지 않음
        # 기본 정보만 반환
        return {
            "available": False,
            "message": "한국 주식 재무제표는 현재 지원되지 않습니다."
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
                "moving_averages": {
                    "ma5": ma5,
                    "ma20": ma20,
                    "ma60": ma60,
                    "ma120": ma120
                },
                "support_resistance": support_resistance,
                "chart_data": chart_data,
                "financials": self._get_us_financials(info)
            }

        except Exception as e:
            logger.error(f"Failed to get US stock detail: {e}")
            return {"error": str(e)}

    def _get_us_financials(self, info: Dict) -> Dict:
        """미국 주식 재무제표 정보"""
        try:
            # 기업 소개 번역
            description_en = info.get('longBusinessSummary', '')
            description_kr = ''
            if description_en:
                # 500자로 제한 후 번역
                description_kr = translate_to_korean(description_en[:500])

            return {
                "available": True,
                "per": info.get('trailingPE'),  # PER
                "pbr": info.get('priceToBook'),  # PBR
                "eps": info.get('trailingEps'),  # EPS
                "roe": info.get('returnOnEquity'),  # ROE (소수점)
                "revenue": info.get('totalRevenue'),  # 매출
                "profit": info.get('netIncomeToCommon'),  # 순이익
                "dividend_yield": info.get('dividendYield'),  # 배당수익률
                "debt_to_equity": info.get('debtToEquity'),  # 부채비율
                "current_ratio": info.get('currentRatio'),  # 유동비율
                "profit_margin": info.get('profitMargins'),  # 이익률
                "sector": info.get('sector'),
                "industry": info.get('industry'),
                "description": description_kr  # 한국어로 번역된 기업 설명
            }
        except Exception as e:
            logger.error(f"Failed to get US financials: {e}")
            return {"available": False, "message": str(e)}


# 싱글톤 인스턴스
stock_service = StockDataService()

import pandas as pd
import numpy as np
from typing import Optional


def calculate_rsi(prices: pd.Series, period: int = 14) -> Optional[float]:
    """
    RSI(Relative Strength Index) 계산

    Args:
        prices: 종가 시리즈 (최소 period + 1 개의 데이터 필요)
        period: RSI 계산 기간 (기본 14일)

    Returns:
        RSI 값 (0-100) 또는 데이터 부족 시 None
    """
    if len(prices) < period + 1:
        return None

    # 가격 변화량 계산
    delta = prices.diff()

    # 상승/하락 분리
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    # 평균 상승/하락 계산 (EMA 방식)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    # RS 및 RSI 계산
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # 마지막 RSI 값 반환
    last_rsi = rsi.iloc[-1]

    if pd.isna(last_rsi):
        return None

    return round(float(last_rsi), 2)


def calculate_rsi_series(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI 시리즈 계산 (차트용)

    Args:
        prices: 종가 시리즈
        period: RSI 계산 기간

    Returns:
        RSI 시리즈
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

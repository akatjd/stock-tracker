"""
Claude Agent SDK용 MCP 도구 정의 (Pro 요금제 백엔드 전용).

- 기존 LangGraph/Ollama 경로(agent.py)와 완전히 분리됨
- 8001 백엔드 API를 합성해서 호출 — 백엔드 변경 없음
- 이 파일을 삭제하면 Claude 백엔드 기능이 완전히 제거됨
"""

import json
import math
import statistics
from typing import Any

import requests

from claude_agent_sdk import tool, create_sdk_mcp_server

BASE_URL = "http://localhost:8001/api/v1"


def _ok(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": text}]}


def _err(text: str) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": f"오류: {text}"}], "is_error": True}


# ── 기존 6개 도구 ─────────────────────────────────────────────────────────────

@tool("get_stock_rsi", "종목의 현재 RSI/현재가/등락률 조회", {"symbol": str, "market": str})
async def get_stock_rsi(args):
    try:
        r = requests.get(
            f"{BASE_URL}/stock/detail/{args['symbol']}",
            params={"market": args["market"], "period": "3mo", "interval": "1d"},
            timeout=15,
        )
        d = r.json()
        if "error" in d:
            return _err(d["error"])
        return _ok(
            f"{d.get('name', args['symbol'])} ({args['symbol']}, {args['market']})\n"
            f"  현재가: {d.get('current_price'):,}\n"
            f"  등락률: {d.get('change_percent'):+.2f}%\n"
            f"  RSI(14): {d.get('rsi'):.1f}"
        )
    except Exception as e:
        return _err(str(e))


@tool(
    "scan_oversold",
    "RSI 과매도 종목 빠른 스캔 (limit은 50 이하 권장)",
    {"market": str, "rsi_threshold": float, "limit": int},
)
async def scan_oversold(args):
    try:
        r = requests.get(
            f"{BASE_URL}/scan/oversold",
            params={
                "market": args.get("market", "all"),
                "rsi_threshold": args.get("rsi_threshold", 30),
                "limit": args.get("limit", 50),
            },
            timeout=120,
        )
        stocks = r.json().get("stocks", [])
        if not stocks:
            return _ok(f"{args.get('market', 'all')} 시장에서 RSI 조건 만족 종목 없음")
        lines = [f"RSI {args.get('rsi_threshold', 30)} 이하 ({len(stocks)}개):"]
        for s in stocks[:15]:
            lines.append(f"  {s['name']}({s['symbol']}) RSI {s['rsi']:.1f} | {s.get('current_price', 'N/A')}")
        if len(stocks) > 15:
            lines.append(f"  ... 외 {len(stocks)-15}개")
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@tool("search_stock", "종목명/코드로 종목 검색", {"query": str})
async def search_stock(args):
    try:
        r = requests.get(
            f"{BASE_URL}/stock/search",
            params={"query": args["query"], "limit": 10},
            timeout=10,
        )
        results = r.json().get("results", [])
        if not results:
            return _ok(f"'{args['query']}' 검색 결과 없음")
        lines = [f"'{args['query']}' 검색 결과:"]
        for s in results:
            lines.append(f"  {s['name']} | {s['symbol']} | {s['market']}")
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@tool(
    "run_backtest",
    "단일 전략 백테스트 (rsi/ma_cross/rsi_ma)",
    {"symbol": str, "market": str, "strategy": str, "buy_rsi": float, "sell_rsi": float, "period": str},
)
async def run_backtest(args):
    try:
        r = requests.get(
            f"{BASE_URL}/stock/{args['symbol']}/backtest",
            params={
                "market": args["market"],
                "strategy": args.get("strategy", "rsi"),
                "buy_rsi": args.get("buy_rsi", 30),
                "sell_rsi": args.get("sell_rsi", 70),
                "period": args.get("period", "2y"),
                "initial_capital": 10000000,
            },
            timeout=30,
        )
        d = r.json()
        if "error" in d:
            return _err(d["error"])
        cur = "₩" if d.get("is_korean") else "$"
        ret = d.get("total_return_percent", 0)
        bh = d.get("buy_hold_return_percent", 0)
        return _ok(
            f"{d.get('symbol')} {args.get('strategy', 'rsi').upper()} ({d.get('data_start')} ~ {d.get('data_end')})\n"
            f"  전략 수익률:    {ret:+.1f}%\n"
            f"  Buy & Hold:    {bh:+.1f}%\n"
            f"  우위:          {'✅ 전략' if ret > bh else '❌ Buy&Hold'}\n"
            f"  거래 횟수:     {d.get('total_trades')}회\n"
            f"  승률:          {d.get('win_rate'):.1f}%\n"
            f"  최대 낙폭:     {d.get('max_drawdown'):.1f}%\n"
            f"  최종 자산:     {cur}{d.get('final_value', 0):,.0f}"
        )
    except Exception as e:
        return _err(str(e))


@tool("get_news", "종목 최신 뉴스 (한국=네이버, 미국=Yahoo)", {"symbol": str, "market": str, "limit": int})
async def get_news(args):
    try:
        r = requests.get(
            f"{BASE_URL}/stock/detail/{args['symbol']}",
            params={"market": args["market"], "period": "1mo", "interval": "1d"},
            timeout=15,
        )
        d = r.json()
        if "error" in d:
            return _err(d["error"])
        news = d.get("news", [])[: args.get("limit", 5)]
        if not news:
            return _ok(f"{args['symbol']} 관련 뉴스 없음")
        lines = [f"{d.get('name', args['symbol'])} 최신 뉴스 ({len(news)}건):"]
        for n in news:
            pub = (n.get("pubDate") or "")[:16]
            lines.append(f"  [{pub}] {n.get('title', '')} — {n.get('publisher', '')}")
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@tool("get_financials", "재무지표 (PER/PBR/ROE/매출/영업이익)", {"symbol": str, "market": str})
async def get_financials(args):
    try:
        r = requests.get(
            f"{BASE_URL}/stock/detail/{args['symbol']}",
            params={"market": args["market"], "period": "1mo", "interval": "1d"},
            timeout=15,
        )
        d = r.json()
        if "error" in d:
            return _err(d["error"])
        f = d.get("financials", {})
        if not f.get("available"):
            return _err(f"{args['symbol']} 재무 데이터 없음")
        basic = f.get("basic", {})
        prof = f.get("profitability", {})
        income = (f.get("incomeStatementYearly") or [{}])[0]
        per = basic.get("trailingPE") or basic.get("forwardPE")
        lines = [f"{d.get('name', args['symbol'])} 재무지표"]
        if per:
            lines.append(f"  PER: {per:.1f}배")
        if basic.get("priceToBook"):
            lines.append(f"  PBR: {basic['priceToBook']:.2f}배")
        if basic.get("dividendYield"):
            lines.append(f"  배당수익률: {basic['dividendYield']:.2f}%")
        lines.append(f"  시가총액: {basic.get('marketCapFormatted', 'N/A')}")
        if prof.get("grossMargin"):
            lines.append(f"  매출총이익률: {prof['grossMargin']:.1f}%")
        if prof.get("operatingMargin"):
            lines.append(f"  영업이익률: {prof['operatingMargin']:.1f}%")
        if prof.get("returnOnEquity"):
            lines.append(f"  ROE: {prof['returnOnEquity']:.1f}%")
        if income.get("year"):
            lines.append(f"  [{income['year']}] 매출 {income.get('totalRevenueFormatted', '?')} / 영업익 {income.get('operatingIncomeFormatted', '?')}")
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


# ── 신규 퀀트 도구 5개 ────────────────────────────────────────────────────────

def _detail(symbol: str, market: str, period: str = "6mo") -> dict:
    r = requests.get(
        f"{BASE_URL}/stock/detail/{symbol}",
        params={"market": market, "period": period, "interval": "1d"},
        timeout=15,
    )
    return r.json()


@tool(
    "compare_stocks",
    "여러 종목 한 번에 비교 (RSI/현재가/등락/PER/ROE). symbols는 콤마 구분.",
    {"symbols": str, "market": str},
)
async def compare_stocks(args):
    try:
        symbols = [s.strip() for s in args["symbols"].split(",") if s.strip()]
        market = args["market"]
        rows = []
        for sym in symbols[:8]:
            d = _detail(sym, market, "3mo")
            if "error" in d:
                rows.append(f"  {sym}: 조회 실패 ({d['error']})")
                continue
            f = d.get("financials", {}).get("basic", {})
            prof = d.get("financials", {}).get("profitability", {})
            per = f.get("trailingPE") or f.get("forwardPE")
            rows.append(
                f"  {d.get('name', sym):12} ({sym:6}) "
                f"가{d.get('current_price', 0):>10,} "
                f"등락{d.get('change_percent', 0):+6.2f}% "
                f"RSI{d.get('rsi', 0):5.1f} "
                f"PER{(per or 0):5.1f} "
                f"ROE{(prof.get('returnOnEquity') or 0):5.1f}%"
            )
        if not rows:
            return _err("비교할 종목 없음")
        return _ok(f"{market} 종목 비교:\n" + "\n".join(rows))
    except Exception as e:
        return _err(str(e))


@tool(
    "compare_strategies",
    "한 종목에 RSI / MA_Cross / RSI+MA 3가지 전략을 모두 백테스트하고 최적 전략 추천",
    {"symbol": str, "market": str, "period": str},
)
async def compare_strategies(args):
    try:
        out = []
        best = None
        for strat in ["rsi", "ma_cross", "rsi_ma"]:
            r = requests.get(
                f"{BASE_URL}/stock/{args['symbol']}/backtest",
                params={
                    "market": args["market"],
                    "strategy": strat,
                    "buy_rsi": 30,
                    "sell_rsi": 70,
                    "period": args.get("period", "2y"),
                    "initial_capital": 10000000,
                },
                timeout=30,
            )
            d = r.json()
            if "error" in d:
                out.append(f"  {strat:9}: 실패 ({d['error']})")
                continue
            ret = d.get("total_return_percent", 0)
            out.append(
                f"  {strat:9}: 수익 {ret:+6.1f}% | 승률 {d.get('win_rate', 0):4.1f}% | "
                f"MDD {d.get('max_drawdown', 0):4.1f}% | 거래 {d.get('total_trades', 0)}회"
            )
            if best is None or ret > best[1]:
                best = (strat, ret, d.get("buy_hold_return_percent", 0))

        if not best:
            return _err("백테스트 전부 실패")
        bh = best[2]
        verdict = f"\n  ➜ 최우수: {best[0].upper()} ({best[1]:+.1f}%) vs Buy&Hold {bh:+.1f}%"
        return _ok(f"{args['symbol']} 전략 비교 ({args.get('period', '2y')}):\n" + "\n".join(out) + verdict)
    except Exception as e:
        return _err(str(e))


@tool(
    "correlation",
    "두 종목의 일일 수익률 상관계수 (-1~+1). 분산투자/페어트레이딩 판단용",
    {"symbol1": str, "market1": str, "symbol2": str, "market2": str, "period": str},
)
async def correlation(args):
    try:
        d1 = _detail(args["symbol1"], args["market1"], args.get("period", "6mo"))
        d2 = _detail(args["symbol2"], args["market2"], args.get("period", "6mo"))
        if "error" in d1 or "error" in d2:
            return _err("종목 데이터 조회 실패")

        def returns(d):
            closes = [c.get("close") for c in (d.get("chart_data") or []) if c.get("close")]
            return [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]

        r1, r2 = returns(d1), returns(d2)
        n = min(len(r1), len(r2))
        if n < 20:
            return _err(f"데이터 부족 (n={n})")
        r1, r2 = r1[-n:], r2[-n:]
        m1, m2 = statistics.mean(r1), statistics.mean(r2)
        num = sum((a - m1) * (b - m2) for a, b in zip(r1, r2))
        den = math.sqrt(sum((a - m1) ** 2 for a in r1) * sum((b - m2) ** 2 for b in r2))
        corr = num / den if den else 0.0
        verdict = (
            "강한 양의 상관 (같이 움직임)" if corr > 0.7
            else "약한 양의 상관" if corr > 0.3
            else "무상관 (분산 효과 큼)" if abs(corr) <= 0.3
            else "약한 음의 상관" if corr > -0.7
            else "강한 음의 상관 (반대 방향)"
        )
        return _ok(
            f"{d1.get('name')} ↔ {d2.get('name')} 상관계수\n"
            f"  기간: {args.get('period', '6mo')} (n={n}일)\n"
            f"  ρ = {corr:+.3f}\n"
            f"  해석: {verdict}"
        )
    except Exception as e:
        return _err(str(e))


@tool(
    "momentum_rank",
    "여러 종목을 모멘텀 점수로 정렬. 점수 = (1개월 수익률) - (RSI 50으로부터 거리). symbols는 콤마 구분.",
    {"symbols": str, "market": str},
)
async def momentum_rank(args):
    try:
        symbols = [s.strip() for s in args["symbols"].split(",") if s.strip()][:10]
        scored = []
        for sym in symbols:
            d = _detail(sym, args["market"], "3mo")
            if "error" in d:
                continue
            chart = d.get("chart_data") or []
            if len(chart) < 22:
                continue
            cur = chart[-1].get("close")
            prev = chart[-22].get("close")
            if not cur or not prev:
                continue
            ret_1m = (cur - prev) / prev * 100
            rsi = d.get("rsi", 50) or 50
            score = ret_1m - abs(rsi - 50) * 0.5
            scored.append((d.get("name", sym), sym, ret_1m, rsi, score))

        if not scored:
            return _err("유효한 종목 없음")
        scored.sort(key=lambda x: x[4], reverse=True)
        lines = ["모멘텀 랭킹 (1개월 수익률 - RSI 편차 보정):"]
        for i, (name, sym, ret, rsi, score) in enumerate(scored, 1):
            mark = "🚀" if i == 1 else "  "
            lines.append(f"  {mark}{i}. {name:12} ({sym:6}) 1M{ret:+6.2f}% RSI{rsi:5.1f} 점수{score:+6.2f}")
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


@tool(
    "screen_quality_oversold",
    "RSI 과매도 + 재무 우량(ROE 높고 PER 낮음) 종목 스크리닝. 가치+모멘텀 결합.",
    {"market": str, "rsi_threshold": float, "min_roe": float, "max_per": float, "limit": int},
)
async def screen_quality_oversold(args):
    try:
        r = requests.get(
            f"{BASE_URL}/scan/oversold",
            params={
                "market": args.get("market", "kr"),
                "rsi_threshold": args.get("rsi_threshold", 35),
                "limit": args.get("limit", 30),
            },
            timeout=120,
        )
        candidates = r.json().get("stocks", [])[:15]
        if not candidates:
            return _ok("RSI 조건 만족 종목 없음")

        min_roe = args.get("min_roe", 8.0)
        max_per = args.get("max_per", 20.0)
        passed = []
        for s in candidates:
            d = _detail(s["symbol"], s.get("market", args.get("market", "kr")).upper(), "1mo")
            if "error" in d:
                continue
            f = d.get("financials", {})
            if not f.get("available"):
                continue
            per = f.get("basic", {}).get("trailingPE") or f.get("basic", {}).get("forwardPE")
            roe = f.get("profitability", {}).get("returnOnEquity")
            if per and roe and per <= max_per and roe >= min_roe:
                passed.append((s, per, roe))

        if not passed:
            return _ok(
                f"필터 통과 종목 없음 (RSI≤{args.get('rsi_threshold', 35)}, "
                f"ROE≥{min_roe}%, PER≤{max_per}). 조건 완화 권장."
            )
        lines = [f"가치+과매도 스크리닝 통과 ({len(passed)}개):"]
        for s, per, roe in passed:
            lines.append(
                f"  {s['name']}({s['symbol']}) RSI{s['rsi']:.1f} PER{per:.1f} ROE{roe:.1f}% | {s.get('current_price', 'N/A')}"
            )
        return _ok("\n".join(lines))
    except Exception as e:
        return _err(str(e))


# ── MCP 서버 등록 ─────────────────────────────────────────────────────────────

ALL_TOOLS = [
    get_stock_rsi, scan_oversold, search_stock, run_backtest, get_news, get_financials,
    compare_stocks, compare_strategies, correlation, momentum_rank, screen_quality_oversold,
]

quant_mcp = create_sdk_mcp_server(name="quant", version="1.0.0", tools=ALL_TOOLS)

# Claude Code의 allowed_tools 화이트리스트용 — "mcp__{server}__{tool}" 포맷
ALLOWED_TOOL_NAMES = [f"mcp__quant__{t.name if hasattr(t, 'name') else t.__name__}" for t in ALL_TOOLS]
# 위 추출은 SDK 내부 구조 의존 — 안전하게 명시적으로:
ALLOWED_TOOL_NAMES = [
    "mcp__quant__get_stock_rsi",
    "mcp__quant__scan_oversold",
    "mcp__quant__search_stock",
    "mcp__quant__run_backtest",
    "mcp__quant__get_news",
    "mcp__quant__get_financials",
    "mcp__quant__compare_stocks",
    "mcp__quant__compare_strategies",
    "mcp__quant__correlation",
    "mcp__quant__momentum_rank",
    "mcp__quant__screen_quality_oversold",
]

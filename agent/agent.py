"""
Stock Analysis Agent — LangGraph + Ollama (qwen2.5:7b)

[Reflection 루프 패턴]
흐름:
  입력 → llm → tools → llm → critic(충분한가?) ─ 부족 → llm (최대 3회)
                                                 └ 충분 → END

핵심 개념:
  - State에 iteration 카운터를 두어 무한 루프 방지
  - critic 노드가 LLM 출력을 평가해 라우팅 결정
  - conditional_edges로 END vs 루프백 분기
"""

import requests
from typing import Annotated

from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing_extensions import TypedDict

BASE_URL = "http://localhost:8001/api/v1"
MAX_ITERATIONS = 3  # 최대 루프 횟수

# ── 도구 정의 ──────────────────────────────────────────────────────────────────

@tool
def get_stock_rsi(symbol: str, market: str) -> str:
    """
    종목의 현재 RSI 값을 조회합니다.
    symbol: 종목코드 또는 심볼 (예: 005930, AAPL)
    market: 시장 (KOSPI, KOSDAQ, NASDAQ, DOW)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/stock/detail/{symbol}",
            params={"market": market, "period": "3mo", "interval": "1d"},
            timeout=15,
        )
        data = r.json()
        if "error" in data:
            return f"오류: {data['error']}"

        rsi = data.get("rsi")
        price = data.get("current_price")
        change = data.get("change_percent")
        name = data.get("name", symbol)

        if rsi is None:
            return f"{name}({symbol}) RSI 데이터를 가져오지 못했습니다."

        return (
            f"{name} ({symbol}, {market})\n"
            f"  현재가: {price:,.0f}\n"
            f"  등락률: {change:+.2f}%\n"
            f"  RSI(14): {rsi:.1f}"
        )
    except Exception as e:
        return f"조회 실패: {e}"


@tool
def scan_oversold(market: str = "all", rsi_threshold: float = 30, limit: int = 50) -> str:
    """
    RSI 과매도 종목을 스캔합니다 (빠른 조회용, 스트리밍 없이 소수 종목만).
    market: all, us, kr, kospi, kosdaq, nasdaq, dow
    rsi_threshold: RSI 기준값 (기본 30 이하)
    limit: 시장당 최대 스캔 종목 수 (너무 크면 오래 걸림, 최대 50 권장)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/scan/oversold",
            params={"market": market, "rsi_threshold": rsi_threshold, "limit": limit},
            timeout=120,
        )
        data = r.json()
        stocks = data.get("stocks", [])
        if not stocks:
            return f"{market} 시장에서 RSI {rsi_threshold} 이하 종목을 찾지 못했습니다."

        lines = [f"RSI {rsi_threshold} 이하 과매도 종목 ({len(stocks)}개):"]
        for s in stocks[:10]:
            lines.append(
                f"  {s['name']}({s['symbol']}) RSI {s['rsi']:.1f} | "
                f"현재가 {s.get('current_price', 'N/A')}"
            )
        if len(stocks) > 10:
            lines.append(f"  ... 외 {len(stocks)-10}개")
        return "\n".join(lines)
    except Exception as e:
        return f"스캔 실패: {e}"


@tool
def search_stock(query: str) -> str:
    """
    종목명 또는 코드로 종목을 검색합니다.
    query: 검색어 (예: 삼성전자, AAPL, 애플)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/stock/search",
            params={"query": query, "limit": 5},
            timeout=10,
        )
        data = r.json()
        results = data.get("results", [])
        if not results:
            return f"'{query}' 검색 결과 없음"

        lines = [f"'{query}' 검색 결과:"]
        for s in results:
            lines.append(f"  {s['name']} | {s['symbol']} | {s['market']}")
        return "\n".join(lines)
    except Exception as e:
        return f"검색 실패: {e}"


@tool
def run_backtest(
    symbol: str,
    market: str,
    strategy: str = "rsi",
    buy_rsi: float = 30,
    sell_rsi: float = 70,
    period: str = "2y",
) -> str:
    """
    종목의 매매 전략을 과거 데이터로 백테스트합니다.
    symbol: 종목코드 또는 심볼 (예: 005930, AAPL)
    market: 시장 (KOSPI, KOSDAQ, NASDAQ, DOW)
    strategy: 전략 (rsi, ma_cross, rsi_ma)
      - rsi: RSI 기반 (buy_rsi 이하 매수, sell_rsi 이상 매도)
      - ma_cross: 이동평균 골든/데드크로스
      - rsi_ma: RSI + 이동평균 복합
    buy_rsi: RSI 매수 기준 (기본 30)
    sell_rsi: RSI 매도 기준 (기본 70)
    period: 백테스트 기간 (1y, 2y, 3y, 5y)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/stock/{symbol}/backtest",
            params={
                "market": market,
                "strategy": strategy,
                "buy_rsi": buy_rsi,
                "sell_rsi": sell_rsi,
                "period": period,
                "initial_capital": 10000000,
            },
            timeout=30,
        )
        d = r.json()
        if "error" in d:
            return f"백테스트 실패: {d['error']}"

        currency = "₩" if d.get("is_korean") else "$"
        ret = d.get("total_return_percent", 0)
        bh = d.get("buy_hold_return_percent", 0)

        return (
            f"{d.get('symbol')} {strategy.upper()} 전략 백테스트 ({d.get('data_start')} ~ {d.get('data_end')})\n"
            f"  전략 수익률:      {ret:+.1f}%\n"
            f"  Buy & Hold:      {bh:+.1f}%\n"
            f"  전략 우위:        {'✅ 전략 승' if ret > bh else '❌ Buy&Hold 승'}\n"
            f"  총 거래 횟수:     {d.get('total_trades')}회\n"
            f"  승률:             {d.get('win_rate'):.1f}%\n"
            f"  최대 낙폭(MDD):   {d.get('max_drawdown'):.1f}%\n"
            f"  최종 자산:        {currency}{d.get('final_value', 0):,.0f}"
        )
    except Exception as e:
        return f"백테스트 실패: {e}"


@tool
def get_news(symbol: str, market: str, limit: int = 5) -> str:
    """
    종목의 최신 뉴스를 가져옵니다.
    symbol: 종목코드 또는 심볼 (예: 005930, AAPL)
    market: 시장 (KOSPI, KOSDAQ, NASDAQ, DOW)
    limit: 뉴스 개수 (기본 5)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/stock/detail/{symbol}",
            params={"market": market, "period": "1mo", "interval": "1d"},
            timeout=15,
        )
        d = r.json()
        if "error" in d:
            return f"오류: {d['error']}"

        news = d.get("news", [])
        if not news:
            return f"{symbol} 관련 뉴스가 없습니다."

        lines = [f"{d.get('name', symbol)} 최신 뉴스 ({len(news[:limit])}건):"]
        for n in news[:limit]:
            pub = n.get("pubDate", "")[:10] if n.get("pubDate") else ""
            lines.append(f"  [{pub}] {n.get('title', '')}")
            lines.append(f"         출처: {n.get('publisher', '')}")
        return "\n".join(lines)
    except Exception as e:
        return f"뉴스 조회 실패: {e}"


@tool
def get_financials(symbol: str, market: str) -> str:
    """
    종목의 재무지표를 조회합니다. (PER, PBR, 배당수익률, 매출/영업이익 등)
    symbol: 종목코드 또는 심볼 (예: 005930, AAPL)
    market: 시장 (KOSPI, KOSDAQ, NASDAQ, DOW)
    """
    try:
        r = requests.get(
            f"{BASE_URL}/stock/detail/{symbol}",
            params={"market": market, "period": "1mo", "interval": "1d"},
            timeout=15,
        )
        d = r.json()
        if "error" in d:
            return f"오류: {d['error']}"

        f = d.get("financials", {})
        if not f.get("available"):
            return f"{symbol} 재무 데이터를 가져오지 못했습니다."

        basic = f.get("basic", {})
        profit = f.get("profitability", {})
        income = f.get("incomeStatementYearly", [{}])[0]  # 최근 연도

        per = basic.get("trailingPE") or basic.get("forwardPE")
        per_label = "Forward PER" if not basic.get("trailingPE") else "PER"

        lines = [f"{d.get('name', symbol)} ({symbol}) 재무지표"]

        # 밸류에이션
        lines.append("  [밸류에이션]")
        if per:
            lines.append(f"    {per_label}: {per:.1f}배")
        if basic.get("priceToBook"):
            lines.append(f"    PBR: {basic['priceToBook']:.2f}배")
        if basic.get("dividendYield"):
            lines.append(f"    배당수익률: {basic['dividendYield']:.2f}%")
        lines.append(f"    시가총액: {basic.get('marketCapFormatted', 'N/A')}")

        # 수익성
        lines.append("  [수익성]")
        if profit.get("grossMargin"):
            lines.append(f"    매출총이익률: {profit['grossMargin']:.1f}%")
        if profit.get("operatingMargin"):
            lines.append(f"    영업이익률: {profit['operatingMargin']:.1f}%")
        if profit.get("returnOnEquity"):
            lines.append(f"    ROE: {profit['returnOnEquity']:.1f}%")

        # 최근 연도 실적
        if income.get("year"):
            lines.append(f"  [{income['year']}년 실적]")
            if income.get("totalRevenueFormatted"):
                lines.append(f"    매출: {income['totalRevenueFormatted']}")
            if income.get("operatingIncomeFormatted"):
                lines.append(f"    영업이익: {income['operatingIncomeFormatted']}")
            if income.get("netIncomeFormatted"):
                lines.append(f"    순이익: {income['netIncomeFormatted']}")

        return "\n".join(lines)
    except Exception as e:
        return f"재무 조회 실패: {e}"


# ── 그래프 상태 ────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int          # 루프 횟수 카운터
    needs_retry: bool       # critic 판단 결과


# ── LLM 설정 ──────────────────────────────────────────────────────────────────

tools = [get_stock_rsi, scan_oversold, search_stock, run_backtest, get_news, get_financials]

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# critic용 LLM — 도구 없이 판단만
critic_llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ── 노드 정의 ─────────────────────────────────────────────────────────────────

def has_chinese(text: str) -> bool:
    """중국어 문자 비율이 5% 이상이면 True"""
    if not text:
        return False
    chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return chinese_chars / len(text) > 0.05


def llm_node(state: State):
    """LLM이 도구 호출 또는 최종 답변을 결정하는 노드"""
    messages = state["messages"]

    if state.get("needs_retry") and state.get("iteration", 0) > 0:
        hint = SystemMessage(content=(
            "Previous answer was incomplete. "
            "Call tools again or provide a more specific answer. "
            "Respond in Korean only."
        ))
        messages = messages + [hint]

    response = llm_with_tools.invoke(messages)

    # 중국어 감지 시 한국어로 재생성 (최대 2회)
    if response.content and has_chinese(response.content) and not response.tool_calls:
        force_korean = SystemMessage(content=(
            "Your previous response contained Chinese characters. "
            "You MUST rewrite it entirely in Korean (한국어). "
            "Do not use any Chinese characters."
        ))
        response = llm_with_tools.invoke(messages + [response, force_korean])

    return {
        "messages": [response],
        "iteration": state.get("iteration", 0) + 1,
    }


def critic_node(state: State) -> dict:
    """
    Critic 노드: 마지막 AI 답변이 사용자 질문에 충분한지 판단.

    판단 기준:
    - 숫자 데이터(RSI, 가격 등)를 요청했는데 실제 수치가 없으면 → 부족
    - 여러 종목 비교 요청인데 한 종목만 답했으면 → 부족
    - 도구를 한 번도 안 쓴 것 같으면 → 부족
    - 충분히 구체적이면 → 충분
    """
    messages = state["messages"]

    # 사용자 원본 질문 추출 (HumanMessage 중 마지막)
    user_question = ""
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            user_question = m.content
            break

    # AI 마지막 답변
    last_ai = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            last_ai = m.content
            break

    if not last_ai:
        return {"needs_retry": True}

    # critic 프롬프트
    critique_prompt = f"""다음 질문과 답변을 평가하세요.

질문: {user_question}
답변: {last_ai}

평가 기준:
- RSI, 현재가, 등락률 등 수치 데이터를 요청했는데 실제 숫자가 없으면 → 불충분
- 여러 종목 비교를 요청했는데 일부만 답했으면 → 불충분
- 질문에 직접 답하지 않고 일반적인 설명만 했으면 → 불충분
- 구체적인 수치와 함께 질문에 직접 답했으면 → 충분

오직 "충분" 또는 "불충분" 중 하나만 답하세요."""

    result = critic_llm.invoke([HumanMessage(content=critique_prompt)])
    verdict = result.content.strip()
    needs_retry = "불충분" in verdict

    return {"needs_retry": needs_retry}


# ── 라우팅 함수 ───────────────────────────────────────────────────────────────

def should_retry(state: State) -> str:
    """
    critic 판단 결과 + 반복 횟수로 다음 노드 결정.
    - 부족하고 아직 여유 있으면 → "llm" (루프백)
    - 충분하거나 한계 도달 → END
    """
    if state.get("needs_retry") and state.get("iteration", 0) < MAX_ITERATIONS:
        return "llm"
    return END


# ── 그래프 조립 ───────────────────────────────────────────────────────────────

graph_builder = StateGraph(State)

graph_builder.add_node("llm", llm_node)
graph_builder.add_node("tools", ToolNode(tools))
graph_builder.add_node("critic", critic_node)

graph_builder.set_entry_point("llm")

# llm → tools (도구 호출 있을 때) or critic (최종 답변일 때)
def route_after_llm(state: State) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "critic"

graph_builder.add_conditional_edges("llm", route_after_llm)

# tools 실행 후 → 다시 llm
graph_builder.add_edge("tools", "llm")

# critic 판단 후 → 루프백 or END
graph_builder.add_conditional_edges("critic", should_retry)

agent = graph_builder.compile()


# ── 실행 ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = SystemMessage(content=(
    "You are a Korean stock analysis assistant. "
    "CRITICAL: Always respond in Korean (한국어). Never use Chinese or English in your response. "
    "Use actual data from tools. Include specific numbers in your answers."
))

def chat(user_input: str, history: list = None) -> tuple[str, list]:
    base = [SYSTEM_PROMPT] if not history else []
    messages = base + (history or []) + [HumanMessage(content=user_input)]
    result = agent.invoke({
        "messages": messages,
        "iteration": 0,
        "needs_retry": False,
    })
    new_history = result["messages"]
    answer = new_history[-1].content
    return answer, new_history


def main():
    print("=" * 60)
    print("Stock Analysis Agent (LangGraph + Reflection 루프)")
    print("종료: quit 또는 exit")
    print("=" * 60)
    print()
    print("예시 질문:")
    print("  - 삼성전자 RSI 알려줘")
    print("  - AAPL이랑 NVDA 비교해줘")
    print("  - 나스닥 과매도 종목 찾아줘")
    print()

    history = []
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "종료"):
            print("종료합니다.")
            break

        print("Agent: ", end="", flush=True)
        answer, history = chat(user_input, history)
        print(answer)
        print()


if __name__ == "__main__":
    main()

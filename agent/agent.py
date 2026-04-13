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

import json
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


# ── 그래프 상태 ────────────────────────────────────────────────────────────────

class State(TypedDict):
    messages: Annotated[list, add_messages]
    iteration: int          # 루프 횟수 카운터
    needs_retry: bool       # critic 판단 결과


# ── LLM 설정 ──────────────────────────────────────────────────────────────────

tools = [get_stock_rsi, scan_oversold, search_stock]

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
llm_with_tools = llm.bind_tools(tools)

# critic용 LLM — 도구 없이 판단만
critic_llm = ChatOllama(model="qwen2.5:7b", temperature=0)


# ── 노드 정의 ─────────────────────────────────────────────────────────────────

def llm_node(state: State):
    """LLM이 도구 호출 또는 최종 답변을 결정하는 노드"""
    # 재시도 시 힌트 메시지 추가
    messages = state["messages"]
    if state.get("needs_retry") and state.get("iteration", 0) > 0:
        hint = SystemMessage(content=(
            "이전 답변이 불완전했습니다. "
            "사용자 질문에 필요한 정보가 누락됐거나 도구를 충분히 활용하지 않았습니다. "
            "도구를 다시 호출하거나 더 구체적인 답변을 작성하세요."
        ))
        messages = messages + [hint]

    response = llm_with_tools.invoke(messages)
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

def chat(user_input: str, history: list = None) -> tuple[str, list]:
    messages = (history or []) + [HumanMessage(content=user_input)]
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

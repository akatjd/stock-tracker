"""
Claude Agent SDK 기반 채팅 (Pro 요금제 사용 — 별도 API 키 불필요).

- 인증: 사용자가 `claude` CLI에 로그인되어 있으면 SDK가 ~/.claude/ 토큰을 그대로 사용
- 기존 agent.py(Ollama)와 완전히 분리 — 이 파일을 삭제하면 Claude 백엔드 제거됨
"""

import asyncio
from typing import AsyncGenerator

from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AssistantMessage, TextBlock

from quant_tools import quant_mcp, ALLOWED_TOOL_NAMES

SYSTEM_PROMPT = (
    "너는 한국어로 답변하는 주식/퀀트 분석 어시스턴트다. "
    "반드시 도구를 사용해 실제 데이터를 가져온 뒤 구체적인 수치와 함께 답하라. "
    "추측이나 일반론을 답변하지 말고, 도구가 반환한 숫자를 인용해라. "
    "여러 종목 비교/스크리닝은 compare_stocks, momentum_rank, "
    "screen_quality_oversold, correlation, compare_strategies 같은 퀀트 도구를 적극 활용해라."
)


def build_options(model: str = "sonnet") -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"quant": quant_mcp},
        allowed_tools=ALLOWED_TOOL_NAMES,
        model=model,
        permission_mode="bypassPermissions",  # 화이트리스트로만 도구 노출하므로 안전
        max_turns=8,
    )


async def stream_chat(message: str, model: str = "sonnet") -> AsyncGenerator[dict, None]:
    """
    Claude Pro로 메시지 처리 → 토큰을 yield.
    yield 형태:
      {"type": "tool_use", "name": "..."}     — 도구 호출 시작
      {"type": "token", "content": "..."}     — 텍스트 청크
      {"type": "done"}                        — 종료
      {"type": "error", "message": "..."}     — 오류
    """
    options = build_options(model)
    try:
        async with ClaudeSDKClient(options=options) as client:
            await client.query(message)

            async for msg in client.receive_response():
                if isinstance(msg, AssistantMessage):
                    for block in msg.content:
                        # 텍스트 블록
                        if isinstance(block, TextBlock):
                            text = block.text
                            if text:
                                # 단어 단위 작은 청크로 나눠서 SSE에 자연스럽게
                                for word in text.split(" "):
                                    yield {"type": "token", "content": word + " "}
                                    await asyncio.sleep(0.005)
                        # 도구 호출 알림
                        elif hasattr(block, "name") and hasattr(block, "input"):
                            yield {"type": "tool_use", "name": getattr(block, "name", "?")}
        yield {"type": "done"}
    except Exception as e:
        yield {"type": "error", "message": str(e)}

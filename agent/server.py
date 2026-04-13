"""
Stock Agent Web Server
채팅 API를 제공하는 FastAPI 서버 (포트 8002)

POST /chat        — 단순 응답 (전체 답변 한 번에)
GET  /chat/stream — SSE 스트리밍 (토큰 단위 실시간)
DELETE /chat/history — 대화 초기화
"""

import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from agent import chat, agent, tools
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

app = FastAPI(title="Stock Agent API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 서버 전체 대화 히스토리 (단순 구현 — 멀티유저 아님)
history: list = []


class ChatRequest(BaseModel):
    message: str
    reset: bool = False


@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    global history
    if req.reset:
        history = []

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as executor:
        answer, history = await loop.run_in_executor(
            executor, lambda: chat(req.message, history)
        )
    return {"answer": answer}


@app.get("/chat/stream")
async def chat_stream(message: str, reset: bool = False):
    """SSE로 토큰 단위 스트리밍"""
    global history

    if reset:
        history = []

    async def generate() -> AsyncGenerator[str, None]:
        global history

        # 도구 실행은 기존 agent 그래프로, LLM 최종 답변만 스트리밍
        messages = history + [HumanMessage(content=message)]

        # 1단계: 도구 호출이 필요한지 graph로 먼저 처리
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            result = await loop.run_in_executor(
                executor, lambda: agent.invoke({"messages": messages})
            )

        updated_messages = result["messages"]

        # 마지막 AI 메시지 내용을 청크로 나눠서 전송 (스트리밍 시뮬레이션)
        final_answer = updated_messages[-1].content

        # 히스토리 업데이트
        history = updated_messages

        # 단어 단위로 스트리밍
        words = final_answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield f"data: {json.dumps({'type': 'token', 'content': chunk}, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.03)  # 타이핑 효과

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.delete("/chat/history")
async def reset_history():
    global history
    history = []
    return {"message": "대화 초기화 완료"}


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8002, reload=True)

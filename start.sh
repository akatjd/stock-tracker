#!/bin/bash

ROOT="$(cd "$(dirname "$0")" && pwd)"

# 색상
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

stop_servers() {
  echo -e "\n${YELLOW}서버 종료 중...${NC}"
  kill "$BACKEND_PID" "$AGENT_PID" "$FRONTEND_PID" 2>/dev/null
  exit 0
}
trap stop_servers SIGINT SIGTERM

# 포트 정리
for port in 8001 8002 5173; do
  pid=$(lsof -ti:$port 2>/dev/null)
  if [ -n "$pid" ]; then
    echo -e "${YELLOW}포트 $port 정리 중 (PID: $pid)${NC}"
    kill -9 $pid 2>/dev/null
  fi
done
sleep 1

echo -e "${GREEN}=== Stock Tracker 시작 ===${NC}"

# 백엔드 (8001)
echo -e "${GREEN}[1/3] 백엔드 시작 (포트 8001)...${NC}"
cd "$ROOT/backend"
.venv/bin/uvicorn app.main:app --reload --port 8001 > /tmp/stock-backend.log 2>&1 &
BACKEND_PID=$!

# 에이전트 (8002)
echo -e "${GREEN}[2/3] AI 에이전트 시작 (포트 8002)...${NC}"
cd "$ROOT/agent"
.venv/bin/uvicorn server:app --port 8002 > /tmp/stock-agent.log 2>&1 &
AGENT_PID=$!

# 프론트엔드 (5173)
echo -e "${GREEN}[3/3] 프론트엔드 시작 (포트 5173)...${NC}"
cd "$ROOT/frontend"
npm run dev > /tmp/stock-frontend.log 2>&1 &
FRONTEND_PID=$!

# 기동 대기
sleep 3

# 상태 확인
echo ""
echo -e "${GREEN}=== 서버 상태 ===${NC}"

check() {
  local name=$1 url=$2
  if curl -s "$url" > /dev/null 2>&1; then
    echo -e "  ${GREEN}✅ $name${NC} → $url"
  else
    echo -e "  ${RED}❌ $name 기동 실패${NC} (로그: /tmp/stock-${name,,}.log)"
  fi
}

check "백엔드"    "http://localhost:8001/health"
check "에이전트"  "http://localhost:8002/health"
check "프론트"    "http://localhost:5173"

echo ""
echo -e "${GREEN}🌐 브라우저: http://localhost:5173${NC}"
echo -e "${YELLOW}종료: Ctrl+C${NC}"
echo ""

wait

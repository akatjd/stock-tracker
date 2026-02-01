# Stock Tracker

RSI 기반 과매도 주식 스캐너 및 기술적 분석 도구입니다. 한국(KOSPI/KOSDAQ)과 미국(NASDAQ) 주식 시장을 지원합니다.

## 주요 기능

### 주식 스캔
- RSI(상대강도지수) 기반 과매도 종목 스캔
- 시장별 필터링 (KOSPI, KOSDAQ, NASDAQ, 전체)
- 시가총액 필터링 (대형주, 중형주, 소형주)
- 섹터별 필터링
- 실시간 스캔 진행률 표시 (SSE)

### 차트 분석
- 캔들스틱 차트 (lightweight-charts)
- 기술적 지표: 볼린저 밴드, MACD, RSI
- 이동평균선: MA5, MA20, MA60, MA120
- 지지선/저항선 자동 계산
- 차트 기간 선택: 1개월 ~ 5년
- 봉 타입 선택: 1시간, 1일, 1주, 1개월

### 차트 그리기 도구
- 추세선 그리기
- 수평선 그리기
- 반직선 그리기
- 자석 모드 (고가/저가/종가 스냅)
- 전체화면 모드

### 재무제표
- 5개년 재무 데이터 표시
- 손익계산서 (매출, 영업이익, 순이익, EBITDA)
- 대차대조표 (자산, 부채, 자본, 현금, 부채)
- 현금흐름표 (영업/투자/재무 현금흐름)
- 수익성 지표 (매출총이익률, 영업이익률, ROE, ROA)
- 한국 주식: 원화(억원) 단위
- 미국 주식: 달러(B/M) 단위

## 기술 스택

### Backend
- Python 3.12
- FastAPI
- yfinance (주가 데이터)
- pandas (데이터 처리)
- pykrx (한국 주식 데이터)

### Frontend
- React 18
- Vite
- lightweight-charts (차트 라이브러리)
- CSS (다크 테마)

## 설치 및 실행

### Backend

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| GET | `/api/v1/scan/oversold` | 과매도 종목 스캔 |
| GET | `/api/v1/scan/oversold/stream` | 실시간 스캔 (SSE) |
| GET | `/api/v1/scan/preview` | 스캔 미리보기 |
| GET | `/api/v1/stock/detail/{symbol}` | 종목 상세 정보 |
| GET | `/api/v1/stock/validate` | 종목 코드 유효성 검사 |
| GET | `/api/v1/symbols/us` | 미국 종목 목록 |
| GET | `/api/v1/symbols/kr` | 한국 종목 목록 |
| GET | `/api/v1/stock/{symbol}/rsi` | RSI 조회 |

## 스크린샷

### 과매도 스캔
- RSI 임계값 설정 (기본 30)
- 시장/시가총액/섹터 필터
- 실시간 스캔 진행률

### 종목 상세
- 캔들스틱 차트
- 기술적 지표
- 5개년 재무제표
- 추세선 그리기 도구

## 라이선스

MIT License

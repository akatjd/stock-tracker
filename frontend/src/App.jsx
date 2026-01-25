import { useState, useRef } from 'react'
import './App.css'

function App() {
  const [stocks, setStocks] = useState([])
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [market, setMarket] = useState('all')
  const [rsiThreshold, setRsiThreshold] = useState(30)
  const [limit, setLimit] = useState(500)
  const [scanInfo, setScanInfo] = useState(null)
  const [marketCap, setMarketCap] = useState('all')
  const [sector, setSector] = useState('all')

  // 실시간 진행 상황 상태
  const [progress, setProgress] = useState({ current: 0, total: 0, percent: 0 })
  const [currentStock, setCurrentStock] = useState({ symbol: '', market: '' })
  const [foundCount, setFoundCount] = useState(0)
  const eventSourceRef = useRef(null)
  const [showOnlyOversold, setShowOnlyOversold] = useState(false)

  const scanOversold = () => {
    setIsLoading(true)
    setError(null)
    setStocks([])
    setProgress({ current: 0, total: 0, percent: 0 })
    setCurrentStock({ symbol: '', market: '' })
    setFoundCount(0)
    setScanInfo(null)

    // 기존 연결 정리
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
    }

    const url = `http://localhost:8000/api/v1/scan/oversold/stream?market=${market}&rsi_threshold=${rsiThreshold}&limit=${limit}&market_cap=${marketCap}&sector=${sector}`
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'progress':
          setProgress({
            current: data.current,
            total: data.total,
            percent: data.percent
          })
          setCurrentStock({
            symbol: data.symbol,
            market: data.market
          })
          setFoundCount(data.found)
          break

        case 'found':
          // 실시간으로 발견된 종목 추가
          setStocks(prev => [...prev, data.stock].sort((a, b) => a.rsi - b.rsi))
          break

        case 'complete':
          setStocks(data.results)
          const oversoldCount = data.results.filter(s => s.is_oversold).length
          setScanInfo({
            total: data.total_count,
            oversold: oversoldCount,
            market: market,
            threshold: rsiThreshold
          })
          setIsLoading(false)
          eventSource.close()
          break

        case 'error':
          setError(data.message)
          setIsLoading(false)
          eventSource.close()
          break

        case 'heartbeat':
          // 연결 유지용, 무시
          break
      }
    }

    eventSource.onerror = () => {
      setError('스캔 연결이 끊어졌습니다')
      setIsLoading(false)
      eventSource.close()
    }
  }

  const cancelScan = () => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close()
      eventSourceRef.current = null
    }
    setIsLoading(false)
  }

  const getMarketLabel = (m) => {
    switch (m) {
      case 'US': return '미국'
      case 'KR': return '한국'
      case 'KOSPI': return 'KOSPI'
      case 'KOSDAQ': return 'KOSDAQ'
      case 'NASDAQ': return 'NASDAQ'
      case 'DOW': return 'DOW'
      default: return m
    }
  }

  const getSectorLabel = (sector) => {
    const labels = {
      technology: '기술',
      finance: '금융',
      healthcare: '헬스케어',
      consumer: '소비재',
      industrial: '산업재',
      energy: '에너지',
      utilities: '유틸리티',
      materials: '소재',
      realestate: '부동산',
      communication: '통신'
    }
    return labels[sector] || '-'
  }

  const getMarketCapLabel = (cap) => {
    const labels = {
      large: '대형',
      mid: '중형',
      small: '소형'
    }
    return labels[cap] || '-'
  }

  const getRsiColor = (rsi) => {
    if (rsi <= 20) return '#dc2626'
    if (rsi <= 30) return '#f97316'
    return '#22c55e'
  }

  return (
    <div className="app">
      <header className="header">
        <h1>RSI 과매도 스캐너</h1>
        <p>일봉 RSI가 30 이하인 종목을 찾아보세요</p>
      </header>

      <main className="main">
        <div className="controls">
          <div className="control-group">
            <label>시장 선택</label>
            <select value={market} onChange={(e) => setMarket(e.target.value)}>
              <option value="all">전체</option>
              <optgroup label="한국">
                <option value="kr">한국 전체 (KOSPI + KOSDAQ)</option>
                <option value="kospi">KOSPI</option>
                <option value="kosdaq">KOSDAQ</option>
              </optgroup>
              <optgroup label="미국">
                <option value="us">미국 전체 (NASDAQ + DOW)</option>
                <option value="nasdaq">NASDAQ</option>
                <option value="dow">DOW (다우존스 30)</option>
              </optgroup>
            </select>
          </div>

          <div className="control-group">
            <label>RSI 기준값</label>
            <input
              type="number"
              value={rsiThreshold}
              onChange={(e) => setRsiThreshold(Number(e.target.value))}
              min="1"
              max="100"
            />
          </div>

          <div className="control-group">
            <label>스캔 종목 수 (시장당)</label>
            <input
              type="number"
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
              min="10"
              max="2000"
              step="100"
            />
          </div>

          <div className="control-group">
            <label>시가총액</label>
            <select value={marketCap} onChange={(e) => setMarketCap(e.target.value)}>
              <option value="all">전체</option>
              <option value="large">대형주</option>
              <option value="mid">중형주</option>
              <option value="small">소형주</option>
            </select>
          </div>

          <div className="control-group">
            <label>섹터</label>
            <select value={sector} onChange={(e) => setSector(e.target.value)}>
              <option value="all">전체</option>
              <option value="technology">기술</option>
              <option value="finance">금융</option>
              <option value="healthcare">헬스케어</option>
              <option value="consumer">소비재</option>
              <option value="industrial">산업재</option>
              <option value="energy">에너지</option>
              <option value="utilities">유틸리티</option>
              <option value="materials">소재</option>
              <option value="realestate">부동산</option>
              <option value="communication">통신</option>
            </select>
          </div>

          <button
            className={`scan-button ${isLoading ? 'scanning' : ''}`}
            onClick={isLoading ? cancelScan : scanOversold}
          >
            {isLoading ? '스캔 중지' : '스캔 시작'}
          </button>
        </div>

        {isLoading && (
          <div className="loading">
            <div className="progress-container">
              <div className="progress-bar">
                <div
                  className="progress-fill"
                  style={{ width: `${progress.percent}%` }}
                />
              </div>
              <div className="progress-text">
                {progress.current} / {progress.total} ({progress.percent}%)
              </div>
            </div>

            <div className="current-stock">
              <span className="scanning-label">스캔 중:</span>
              <span className={`market-badge ${currentStock.market.toLowerCase()}`}>
                {currentStock.market}
              </span>
              <span className="stock-symbol">{currentStock.symbol}</span>
            </div>

            <div className="found-count">
              발견된 과매도 종목: <strong>{foundCount}</strong>개
            </div>

            {stocks.length > 0 && (
              <div className="found-stocks-preview">
                <h4>실시간 발견 종목</h4>
                <div className="found-list">
                  {stocks.slice(0, 10).map((stock, idx) => (
                    <div key={`${stock.symbol}-${idx}`} className="found-item">
                      <span className={`market-badge small ${stock.market.toLowerCase()}`}>
                        {stock.market}
                      </span>
                      <span className="found-symbol">{stock.symbol}</span>
                      <span className="found-name">{stock.name}</span>
                      <span className="found-rsi" style={{ backgroundColor: getRsiColor(stock.rsi) }}>
                        RSI {stock.rsi.toFixed(1)}
                      </span>
                    </div>
                  ))}
                  {stocks.length > 10 && (
                    <div className="more-count">+{stocks.length - 10}개 더...</div>
                  )}
                </div>
              </div>
            )}

            <p className="loading-hint">스캔을 중지하려면 '스캔 중지' 버튼을 클릭하세요</p>
          </div>
        )}

        {error && (
          <div className="error">
            <p>오류: {error}</p>
          </div>
        )}

        {scanInfo && !isLoading && (
          <div className="scan-summary">
            <div className="summary-card oversold">
              <div className="summary-number">{stocks.filter(s => s.is_oversold).length}</div>
              <div className="summary-label">과매도 종목 (RSI ≤ {scanInfo.threshold})</div>
            </div>
            <div className="summary-card total">
              <div className="summary-number">{stocks.length}</div>
              <div className="summary-label">전체 스캔 종목</div>
            </div>
          </div>
        )}

        {stocks.length > 0 && !isLoading && (
          <div className="results">
            <div className="results-header">
              <h3>스캔 결과</h3>
              <div className="filter-toggle">
                <button
                  className={`toggle-btn ${!showOnlyOversold ? 'active' : ''}`}
                  onClick={() => setShowOnlyOversold(false)}
                >
                  전체 ({stocks.length})
                </button>
                <button
                  className={`toggle-btn ${showOnlyOversold ? 'active' : ''}`}
                  onClick={() => setShowOnlyOversold(true)}
                >
                  과매도만 ({stocks.filter(s => s.is_oversold).length})
                </button>
              </div>
            </div>
            <table>
              <thead>
                <tr>
                  <th>시장</th>
                  <th>종목코드</th>
                  <th>종목명</th>
                  <th>섹터</th>
                  <th>시총</th>
                  <th>현재가</th>
                  <th>등락률</th>
                  <th>RSI (14)</th>
                </tr>
              </thead>
              <tbody>
                {(showOnlyOversold ? stocks.filter(s => s.is_oversold) : stocks).map((stock, index) => (
                  <tr key={`${stock.symbol}-${index}`} className={stock.is_oversold ? 'oversold-row' : ''}>
                    <td>
                      <span className={`market-badge ${stock.market.toLowerCase()}`}>
                        {getMarketLabel(stock.market)}
                      </span>
                    </td>
                    <td className="symbol">{stock.symbol}</td>
                    <td className="name">{stock.name}</td>
                    <td className="sector">{getSectorLabel(stock.sector)}</td>
                    <td className="market-cap-cell">{getMarketCapLabel(stock.market_cap_label)}</td>
                    <td className="price">
                      {['KOSPI', 'KOSDAQ'].includes(stock.market)
                        ? `₩${stock.price.toLocaleString()}`
                        : `$${stock.price.toFixed(2)}`
                      }
                    </td>
                    <td className={`change ${stock.change_percent >= 0 ? 'positive' : 'negative'}`}>
                      {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent.toFixed(2)}%
                    </td>
                    <td>
                      <span
                        className="rsi-badge"
                        style={{ backgroundColor: getRsiColor(stock.rsi) }}
                      >
                        {stock.rsi.toFixed(1)}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {!isLoading && stocks.length === 0 && scanInfo && (
          <div className="no-results">
            <p>스캔된 종목이 없습니다</p>
          </div>
        )}
      </main>

      <footer className="footer">
        <p>Stock Tracker - RSI 과매도 스캐너 v1.0</p>
        <p className="disclaimer">투자 권유가 아닙니다. 투자 결정은 본인 책임입니다.</p>
      </footer>
    </div>
  )
}

export default App

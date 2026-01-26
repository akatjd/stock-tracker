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
  const [period, setPeriod] = useState('day')  // 봉 타입: day, week, month

  // 실시간 진행 상황 상태
  const [progress, setProgress] = useState({ current: 0, total: 0, percent: 0 })
  const [currentStock, setCurrentStock] = useState({ symbol: '', market: '' })
  const [foundCount, setFoundCount] = useState(0)
  const eventSourceRef = useRef(null)
  const [showOnlyOversold, setShowOnlyOversold] = useState(false)

  // 미리보기 상태
  const [previewData, setPreviewData] = useState(null)
  const [isLoadingPreview, setIsLoadingPreview] = useState(false)
  const [showPreview, setShowPreview] = useState(false)
  const [previewPage, setPreviewPage] = useState(1)
  const [previewSearch, setPreviewSearch] = useState('')
  const [searchInput, setSearchInput] = useState('')

  // 커스텀 종목 추가 상태
  const [customStocks, setCustomStocks] = useState([])
  const [addStockInput, setAddStockInput] = useState('')
  const [addStockMarket, setAddStockMarket] = useState('KOSPI')

  // 미리보기 함수
  const fetchPreview = async (page = 1, search = '') => {
    setIsLoadingPreview(true)
    setError(null)

    try {
      let url = `http://localhost:8000/api/v1/scan/preview?market=${market}&limit=${limit}&market_cap=${marketCap}&sector=${sector}&page=${page}&page_size=50`
      if (search) {
        url += `&search=${encodeURIComponent(search)}`
      }
      const response = await fetch(url)
      const data = await response.json()
      setPreviewData(data)
      setPreviewPage(page)
      setPreviewSearch(search)
      setShowPreview(true)
    } catch (err) {
      setError('미리보기를 불러오는데 실패했습니다: ' + err.message)
    } finally {
      setIsLoadingPreview(false)
    }
  }

  // 페이지 변경
  const handlePageChange = (newPage) => {
    fetchPreview(newPage, previewSearch)
  }

  // 검색 실행
  const handleSearch = () => {
    setPreviewPage(1)
    fetchPreview(1, searchInput)
  }

  // 검색 초기화
  const handleSearchReset = () => {
    setSearchInput('')
    setPreviewSearch('')
    fetchPreview(1, '')
  }

  // 미리보기 닫기
  const closePreview = () => {
    setShowPreview(false)
    setPreviewData(null)
    setPreviewPage(1)
    setPreviewSearch('')
    setSearchInput('')
  }

  // 커스텀 종목 추가
  const handleAddStock = () => {
    if (!addStockInput.trim()) return

    const symbol = addStockInput.trim().toUpperCase()

    // 이미 추가된 종목인지 확인
    if (customStocks.some(s => s.symbol === symbol && s.market === addStockMarket)) {
      alert('이미 추가된 종목입니다.')
      return
    }

    const newStock = {
      symbol: symbol,
      name: symbol,  // 이름은 스캔 시 조회됨
      market: addStockMarket,
      isCustom: true
    }

    setCustomStocks(prev => [...prev, newStock])
    setAddStockInput('')
  }

  // 커스텀 종목 삭제
  const handleRemoveStock = (symbol, market) => {
    setCustomStocks(prev => prev.filter(s => !(s.symbol === symbol && s.market === market)))
  }

  // 커스텀 종목 전체 삭제
  const handleClearCustomStocks = () => {
    setCustomStocks([])
  }

  // 미리보기에서 스캔 시작
  const startScanFromPreview = () => {
    setShowPreview(false)
    scanOversold()
  }

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

    // 커스텀 종목을 쿼리에 추가
    let url = `http://localhost:8000/api/v1/scan/oversold/stream?market=${market}&rsi_threshold=${rsiThreshold}&limit=${limit}&market_cap=${marketCap}&sector=${sector}&period=${period}`
    if (customStocks.length > 0) {
      url += `&custom_stocks=${encodeURIComponent(JSON.stringify(customStocks))}`
    }
    const eventSource = new EventSource(url)
    eventSourceRef.current = eventSource

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data)

      switch (data.type) {
        case 'connected':
          // 연결 성공
          console.log('SSE connected:', data.message)
          break

        case 'start':
          // 스캔 시작, 총 종목 수 수신
          setProgress(prev => ({ ...prev, total: data.total }))
          break

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

  const formatMarketCap = (marketCap, market) => {
    if (!marketCap) return '-'

    // 한국 주식: 원화 기준으로 억 단위 (1억 = 100,000,000)
    // 미국 주식: 달러 기준으로 억 단위 환산 (환율 약 1,350원 가정)
    const isKorean = ['KOSPI', 'KOSDAQ'].includes(market)

    if (isKorean) {
      // 한국: 원화를 억 단위로 변환
      const billions = marketCap / 100000000
      if (billions >= 10000) {
        return `${(billions / 10000).toFixed(1)}조`
      }
      return `${Math.round(billions).toLocaleString()}억`
    } else {
      // 미국: 달러를 원화 억 단위로 환산 (1달러 = 1,350원)
      const krwValue = marketCap * 1350
      const billions = krwValue / 100000000
      if (billions >= 10000) {
        return `${(billions / 10000).toFixed(1)}조`
      }
      return `${Math.round(billions).toLocaleString()}억`
    }
  }

  const getRsiColor = (rsi) => {
    if (rsi <= 20) return '#dc2626'
    if (rsi <= 30) return '#f97316'
    return '#22c55e'
  }

  const getPeriodLabel = (p) => {
    switch (p) {
      case 'day': return '일봉'
      case 'week': return '주봉'
      case 'month': return '월봉'
      default: return p
    }
  }

  return (
    <div className="app">
      <header className="header">
        <h1>RSI 과매도 스캐너</h1>
        <p>일봉/주봉/월봉 RSI가 30 이하인 종목을 찾아보세요</p>
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
            <label>봉 타입</label>
            <select value={period} onChange={(e) => setPeriod(e.target.value)}>
              <option value="day">일봉</option>
              <option value="week">주봉</option>
              <option value="month">월봉</option>
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
            className="preview-button"
            onClick={() => fetchPreview()}
            disabled={isLoading || isLoadingPreview}
          >
            {isLoadingPreview ? '불러오는 중...' : '종목 미리보기'}
          </button>

          <button
            className={`scan-button ${isLoading ? 'scanning' : ''}`}
            onClick={isLoading ? cancelScan : scanOversold}
          >
            {isLoading ? '스캔 중지' : '스캔 시작'}
          </button>
        </div>

        {/* 미리보기 모달 */}
        {showPreview && previewData && (
          <div className="preview-modal">
            <div className="preview-content">
              <div className="preview-header">
                <h3>스캔 대상 종목 미리보기</h3>
                <button className="close-button" onClick={closePreview}>×</button>
              </div>

              <div className="preview-summary">
                <div className="preview-total">
                  총 <strong>{previewData.total_count.toLocaleString()}</strong>개 종목
                  {previewSearch && (
                    <span className="search-result-count">
                      {' '}(검색 결과: {previewData.filtered_count}개)
                    </span>
                  )}
                </div>
                <div className="preview-markets">
                  {Object.entries(previewData.market_counts).map(([m, count]) => (
                    <span key={m} className={`market-badge ${m.toLowerCase()}`}>
                      {m}: {count}개
                    </span>
                  ))}
                </div>
              </div>

              {/* 검색 영역 */}
              <div className="preview-search">
                <input
                  type="text"
                  placeholder="종목코드 또는 종목명 검색..."
                  value={searchInput}
                  onChange={(e) => setSearchInput(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
                />
                <button className="search-button" onClick={handleSearch}>
                  검색
                </button>
                {previewSearch && (
                  <button className="search-reset-button" onClick={handleSearchReset}>
                    초기화
                  </button>
                )}
              </div>

              {/* 종목 추가 영역 */}
              <div className="add-stock-section">
                <div className="add-stock-header">
                  <span>종목 직접 추가</span>
                  {customStocks.length > 0 && (
                    <button className="clear-all-button" onClick={handleClearCustomStocks}>
                      전체 삭제
                    </button>
                  )}
                </div>
                <div className="add-stock-form">
                  <select
                    value={addStockMarket}
                    onChange={(e) => setAddStockMarket(e.target.value)}
                    className="market-select"
                  >
                    <option value="KOSPI">KOSPI</option>
                    <option value="KOSDAQ">KOSDAQ</option>
                    <option value="NASDAQ">NASDAQ</option>
                    <option value="NYSE">NYSE</option>
                  </select>
                  <input
                    type="text"
                    placeholder="종목코드 입력 (예: 005930, AAPL)"
                    value={addStockInput}
                    onChange={(e) => setAddStockInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAddStock()}
                  />
                  <button className="add-stock-button" onClick={handleAddStock}>
                    추가
                  </button>
                </div>
                {customStocks.length > 0 && (
                  <div className="custom-stocks-list">
                    {customStocks.map((stock, idx) => (
                      <div key={`custom-${stock.symbol}-${idx}`} className="custom-stock-item">
                        <span className={`market-badge small ${stock.market.toLowerCase()}`}>
                          {stock.market}
                        </span>
                        <span className="custom-stock-symbol">{stock.symbol}</span>
                        <button
                          className="remove-stock-button"
                          onClick={() => handleRemoveStock(stock.symbol, stock.market)}
                        >
                          ×
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              <div className="preview-list">
                {isLoadingPreview ? (
                  <div className="preview-loading">불러오는 중...</div>
                ) : previewData.stocks.length === 0 ? (
                  <div className="preview-empty">검색 결과가 없습니다</div>
                ) : (
                  <table>
                    <thead>
                      <tr>
                        <th>시장</th>
                        <th>종목코드</th>
                        <th>종목명</th>
                      </tr>
                    </thead>
                    <tbody>
                      {previewData.stocks.map((stock, idx) => (
                        <tr key={`${stock.symbol}-${idx}`}>
                          <td>
                            <span className={`market-badge small ${stock.market.toLowerCase()}`}>
                              {stock.market}
                            </span>
                          </td>
                          <td className="symbol">{stock.symbol}</td>
                          <td className="name">{stock.name}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}
              </div>

              {/* 페이지네이션 */}
              {previewData.total_pages > 1 && (
                <div className="preview-pagination">
                  <button
                    className="page-button"
                    onClick={() => handlePageChange(1)}
                    disabled={!previewData.has_prev}
                  >
                    ««
                  </button>
                  <button
                    className="page-button"
                    onClick={() => handlePageChange(previewPage - 1)}
                    disabled={!previewData.has_prev}
                  >
                    «
                  </button>
                  <span className="page-info">
                    {previewPage} / {previewData.total_pages} 페이지
                  </span>
                  <button
                    className="page-button"
                    onClick={() => handlePageChange(previewPage + 1)}
                    disabled={!previewData.has_next}
                  >
                    »
                  </button>
                  <button
                    className="page-button"
                    onClick={() => handlePageChange(previewData.total_pages)}
                    disabled={!previewData.has_next}
                  >
                    »»
                  </button>
                </div>
              )}

              <div className="preview-actions">
                <button className="cancel-button" onClick={closePreview}>
                  취소
                </button>
                <button className="scan-button" onClick={startScanFromPreview}>
                  이 종목들 스캔 시작
                </button>
              </div>
            </div>
          </div>
        )}

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
                  <th>RSI ({getPeriodLabel(period)})</th>
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
                    <td className="market-cap-cell">
                      <span className="market-cap-value">{formatMarketCap(stock.market_cap, stock.market)}</span>
                      <span className="market-cap-label">{getMarketCapLabel(stock.market_cap_label)}</span>
                    </td>
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

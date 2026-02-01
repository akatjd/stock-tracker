import { useState, useRef } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Area, AreaChart, ComposedChart, Bar, ReferenceLine, Legend, Cell
} from 'recharts'
import TradingChart from './components/TradingChart'
import './App.css'

function App() {
  const [stocks, setStocks] = useState([])

  // 종목 상세 모달 상태
  const [selectedStock, setSelectedStock] = useState(null)
  const [stockDetail, setStockDetail] = useState(null)

  // 차트 지표 토글 상태
  const [chartIndicators, setChartIndicators] = useState({
    ma5: true,
    ma20: true,
    ma60: false,
    ma120: false,
    bollinger: true,
    volume: true,
    rsi: true,
    macd: true
  })
  // 차트 기간/봉 타입 상태
  const [chartPeriod, setChartPeriod] = useState('6mo')  // 1mo, 3mo, 6mo, 1y, 2y, 5y
  const [chartInterval, setChartInterval] = useState('1d')  // 1h, 4h, 1d, 1wk, 1mo
  const [isLoadingDetail, setIsLoadingDetail] = useState(false)
  const [showDetail, setShowDetail] = useState(false)
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

  // 정렬 상태
  const [sortColumn, setSortColumn] = useState('rsi')  // 기본 RSI 정렬
  const [sortDirection, setSortDirection] = useState('asc')  // 'asc' | 'desc'

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
  const [isValidating, setIsValidating] = useState(false)
  const [validationMessage, setValidationMessage] = useState(null)  // { type: 'error' | 'success', text: string }

  // 종목 직접 검색 상태
  const [directSearchSymbol, setDirectSearchSymbol] = useState('')
  const [directSearchMarket, setDirectSearchMarket] = useState('KOSPI')
  const [isSearching, setIsSearching] = useState(false)
  const [searchError, setSearchError] = useState(null)

  // 미리보기 함수
  const fetchPreview = async (page = 1, search = '') => {
    setIsLoadingPreview(true)
    setError(null)

    try {
      let url = `http://localhost:8001/api/v1/scan/preview?market=${market}&limit=${limit}&market_cap=${marketCap}&sector=${sector}&page=${page}&page_size=50`
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
    setValidationMessage(null)
  }

  // 커스텀 종목 추가
  const handleAddStock = async () => {
    if (!addStockInput.trim()) return

    const symbol = addStockInput.trim().toUpperCase()
    setValidationMessage(null)

    // 1. 이미 커스텀 목록에 추가된 종목인지 확인
    if (customStocks.some(s => s.symbol === symbol && s.market === addStockMarket)) {
      setValidationMessage({ type: 'error', text: '이미 추가된 종목입니다.' })
      return
    }

    setIsValidating(true)
    try {
      // 2. 미리보기 API로 전체 목록에서 종목 검색 (해당 시장에서만)
      const previewCheckUrl = `http://localhost:8001/api/v1/scan/preview?market=${addStockMarket.toLowerCase()}&limit=500&search=${encodeURIComponent(symbol)}&page=1&page_size=10`
      const previewResponse = await fetch(previewCheckUrl)
      const previewResult = await previewResponse.json()

      // 정확히 같은 종목코드가 있는지 확인
      const existsInPreview = previewResult.stocks.some(
        s => s.symbol === symbol && s.market === addStockMarket
      )
      if (existsInPreview) {
        setValidationMessage({ type: 'error', text: '이미 스캔 목록에 포함된 종목입니다.' })
        setIsValidating(false)
        return
      }

      // 3. 백엔드 API로 종목 존재 여부 검증
      const response = await fetch(
        `http://localhost:8001/api/v1/stock/validate?symbol=${encodeURIComponent(symbol)}&market=${encodeURIComponent(addStockMarket)}`
      )
      const result = await response.json()

      if (!result.valid) {
        setValidationMessage({ type: 'error', text: result.message })
        return
      }

      // 검증 성공: 종목 추가
      const newStock = {
        symbol: result.symbol,
        name: result.name,
        market: addStockMarket,
        isCustom: true
      }

      setCustomStocks(prev => [...prev, newStock])
      setAddStockInput('')
      setValidationMessage({ type: 'success', text: `${result.name} (${result.symbol}) 추가됨` })

      // 성공 메시지 3초 후 제거
      setTimeout(() => setValidationMessage(null), 3000)

    } catch (err) {
      setValidationMessage({ type: 'error', text: '종목 검증 중 오류가 발생했습니다.' })
    } finally {
      setIsValidating(false)
    }
  }

  // 커스텀 종목 삭제
  const handleRemoveStock = (symbol, market) => {
    setCustomStocks(prev => prev.filter(s => !(s.symbol === symbol && s.market === market)))
  }

  // 커스텀 종목 전체 삭제
  const handleClearCustomStocks = () => {
    setCustomStocks([])
  }

  // 종목 상세 정보 조회
  const fetchStockDetail = async (stock, period = chartPeriod, interval = chartInterval) => {
    setSelectedStock(stock)
    setIsLoadingDetail(true)
    setShowDetail(true)
    setStockDetail(null)

    try {
      const response = await fetch(
        `http://localhost:8001/api/v1/stock/detail/${encodeURIComponent(stock.symbol)}?market=${encodeURIComponent(stock.market)}&period=${period}&interval=${interval}`
      )
      const data = await response.json()

      if (data.error) {
        setStockDetail({ error: data.error })
      } else {
        setStockDetail(data)
      }
    } catch (err) {
      setStockDetail({ error: '상세 정보를 불러오는데 실패했습니다.' })
    } finally {
      setIsLoadingDetail(false)
    }
  }

  // 차트 기간/봉 타입 변경 시 데이터 새로고침
  const refreshChartData = async (newPeriod, newInterval) => {
    if (!selectedStock) return

    setChartPeriod(newPeriod)
    setChartInterval(newInterval)
    setIsLoadingDetail(true)

    try {
      const response = await fetch(
        `http://localhost:8001/api/v1/stock/detail/${encodeURIComponent(selectedStock.symbol)}?market=${encodeURIComponent(selectedStock.market)}&period=${newPeriod}&interval=${newInterval}`
      )
      const data = await response.json()

      if (!data.error) {
        setStockDetail(data)
      }
    } catch (err) {
      console.error('Failed to refresh chart data:', err)
    } finally {
      setIsLoadingDetail(false)
    }
  }

  // 상세 모달 닫기
  const closeDetail = () => {
    setShowDetail(false)
    setSelectedStock(null)
    setStockDetail(null)
    // 초기값으로 리셋
    setChartPeriod('6mo')
    setChartInterval('1d')
  }

  // 종목 직접 검색
  const handleDirectSearch = async () => {
    if (!directSearchSymbol.trim()) {
      setSearchError('종목 코드를 입력해주세요.')
      return
    }

    const symbol = directSearchSymbol.trim().toUpperCase()
    setIsSearching(true)
    setSearchError(null)

    try {
      // 먼저 종목 유효성 검사
      const validateResponse = await fetch(
        `http://localhost:8001/api/v1/stock/validate?symbol=${encodeURIComponent(symbol)}&market=${directSearchMarket}`
      )
      const validateData = await validateResponse.json()

      if (!validateData.valid) {
        setSearchError(validateData.message || '유효하지 않은 종목입니다.')
        setIsSearching(false)
        return
      }

      // 유효한 종목이면 상세 정보 조회
      const stock = {
        symbol: symbol,
        name: validateData.name || symbol,
        market: directSearchMarket
      }

      setDirectSearchSymbol('')
      setSearchError(null)
      fetchStockDetail(stock)
    } catch (err) {
      setSearchError('종목 검색에 실패했습니다: ' + err.message)
    } finally {
      setIsSearching(false)
    }
  }

  // Enter 키로 검색
  const handleSearchKeyDown = (e) => {
    if (e.key === 'Enter') {
      handleDirectSearch()
    }
  }

  // 숫자 포맷팅 (억/조 단위)
  const formatNumber = (num, isKorean = false) => {
    if (!num) return '-'
    if (isKorean) {
      const billions = num / 100000000
      if (billions >= 10000) return `${(billions / 10000).toFixed(1)}조`
      return `${Math.round(billions).toLocaleString()}억`
    } else {
      if (num >= 1000000000000) return `$${(num / 1000000000000).toFixed(2)}T`
      if (num >= 1000000000) return `$${(num / 1000000000).toFixed(2)}B`
      if (num >= 1000000) return `$${(num / 1000000).toFixed(2)}M`
      return `$${num.toLocaleString()}`
    }
  }

  // 퍼센트 포맷팅
  const formatPercent = (value) => {
    if (value === null || value === undefined) return '-'
    return `${(value * 100).toFixed(2)}%`
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
    let url = `http://localhost:8001/api/v1/scan/oversold/stream?market=${market}&rsi_threshold=${rsiThreshold}&limit=${limit}&market_cap=${marketCap}&sector=${sector}&period=${period}`
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

  // 정렬 핸들러
  const handleSort = (column) => {
    if (sortColumn === column) {
      // 같은 칼럼 클릭 시 방향 전환
      setSortDirection(prev => prev === 'asc' ? 'desc' : 'asc')
    } else {
      // 다른 칼럼 클릭 시 해당 칼럼으로 오름차순 정렬
      setSortColumn(column)
      setSortDirection('asc')
    }
  }

  // 정렬된 주식 목록 반환
  const getSortedStocks = () => {
    const filtered = showOnlyOversold ? stocks.filter(s => s.is_oversold) : stocks

    return [...filtered].sort((a, b) => {
      let aVal, bVal

      switch (sortColumn) {
        case 'market':
          aVal = a.market || ''
          bVal = b.market || ''
          break
        case 'symbol':
          aVal = a.symbol || ''
          bVal = b.symbol || ''
          break
        case 'name':
          aVal = a.name || ''
          bVal = b.name || ''
          break
        case 'sector':
          aVal = a.sector || ''
          bVal = b.sector || ''
          break
        case 'market_cap':
          aVal = a.market_cap || 0
          bVal = b.market_cap || 0
          break
        case 'price':
          aVal = a.price || 0
          bVal = b.price || 0
          break
        case 'change_percent':
          aVal = a.change_percent || 0
          bVal = b.change_percent || 0
          break
        case 'rsi':
          aVal = a.rsi || 0
          bVal = b.rsi || 0
          break
        default:
          return 0
      }

      // 문자열 비교
      if (typeof aVal === 'string') {
        const cmp = aVal.localeCompare(bVal, 'ko')
        return sortDirection === 'asc' ? cmp : -cmp
      }

      // 숫자 비교
      const cmp = aVal - bVal
      return sortDirection === 'asc' ? cmp : -cmp
    })
  }

  // 정렬 아이콘
  const SortIcon = ({ column }) => {
    if (sortColumn !== column) {
      return <span className="sort-icon inactive">⇅</span>
    }
    return <span className="sort-icon active">{sortDirection === 'asc' ? '↑' : '↓'}</span>
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-top">
          <div className="header-title">
            <h1>RSI 과매도 스캐너</h1>
            <p>일봉/주봉/월봉 RSI가 30 이하인 종목을 찾아보세요</p>
          </div>
          <div className="stock-search">
            <div className="search-inputs">
              <select
                value={directSearchMarket}
                onChange={(e) => setDirectSearchMarket(e.target.value)}
                className="search-market-select"
              >
                <optgroup label="한국">
                  <option value="KOSPI">KOSPI</option>
                  <option value="KOSDAQ">KOSDAQ</option>
                </optgroup>
                <optgroup label="미국">
                  <option value="NASDAQ">NASDAQ</option>
                </optgroup>
              </select>
              <input
                type="text"
                value={directSearchSymbol}
                onChange={(e) => setDirectSearchSymbol(e.target.value)}
                onKeyDown={handleSearchKeyDown}
                placeholder="종목코드 (예: 005930, AAPL)"
                className="search-input"
              />
              <button
                onClick={handleDirectSearch}
                disabled={isSearching}
                className="search-button"
              >
                {isSearching ? '검색중...' : '검색'}
              </button>
            </div>
            {searchError && <div className="search-error">{searchError}</div>}
          </div>
        </div>
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
                    onChange={(e) => {
                      setAddStockMarket(e.target.value)
                      setValidationMessage(null)
                    }}
                    className="market-select"
                    disabled={isValidating}
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
                    onChange={(e) => {
                      setAddStockInput(e.target.value)
                      setValidationMessage(null)
                    }}
                    onKeyDown={(e) => e.key === 'Enter' && !isValidating && handleAddStock()}
                    disabled={isValidating}
                  />
                  <button
                    className="add-stock-button"
                    onClick={handleAddStock}
                    disabled={isValidating}
                  >
                    {isValidating ? '검증중...' : '추가'}
                  </button>
                </div>
                {validationMessage && (
                  <div className={`validation-message ${validationMessage.type}`}>
                    {validationMessage.text}
                  </div>
                )}
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

        {/* 종목 상세 모달 */}
        {showDetail && (
          <div className="detail-modal">
            <div className="detail-content">
              <div className="detail-header">
                <div className="detail-title">
                  {selectedStock && (
                    <>
                      <span className={`market-badge ${selectedStock.market.toLowerCase()}`}>
                        {selectedStock.market}
                      </span>
                      <h2>{selectedStock.name}</h2>
                      <span className="detail-symbol">{selectedStock.symbol}</span>
                    </>
                  )}
                </div>
                <button className="close-button" onClick={closeDetail}>×</button>
              </div>

              {/* 초기 로딩 (데이터가 없을 때만 전체 스피너 표시) */}
              {isLoadingDetail && !stockDetail && (
                <div className="detail-loading">
                  <div className="spinner"></div>
                  <p>상세 정보를 불러오는 중...</p>
                </div>
              )}
              {stockDetail?.error ? (
                <div className="detail-error">
                  <p>{stockDetail.error}</p>
                </div>
              ) : stockDetail && (
                <div className="detail-body">
                  {/* 가격 정보 */}
                  <div className="price-section">
                    <div className="current-price-large">
                      {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                        ? `₩${stockDetail.current_price?.toLocaleString()}`
                        : `$${stockDetail.current_price?.toFixed(2)}`
                      }
                    </div>
                    <div className={`price-change-large ${stockDetail.change >= 0 ? 'positive' : 'negative'}`}>
                      {stockDetail.change >= 0 ? '+' : ''}
                      {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                        ? `₩${stockDetail.change?.toLocaleString()}`
                        : `$${stockDetail.change?.toFixed(2)}`
                      }
                      {' '}({stockDetail.change_percent >= 0 ? '+' : ''}{stockDetail.change_percent?.toFixed(2)}%)
                    </div>
                  </div>

                  {/* 주요 지표 */}
                  <div className="indicators-grid">
                    <div className="indicator-card">
                      <div className="indicator-label">RSI (14)</div>
                      <div className="indicator-value" style={{ color: getRsiColor(stockDetail.rsi) }}>
                        {stockDetail.rsi?.toFixed(1)}
                      </div>
                    </div>
                    <div className="indicator-card">
                      <div className="indicator-label">52주 최고</div>
                      <div className="indicator-value">
                        {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                          ? `₩${stockDetail.high_52w?.toLocaleString()}`
                          : `$${stockDetail.high_52w?.toFixed(2)}`
                        }
                      </div>
                    </div>
                    <div className="indicator-card">
                      <div className="indicator-label">52주 최저</div>
                      <div className="indicator-value">
                        {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                          ? `₩${stockDetail.low_52w?.toLocaleString()}`
                          : `$${stockDetail.low_52w?.toFixed(2)}`
                        }
                      </div>
                    </div>
                    <div className="indicator-card">
                      <div className="indicator-label">시가총액</div>
                      <div className="indicator-value">
                        {formatNumber(stockDetail.market_cap, ['KOSPI', 'KOSDAQ'].includes(stockDetail.market))}
                      </div>
                    </div>
                  </div>

                  {/* 이동평균 */}
                  <div className="ma-section">
                    <h4>이동평균</h4>
                    <div className="ma-grid">
                      <div className="ma-item">
                        <span className="ma-label">MA5</span>
                        <span className="ma-value">
                          {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                            ? `₩${stockDetail.moving_averages?.ma5?.toLocaleString()}`
                            : `$${stockDetail.moving_averages?.ma5?.toFixed(2)}`
                          }
                        </span>
                      </div>
                      <div className="ma-item">
                        <span className="ma-label">MA20</span>
                        <span className="ma-value">
                          {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                            ? `₩${stockDetail.moving_averages?.ma20?.toLocaleString()}`
                            : `$${stockDetail.moving_averages?.ma20?.toFixed(2)}`
                          }
                        </span>
                      </div>
                      {stockDetail.moving_averages?.ma60 && (
                        <div className="ma-item">
                          <span className="ma-label">MA60</span>
                          <span className="ma-value">
                            {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                              ? `₩${stockDetail.moving_averages?.ma60?.toLocaleString()}`
                              : `$${stockDetail.moving_averages?.ma60?.toFixed(2)}`
                            }
                          </span>
                        </div>
                      )}
                      {stockDetail.moving_averages?.ma120 && (
                        <div className="ma-item">
                          <span className="ma-label">MA120</span>
                          <span className="ma-value">
                            {['KOSPI', 'KOSDAQ'].includes(stockDetail.market)
                              ? `₩${stockDetail.moving_averages?.ma120?.toLocaleString()}`
                              : `$${stockDetail.moving_averages?.ma120?.toFixed(2)}`
                            }
                          </span>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* 차트 지표 토글 */}
                  <div className="chart-section">
                    {/* 기간/봉 타입 선택 */}
                    <div className="chart-controls">
                      <div className="control-group">
                        <span className="control-label">기간</span>
                        <div className="control-buttons">
                          {[
                            { value: '1mo', label: '1개월' },
                            { value: '3mo', label: '3개월' },
                            { value: '6mo', label: '6개월' },
                            { value: '1y', label: '1년' },
                            { value: '2y', label: '2년' },
                            { value: '5y', label: '5년' }
                          ].map(p => (
                            <button
                              key={p.value}
                              className={`control-btn ${chartPeriod === p.value ? 'active' : ''}`}
                              onClick={() => refreshChartData(p.value, chartInterval)}
                              disabled={isLoadingDetail}
                            >
                              {p.label}
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="control-group">
                        <span className="control-label">봉 타입</span>
                        <div className="control-buttons">
                          {[
                            { value: '1h', label: '1시간' },
                            { value: '4h', label: '4시간' },
                            { value: '1d', label: '일봉' },
                            { value: '1wk', label: '주봉' },
                            { value: '1mo', label: '월봉' }
                          ].map(i => (
                            <button
                              key={i.value}
                              className={`control-btn ${chartInterval === i.value ? 'active' : ''}`}
                              onClick={() => refreshChartData(chartPeriod, i.value)}
                              disabled={isLoadingDetail}
                            >
                              {i.label}
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>

                    <div className="chart-header">
                      <h4>기술적 분석 차트 {stockDetail.interval && `(${stockDetail.interval})`}</h4>
                      <div className="indicator-toggles">
                        <label className={`toggle-btn ${chartIndicators.ma5 ? 'active' : ''}`}>
                          <input
                            type="checkbox"
                            checked={chartIndicators.ma5}
                            onChange={(e) => setChartIndicators({...chartIndicators, ma5: e.target.checked})}
                          />
                          <span style={{color: '#ff6b6b'}}>MA5</span>
                        </label>
                        <label className={`toggle-btn ${chartIndicators.ma20 ? 'active' : ''}`}>
                          <input
                            type="checkbox"
                            checked={chartIndicators.ma20}
                            onChange={(e) => setChartIndicators({...chartIndicators, ma20: e.target.checked})}
                          />
                          <span style={{color: '#ffd93d'}}>MA20</span>
                        </label>
                        <label className={`toggle-btn ${chartIndicators.ma60 ? 'active' : ''}`}>
                          <input
                            type="checkbox"
                            checked={chartIndicators.ma60}
                            onChange={(e) => setChartIndicators({...chartIndicators, ma60: e.target.checked})}
                          />
                          <span style={{color: '#6bcb77'}}>MA60</span>
                        </label>
                        <label className={`toggle-btn ${chartIndicators.ma120 ? 'active' : ''}`}>
                          <input
                            type="checkbox"
                            checked={chartIndicators.ma120}
                            onChange={(e) => setChartIndicators({...chartIndicators, ma120: e.target.checked})}
                          />
                          <span style={{color: '#9d4edd'}}>MA120</span>
                        </label>
                        <label className={`toggle-btn ${chartIndicators.bollinger ? 'active' : ''}`}>
                          <input
                            type="checkbox"
                            checked={chartIndicators.bollinger}
                            onChange={(e) => setChartIndicators({...chartIndicators, bollinger: e.target.checked})}
                          />
                          <span style={{color: '#4ecdc4'}}>볼린저</span>
                        </label>
                      </div>
                    </div>

                    {/* 메인 가격 차트 (캔들스틱 + 추세선 그리기) */}
                    <div style={{ position: 'relative' }}>
                      <TradingChart
                        data={stockDetail.chart_data}
                        indicators={chartIndicators}
                        supportResistance={stockDetail.support_resistance}
                        height={400}
                        period={chartPeriod}
                        interval={chartInterval}
                        onPeriodChange={(p) => refreshChartData(p, chartInterval)}
                        onIntervalChange={(i) => refreshChartData(chartPeriod, i)}
                        isLoading={isLoadingDetail}
                      />
                      {/* 데이터 새로고침 중 오버레이 */}
                      {isLoadingDetail && (
                        <div style={{
                          position: 'absolute',
                          top: 0,
                          left: 0,
                          right: 0,
                          bottom: 0,
                          background: 'rgba(26, 26, 46, 0.7)',
                          display: 'flex',
                          alignItems: 'center',
                          justifyContent: 'center',
                          zIndex: 100,
                          borderRadius: '8px'
                        }}>
                          <div className="spinner" style={{ width: '30px', height: '30px' }}></div>
                        </div>
                      )}
                    </div>

                    {/* 거래량 차트 */}
                    {chartIndicators.volume && (
                      <div className="chart-container sub-chart">
                        <div className="sub-chart-title">거래량</div>
                        <ResponsiveContainer width="100%" height={80}>
                          <ComposedChart data={stockDetail.chart_data}>
                            <XAxis dataKey="date" tick={false} axisLine={false} />
                            <YAxis tick={{ fill: '#a0a0a0', fontSize: 9 }} tickFormatter={(v) => `${(v / 1000000).toFixed(0)}M`} width={40} />
                            <Tooltip
                              contentStyle={{ background: 'rgba(26, 26, 46, 0.95)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff', fontSize: '11px' }}
                              formatter={(value) => [`${(value / 1000000).toFixed(2)}M`, '거래량']}
                            />
                            <Bar dataKey="volume" fill="#667eea" fillOpacity={0.6} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* RSI 차트 */}
                    {chartIndicators.rsi && (
                      <div className="chart-container sub-chart">
                        <div className="sub-chart-title">RSI (14)</div>
                        <ResponsiveContainer width="100%" height={100}>
                          <ComposedChart data={stockDetail.chart_data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="date" tick={false} axisLine={false} />
                            <YAxis domain={[0, 100]} ticks={[30, 50, 70]} tick={{ fill: '#a0a0a0', fontSize: 9 }} width={30} />
                            <Tooltip
                              contentStyle={{ background: 'rgba(26, 26, 46, 0.95)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff', fontSize: '11px' }}
                              formatter={(value) => [value?.toFixed(2), 'RSI']}
                            />
                            <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" />
                            <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" />
                            <Area type="monotone" dataKey="rsi" stroke="#f59e0b" strokeWidth={1.5} fill="#f59e0b" fillOpacity={0.2} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* MACD 차트 */}
                    {chartIndicators.macd && (
                      <div className="chart-container sub-chart">
                        <div className="sub-chart-title">MACD (12, 26, 9)</div>
                        <ResponsiveContainer width="100%" height={100}>
                          <ComposedChart data={stockDetail.chart_data}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis dataKey="date" tick={{ fill: '#a0a0a0', fontSize: 9 }} tickFormatter={(v) => v.slice(5)} interval={20} />
                            <YAxis tick={{ fill: '#a0a0a0', fontSize: 9 }} width={40} />
                            <Tooltip
                              contentStyle={{ background: 'rgba(26, 26, 46, 0.95)', border: '1px solid rgba(255,255,255,0.2)', borderRadius: '8px', color: '#fff', fontSize: '11px' }}
                              formatter={(value, name) => {
                                const labels = { macd: 'MACD', macd_signal: 'Signal', macd_histogram: 'Histogram' }
                                return [value?.toFixed(2), labels[name] || name]
                              }}
                            />
                            <ReferenceLine y={0} stroke="rgba(255,255,255,0.3)" />
                            <Bar dataKey="macd_histogram">
                              {stockDetail.chart_data.map((entry, index) => (
                                <Cell key={index} fill={entry.macd_histogram >= 0 ? '#22c55e' : '#ef4444'} />
                              ))}
                            </Bar>
                            <Line type="monotone" dataKey="macd" stroke="#3b82f6" strokeWidth={1.5} dot={false} />
                            <Line type="monotone" dataKey="macd_signal" stroke="#f59e0b" strokeWidth={1.5} dot={false} />
                          </ComposedChart>
                        </ResponsiveContainer>
                      </div>
                    )}

                    {/* 지표 토글 (하단) */}
                    <div className="indicator-toggles bottom-toggles">
                      <label className={`toggle-btn ${chartIndicators.volume ? 'active' : ''}`}>
                        <input
                          type="checkbox"
                          checked={chartIndicators.volume}
                          onChange={(e) => setChartIndicators({...chartIndicators, volume: e.target.checked})}
                        />
                        거래량
                      </label>
                      <label className={`toggle-btn ${chartIndicators.rsi ? 'active' : ''}`}>
                        <input
                          type="checkbox"
                          checked={chartIndicators.rsi}
                          onChange={(e) => setChartIndicators({...chartIndicators, rsi: e.target.checked})}
                        />
                        RSI
                      </label>
                      <label className={`toggle-btn ${chartIndicators.macd ? 'active' : ''}`}>
                        <input
                          type="checkbox"
                          checked={chartIndicators.macd}
                          onChange={(e) => setChartIndicators({...chartIndicators, macd: e.target.checked})}
                        />
                        MACD
                      </label>
                    </div>
                  </div>

                  {/* 재무제표 (5개년) */}
                  {stockDetail.financials?.available && (
                    <div className="financials-section">
                      <h4>재무 정보 (5개년)</h4>

                      {/* 기본 정보 */}
                      {stockDetail.financials.basic && (
                        <div className="financials-subsection">
                          <h5>투자 지표</h5>
                          <div className="financials-grid">
                            {stockDetail.financials.basic.trailingPE && (
                              <div className="financial-item">
                                <span className="financial-label">PER</span>
                                <span className="financial-value">{stockDetail.financials.basic.trailingPE}</span>
                              </div>
                            )}
                            {stockDetail.financials.basic.forwardPE && (
                              <div className="financial-item">
                                <span className="financial-label">Forward PER</span>
                                <span className="financial-value">{stockDetail.financials.basic.forwardPE}</span>
                              </div>
                            )}
                            {stockDetail.financials.basic.priceToBook && (
                              <div className="financial-item">
                                <span className="financial-label">PBR</span>
                                <span className="financial-value">{stockDetail.financials.basic.priceToBook}</span>
                              </div>
                            )}
                            {stockDetail.financials.basic.dividendYield && (
                              <div className="financial-item">
                                <span className="financial-label">배당수익률</span>
                                <span className="financial-value">{stockDetail.financials.basic.dividendYield}%</span>
                              </div>
                            )}
                            {stockDetail.financials.basic.marketCapFormatted && (
                              <div className="financial-item">
                                <span className="financial-label">시가총액</span>
                                <span className="financial-value">{stockDetail.financials.basic.marketCapFormatted}</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* 5개년 손익계산서 */}
                      {stockDetail.financials.incomeStatementYearly?.length > 0 && (
                        <div className="financials-subsection">
                          <h5>손익계산서 (연간)</h5>
                          <div className="financials-table-wrapper">
                            <table className="financials-table">
                              <thead>
                                <tr>
                                  <th>항목</th>
                                  {stockDetail.financials.incomeStatementYearly.map(y => (
                                    <th key={y.year}>{y.year}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td>매출액</td>
                                  {stockDetail.financials.incomeStatementYearly.map(y => (
                                    <td key={y.year}>{y.totalRevenueFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>매출총이익</td>
                                  {stockDetail.financials.incomeStatementYearly.map(y => (
                                    <td key={y.year}>{y.grossProfitFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>영업이익</td>
                                  {stockDetail.financials.incomeStatementYearly.map(y => (
                                    <td key={y.year}>{y.operatingIncomeFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>순이익</td>
                                  {stockDetail.financials.incomeStatementYearly.map(y => (
                                    <td key={y.year}>{y.netIncomeFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>EBITDA</td>
                                  {stockDetail.financials.incomeStatementYearly.map(y => (
                                    <td key={y.year}>{y.ebitdaFormatted || '-'}</td>
                                  ))}
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {/* 5개년 대차대조표 */}
                      {stockDetail.financials.balanceSheetYearly?.length > 0 && (
                        <div className="financials-subsection">
                          <h5>대차대조표 (연간)</h5>
                          <div className="financials-table-wrapper">
                            <table className="financials-table">
                              <thead>
                                <tr>
                                  <th>항목</th>
                                  {stockDetail.financials.balanceSheetYearly.map(y => (
                                    <th key={y.year}>{y.year}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td>총자산</td>
                                  {stockDetail.financials.balanceSheetYearly.map(y => (
                                    <td key={y.year}>{y.totalAssetsFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>총부채</td>
                                  {stockDetail.financials.balanceSheetYearly.map(y => (
                                    <td key={y.year}>{y.totalLiabilitiesFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>자기자본</td>
                                  {stockDetail.financials.balanceSheetYearly.map(y => (
                                    <td key={y.year}>{y.totalEquityFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>현금</td>
                                  {stockDetail.financials.balanceSheetYearly.map(y => (
                                    <td key={y.year}>{y.cashFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>총부채</td>
                                  {stockDetail.financials.balanceSheetYearly.map(y => (
                                    <td key={y.year}>{y.totalDebtFormatted || '-'}</td>
                                  ))}
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {/* 5개년 현금흐름표 */}
                      {stockDetail.financials.cashFlowYearly?.length > 0 && (
                        <div className="financials-subsection">
                          <h5>현금흐름표 (연간)</h5>
                          <div className="financials-table-wrapper">
                            <table className="financials-table">
                              <thead>
                                <tr>
                                  <th>항목</th>
                                  {stockDetail.financials.cashFlowYearly.map(y => (
                                    <th key={y.year}>{y.year}</th>
                                  ))}
                                </tr>
                              </thead>
                              <tbody>
                                <tr>
                                  <td>영업활동</td>
                                  {stockDetail.financials.cashFlowYearly.map(y => (
                                    <td key={y.year}>{y.operatingCashFlowFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>투자활동</td>
                                  {stockDetail.financials.cashFlowYearly.map(y => (
                                    <td key={y.year}>{y.investingCashFlowFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>재무활동</td>
                                  {stockDetail.financials.cashFlowYearly.map(y => (
                                    <td key={y.year}>{y.financingCashFlowFormatted || '-'}</td>
                                  ))}
                                </tr>
                                <tr>
                                  <td>잉여현금흐름</td>
                                  {stockDetail.financials.cashFlowYearly.map(y => (
                                    <td key={y.year}>{y.freeCashFlowFormatted || '-'}</td>
                                  ))}
                                </tr>
                              </tbody>
                            </table>
                          </div>
                        </div>
                      )}

                      {/* 수익성 지표 */}
                      {stockDetail.financials.profitability && (
                        <div className="financials-subsection">
                          <h5>수익성 지표 (현재)</h5>
                          <div className="financials-grid">
                            {stockDetail.financials.profitability.grossMargin && (
                              <div className="financial-item">
                                <span className="financial-label">매출총이익률</span>
                                <span className="financial-value">{stockDetail.financials.profitability.grossMargin}%</span>
                              </div>
                            )}
                            {stockDetail.financials.profitability.operatingMargin && (
                              <div className="financial-item">
                                <span className="financial-label">영업이익률</span>
                                <span className="financial-value">{stockDetail.financials.profitability.operatingMargin}%</span>
                              </div>
                            )}
                            {stockDetail.financials.profitability.profitMargin && (
                              <div className="financial-item">
                                <span className="financial-label">순이익률</span>
                                <span className="financial-value">{stockDetail.financials.profitability.profitMargin}%</span>
                              </div>
                            )}
                            {stockDetail.financials.profitability.returnOnAssets && (
                              <div className="financial-item">
                                <span className="financial-label">ROA</span>
                                <span className="financial-value">{stockDetail.financials.profitability.returnOnAssets}%</span>
                              </div>
                            )}
                            {stockDetail.financials.profitability.returnOnEquity && (
                              <div className="financial-item">
                                <span className="financial-label">ROE</span>
                                <span className="financial-value">{stockDetail.financials.profitability.returnOnEquity}%</span>
                              </div>
                            )}
                          </div>
                        </div>
                      )}

                      {/* 기존 형식 지원 (미국 주식) */}
                      {!stockDetail.financials.basic && stockDetail.financials.per && (
                        <div className="financials-grid">
                          {stockDetail.financials.per && (
                            <div className="financial-item">
                              <span className="financial-label">PER</span>
                              <span className="financial-value">{stockDetail.financials.per.toFixed(2)}</span>
                            </div>
                          )}
                          {stockDetail.financials.pbr && (
                            <div className="financial-item">
                              <span className="financial-label">PBR</span>
                              <span className="financial-value">{stockDetail.financials.pbr.toFixed(2)}</span>
                            </div>
                          )}
                          {stockDetail.financials.eps && (
                            <div className="financial-item">
                              <span className="financial-label">EPS</span>
                              <span className="financial-value">${stockDetail.financials.eps.toFixed(2)}</span>
                            </div>
                          )}
                          {stockDetail.financials.roe && (
                            <div className="financial-item">
                              <span className="financial-label">ROE</span>
                              <span className="financial-value">{formatPercent(stockDetail.financials.roe)}</span>
                            </div>
                          )}
                        </div>
                      )}
                      {stockDetail.financials.sector && (
                        <div className="company-info">
                          <p><strong>섹터:</strong> {stockDetail.financials.sector}</p>
                          <p><strong>산업:</strong> {stockDetail.financials.industry}</p>
                        </div>
                      )}
                      {stockDetail.financials.description && (
                        <div className="company-description">
                          <h5>기업 소개</h5>
                          <p>{stockDetail.financials.description}</p>
                        </div>
                      )}
                    </div>
                  )}

                  {stockDetail.financials && !stockDetail.financials.available && (
                    <div className="financials-unavailable">
                      <p>{stockDetail.financials.message}</p>
                    </div>
                  )}
                </div>
              )}
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
                  <th className="sortable" onClick={() => handleSort('market')}>
                    시장 <SortIcon column="market" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('symbol')}>
                    종목코드 <SortIcon column="symbol" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('name')}>
                    종목명 <SortIcon column="name" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('sector')}>
                    섹터 <SortIcon column="sector" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('market_cap')}>
                    시총 <SortIcon column="market_cap" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('price')}>
                    현재가 <SortIcon column="price" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('change_percent')}>
                    등락률 <SortIcon column="change_percent" />
                  </th>
                  <th className="sortable" onClick={() => handleSort('rsi')}>
                    RSI ({getPeriodLabel(period)}) <SortIcon column="rsi" />
                  </th>
                </tr>
              </thead>
              <tbody>
                {getSortedStocks().map((stock, index) => (
                  <tr
                    key={`${stock.symbol}-${index}`}
                    className={`clickable-row ${stock.is_oversold ? 'oversold-row' : ''}`}
                    onClick={() => fetchStockDetail(stock)}
                  >
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

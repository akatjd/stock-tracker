import { useEffect, useRef, useState } from 'react'
import { createChart, CrosshairMode, CandlestickSeries, LineSeries, HistogramSeries } from 'lightweight-charts'

// 날짜 문자열을 lightweight-charts 형식으로 변환
// 일봉/주봉/월봉: 'YYYY-MM-DD' 문자열 반환
// 시간봉: Unix timestamp (초) 반환
const parseDate = (dateStr) => {
  if (!dateStr) return null

  // YYYY-MM-DD 형식 (일봉, 주봉, 월봉) - 문자열 그대로 반환
  if (dateStr.match(/^\d{4}-\d{2}-\d{2}$/)) {
    return dateStr
  }

  // YYYY-MM-DD HH:MM 형식 (시간봉) - Unix timestamp로 변환
  if (dateStr.match(/^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$/)) {
    return Math.floor(new Date(dateStr.replace(' ', 'T')).getTime() / 1000)
  }

  // MM/DD HH:MM 형식 (레거시) - Unix timestamp로 변환
  if (dateStr.match(/^\d{2}\/\d{2} \d{2}:\d{2}$/)) {
    const currentYear = new Date().getFullYear()
    const [datePart, timePart] = dateStr.split(' ')
    const [month, day] = datePart.split('/')
    const [hour, minute] = timePart.split(':')
    return Math.floor(new Date(currentYear, parseInt(month) - 1, parseInt(day), parseInt(hour), parseInt(minute)).getTime() / 1000)
  }

  // 다른 형식은 그대로 파싱 시도
  const parsed = new Date(dateStr).getTime()
  return isNaN(parsed) ? null : Math.floor(parsed / 1000)
}

const TradingChart = ({
  data,
  indicators = {},
  supportResistance = {},
  isKorean = false,
  height = 400
}) => {
  const chartContainerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)
  const volumeSeriesRef = useRef(null)
  const lineSeriesRefs = useRef({})

  // 추세선 그리기 상태
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawMode, setDrawMode] = useState(null) // 'trendline', 'horizontal', 'ray'
  const [trendLines, setTrendLines] = useState([])
  const [currentLine, setCurrentLine] = useState(null)
  const canvasRef = useRef(null)
  const [magnetMode, setMagnetMode] = useState(true) // 자석 모드 (봉 고가/저가 스냅)
  const candleDataRef = useRef([]) // 캔들 데이터 참조용

  // 전체 화면 상태
  const [isFullscreen, setIsFullscreen] = useState(false)

  // 거래량 차트 높이 비율 (0.05 ~ 0.5)
  const [volumeRatio, setVolumeRatio] = useState(0.15)
  const [isResizingVolume, setIsResizingVolume] = useState(false)

  // 전체 화면 차트 높이 계산 (상단 도구바 ~60px + 하단 도움말 ~40px + 여유 공간)
  const [fullscreenHeight, setFullscreenHeight] = useState(window.innerHeight - 180)

  // 전체 화면 높이 업데이트
  useEffect(() => {
    if (!isFullscreen) return

    const updateHeight = () => {
      setFullscreenHeight(window.innerHeight - 180)
    }

    updateHeight()
    window.addEventListener('resize', updateHeight)
    return () => window.removeEventListener('resize', updateHeight)
  }, [isFullscreen])

  // 차트 초기화
  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return

    // 컨테이너 크기 확인
    const containerWidth = chartContainerRef.current.clientWidth
    console.log('Container width:', containerWidth)

    if (containerWidth === 0) {
      // 컨테이너가 아직 렌더링되지 않은 경우 재시도
      const timer = setTimeout(() => {
        if (chartContainerRef.current) {
          chartContainerRef.current.dispatchEvent(new Event('resize'))
        }
      }, 100)
      return () => clearTimeout(timer)
    }

    // 기존 차트 제거
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    let chart = null

    try {
      // 차트 생성
      chart = createChart(chartContainerRef.current, {
        width: containerWidth || 800,
        height: isFullscreen ? fullscreenHeight : height,
        layout: {
          background: { type: 'solid', color: 'transparent' },
          textColor: '#a0a0a0',
        },
        grid: {
          vertLines: { color: 'rgba(255, 255, 255, 0.1)' },
          horzLines: { color: 'rgba(255, 255, 255, 0.1)' },
        },
        crosshair: {
          mode: CrosshairMode.Normal,
          vertLine: {
            width: 1,
            color: 'rgba(102, 126, 234, 0.5)',
            style: 2,
          },
          horzLine: {
            width: 1,
            color: 'rgba(102, 126, 234, 0.5)',
            style: 2,
          },
        },
        rightPriceScale: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
          scaleMargins: {
            top: 0.1,
            bottom: 0.2,
          },
        },
        timeScale: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
          timeVisible: true,
          secondsVisible: false,
        },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: true,
        },
        handleScale: {
          axisPressedMouseMove: true,
          mouseWheel: true,
          pinch: true,
        },
      })

      chartRef.current = chart

      // 캔들스틱 시리즈 추가
      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#22c55e',
        downColor: '#ef4444',
        borderDownColor: '#ef4444',
        borderUpColor: '#22c55e',
        wickDownColor: '#ef4444',
        wickUpColor: '#22c55e',
      })

      // 데이터 변환 및 설정 (날짜를 Unix timestamp로 변환)
      console.log('Raw data sample:', data[0], data[data.length - 1])

      // 중복 타임스탬프 제거를 위한 Map 사용
      const candleMap = new Map()
      data.forEach(d => {
        const time = parseDate(d.date)
        if (time !== null && time !== undefined) {
          candleMap.set(String(time), {
            time,
            open: d.open,
            high: d.high,
            low: d.low,
            close: d.close,
          })
        }
      })

      const candleData = Array.from(candleMap.values()).sort((a, b) => {
        // 문자열 날짜는 알파벳 순으로 정렬 (ISO 형식이므로 정확함)
        if (typeof a.time === 'string' && typeof b.time === 'string') {
          return a.time.localeCompare(b.time)
        }
        // 숫자 타임스탬프는 숫자 비교
        return a.time - b.time
      })

      console.log('Parsed candle data sample:', candleData[0], candleData[candleData.length - 1])
      console.log('Total candles:', candleData.length)

      if (candleData.length === 0) {
        console.error('No valid candle data')
        return
      }

      candleSeries.setData(candleData)
      candleSeriesRef.current = candleSeries
      candleDataRef.current = candleData // 자석 기능용 데이터 저장

      // 데이터 정렬 헬퍼 함수
      const sortByTime = (arr) => arr.sort((a, b) => {
        if (typeof a.time === 'string' && typeof b.time === 'string') {
          return a.time.localeCompare(b.time)
        }
        return a.time - b.time
      })

      // 거래량 시리즈 추가
      if (indicators.volume) {
        const volumeSeries = chart.addSeries(HistogramSeries, {
          color: '#667eea',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
        })
        volumeSeries.priceScale().applyOptions({
          scaleMargins: {
            top: 1 - volumeRatio,
            bottom: 0,
          },
        })
        const volumeData = data
          .map(d => ({
            time: parseDate(d.date),
            value: d.volume,
            color: d.close >= d.open ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)',
          }))
          .filter(d => d.time !== null && d.time !== undefined)
        volumeSeries.setData(sortByTime(volumeData))
        volumeSeriesRef.current = volumeSeries
      }

      // 이동평균선 추가
      const maColors = {
        ma5: '#ff6b6b',
        ma20: '#ffd93d',
        ma60: '#6bcb77',
        ma120: '#9d4edd',
      }

      Object.entries(maColors).forEach(([key, color]) => {
        if (indicators[key]) {
          const maData = data
            .filter(d => d[key] !== null && d[key] !== undefined)
            .map(d => ({ time: parseDate(d.date), value: d[key] }))
            .filter(d => d.time !== null && d.time !== undefined)

          if (maData.length > 0) {
            const maSeries = chart.addSeries(LineSeries, {
              color: color,
              lineWidth: 1,
              priceLineVisible: false,
              lastValueVisible: false,
            })
            maSeries.setData(sortByTime(maData))
            lineSeriesRefs.current[key] = maSeries
          }
        }
      })

      // 볼린저 밴드 추가
      if (indicators.bollinger) {
        const bbUpperData = data
          .filter(d => d.bb_upper !== null && d.bb_upper !== undefined)
          .map(d => ({ time: parseDate(d.date), value: d.bb_upper }))
          .filter(d => d.time !== null && d.time !== undefined)
        const bbLowerData = data
          .filter(d => d.bb_lower !== null && d.bb_lower !== undefined)
          .map(d => ({ time: parseDate(d.date), value: d.bb_lower }))
          .filter(d => d.time !== null && d.time !== undefined)
        const bbMiddleData = data
          .filter(d => d.bb_middle !== null && d.bb_middle !== undefined)
          .map(d => ({ time: parseDate(d.date), value: d.bb_middle }))
          .filter(d => d.time !== null && d.time !== undefined)

        if (bbUpperData.length > 0) {
          const bbUpperSeries = chart.addSeries(LineSeries, {
            color: '#4ecdc4',
            lineWidth: 1,
            lineStyle: 0,
            priceLineVisible: false,
            lastValueVisible: false,
          })
          bbUpperSeries.setData(sortByTime(bbUpperData))

          const bbLowerSeries = chart.addSeries(LineSeries, {
            color: '#4ecdc4',
            lineWidth: 1,
            lineStyle: 0,
            priceLineVisible: false,
            lastValueVisible: false,
          })
          bbLowerSeries.setData(sortByTime(bbLowerData))

          const bbMiddleSeries = chart.addSeries(LineSeries, {
            color: '#4ecdc4',
            lineWidth: 1,
            lineStyle: 2,
            priceLineVisible: false,
            lastValueVisible: false,
          })
          bbMiddleSeries.setData(sortByTime(bbMiddleData))
        }
      }

      // 지지선/저항선 추가
      if (supportResistance.resistance) {
        supportResistance.resistance.forEach(level => {
          candleSeries.createPriceLine({
            price: level,
            color: '#ef4444',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'R',
          })
        })
      }
      if (supportResistance.support) {
        supportResistance.support.forEach(level => {
          candleSeries.createPriceLine({
            price: level,
            color: '#22c55e',
            lineWidth: 1,
            lineStyle: 2,
            axisLabelVisible: true,
            title: 'S',
          })
        })
      }

      // 차트 크기 조정
      chart.timeScale().fitContent()

    } catch (error) {
      console.error('Error initializing chart:', error)
    }

    // 리사이즈 핸들러
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [data, indicators, supportResistance, height, isFullscreen, volumeRatio, fullscreenHeight])

  // 캔버스 크기 초기화
  useEffect(() => {
    if (!canvasRef.current || !chartContainerRef.current) return

    const resizeCanvas = () => {
      const rect = chartContainerRef.current.getBoundingClientRect()
      canvasRef.current.width = rect.width
      canvasRef.current.height = rect.height
      drawLines() // 리사이즈 후 다시 그리기
    }

    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    return () => window.removeEventListener('resize', resizeCanvas)
  }, [data, height, trendLines, isFullscreen])

  // 그리기 모드일 때 차트 인터랙션 비활성화
  useEffect(() => {
    if (!chartRef.current) return

    try {
      if (drawMode) {
        // 그리기 모드: 차트 스크롤/확대 비활성화
        chartRef.current.applyOptions({
          handleScroll: false,
          handleScale: false,
        })
      } else {
        // 일반 모드: 차트 스크롤/확대 활성화
        chartRef.current.applyOptions({
          handleScroll: {
            mouseWheel: true,
            pressedMouseMove: true,
            horzTouchDrag: true,
            vertTouchDrag: true,
          },
          handleScale: {
            axisPressedMouseMove: true,
            mouseWheel: true,
            pinch: true,
          },
        })
      }
    } catch (e) {
      console.error('Error toggling chart interaction:', e)
    }
  }, [drawMode])

  // 자석 기능: 마우스 좌표를 가장 가까운 봉의 고가/저가로 스냅
  const snapToCandle = (x, y) => {
    if (!magnetMode || !chartRef.current || !candleSeriesRef.current || candleDataRef.current.length === 0) {
      return { x, y }
    }

    try {
      const chart = chartRef.current
      const series = candleSeriesRef.current
      const timeScale = chart.timeScale()

      // x 좌표를 시간으로 변환
      const time = timeScale.coordinateToTime(x)
      if (!time) return { x, y }

      // 가장 가까운 캔들 찾기
      let nearestCandle = null
      let minTimeDiff = Infinity

      for (const candle of candleDataRef.current) {
        let timeDiff
        if (typeof candle.time === 'string' && typeof time === 'string') {
          timeDiff = Math.abs(new Date(candle.time).getTime() - new Date(time).getTime())
        } else {
          timeDiff = Math.abs(Number(candle.time) - Number(time))
        }
        if (timeDiff < minTimeDiff) {
          minTimeDiff = timeDiff
          nearestCandle = candle
        }
      }

      if (!nearestCandle) return { x, y }

      // 캔들의 시간 좌표
      const candleX = timeScale.timeToCoordinate(nearestCandle.time)
      if (candleX === null) return { x, y }

      // 고가와 저가의 y 좌표 계산
      const highY = series.priceToCoordinate(nearestCandle.high)
      const lowY = series.priceToCoordinate(nearestCandle.low)

      if (highY === null || lowY === null) return { x, y }

      // 마우스 y 좌표와 더 가까운 쪽으로 스냅
      const distToHigh = Math.abs(y - highY)
      const distToLow = Math.abs(y - lowY)

      const snappedY = distToHigh < distToLow ? highY : lowY

      return { x: candleX, y: snappedY }
    } catch (e) {
      console.error('Snap error:', e)
      return { x, y }
    }
  }

  // 추세선 그리기 핸들러
  const handleCanvasMouseDown = (e) => {
    if (!drawMode || !canvasRef.current) return

    e.preventDefault()
    e.stopPropagation()

    const rect = canvasRef.current.getBoundingClientRect()
    const rawX = e.clientX - rect.left
    const rawY = e.clientY - rect.top

    // 자석 모드 적용
    const { x, y } = snapToCandle(rawX, rawY)

    console.log('Drawing started at:', x, y, '(raw:', rawX, rawY, ')')
    setIsDrawing(true)
    setCurrentLine({ startX: x, startY: y, endX: x, endY: y })
  }

  const handleCanvasMouseMove = (e) => {
    if (!isDrawing || !currentLine || !canvasRef.current) return

    e.preventDefault()
    e.stopPropagation()

    const rect = canvasRef.current.getBoundingClientRect()
    const rawX = e.clientX - rect.left
    const rawY = e.clientY - rect.top

    // 자석 모드 적용
    const { x, y } = snapToCandle(rawX, rawY)

    setCurrentLine(prev => ({ ...prev, endX: x, endY: y }))
  }

  const handleCanvasMouseUp = (e) => {
    if (!isDrawing || !currentLine) return

    e.preventDefault()
    e.stopPropagation()

    console.log('Drawing ended, line:', currentLine)
    setTrendLines(prev => [...prev, { ...currentLine, mode: drawMode }])
    setCurrentLine(null)
    setIsDrawing(false)
  }

  // 캔버스에 추세선 그리기
  const drawLines = () => {
    if (!canvasRef.current) return

    const ctx = canvasRef.current.getContext('2d')
    const rect = canvasRef.current.getBoundingClientRect()
    canvasRef.current.width = rect.width
    canvasRef.current.height = rect.height

    ctx.clearRect(0, 0, rect.width, rect.height)

    // 저장된 추세선 그리기
    trendLines.forEach(line => {
      ctx.beginPath()
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.moveTo(line.startX, line.startY)

      if (line.mode === 'horizontal') {
        ctx.lineTo(rect.width, line.startY)
      } else if (line.mode === 'ray') {
        const dx = line.endX - line.startX
        const dy = line.endY - line.startY
        const length = Math.sqrt(dx * dx + dy * dy)
        const unitX = dx / length
        const unitY = dy / length
        ctx.lineTo(line.startX + unitX * 2000, line.startY + unitY * 2000)
      } else {
        ctx.lineTo(line.endX, line.endY)
      }

      ctx.stroke()
    })

    // 현재 그리는 선 그리기
    if (currentLine) {
      ctx.beginPath()
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.moveTo(currentLine.startX, currentLine.startY)

      if (drawMode === 'horizontal') {
        ctx.lineTo(rect.width, currentLine.startY)
      } else {
        ctx.lineTo(currentLine.endX, currentLine.endY)
      }

      ctx.stroke()
      ctx.setLineDash([])
    }
  }

  // 추세선 모두 삭제
  const clearTrendLines = () => {
    setTrendLines([])
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
    }
  }

  // 거래량 차트 높이 드래그 리사이즈
  const handleVolumeResizeStart = (e) => {
    if (!indicators.volume) return
    e.preventDefault()
    setIsResizingVolume(true)
  }

  useEffect(() => {
    if (!isResizingVolume) return

    const handleMouseMove = (e) => {
      if (!chartContainerRef.current) return
      const rect = chartContainerRef.current.getBoundingClientRect()
      const chartHeight = rect.height
      const mouseY = e.clientY - rect.top
      // 마우스 위치에서 거래량 비율 계산 (아래에서부터)
      const newRatio = Math.max(0.05, Math.min(0.5, (chartHeight - mouseY) / chartHeight))
      setVolumeRatio(newRatio)
    }

    const handleMouseUp = () => {
      setIsResizingVolume(false)
    }

    window.addEventListener('mousemove', handleMouseMove)
    window.addEventListener('mouseup', handleMouseUp)
    return () => {
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('mouseup', handleMouseUp)
    }
  }, [isResizingVolume])

  // 마지막 선 삭제 (Ctrl+Z)
  const undoLastLine = () => {
    if (trendLines.length > 0) {
      setTrendLines(prev => prev.slice(0, -1))
    }
  }

  // 키보드 단축키 핸들러
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ctrl+Z: 실행 취소
      if ((e.ctrlKey || e.metaKey) && e.key === 'z') {
        e.preventDefault()
        undoLastLine()
      }
      // ESC: 그리기 모드 해제
      if (e.key === 'Escape') {
        setDrawMode(null)
        setCurrentLine(null)
        setIsDrawing(false)
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [trendLines])

  useEffect(() => {
    drawLines()
  }, [trendLines, currentLine, drawMode])

  const chartContent = (
    <>
      {/* 그리기 도구 */}
      <div className="drawing-tools">
        <button
          className={`draw-btn ${drawMode === 'trendline' ? 'active' : ''}`}
          onClick={() => setDrawMode(drawMode === 'trendline' ? null : 'trendline')}
          title="추세선"
        >
          📈 추세선
        </button>
        <button
          className={`draw-btn ${drawMode === 'horizontal' ? 'active' : ''}`}
          onClick={() => setDrawMode(drawMode === 'horizontal' ? null : 'horizontal')}
          title="수평선"
        >
          ➖ 수평선
        </button>
        <button
          className={`draw-btn ${drawMode === 'ray' ? 'active' : ''}`}
          onClick={() => setDrawMode(drawMode === 'ray' ? null : 'ray')}
          title="반직선"
        >
          ↗️ 반직선
        </button>
        <button
          className={`draw-btn ${magnetMode ? 'active' : ''}`}
          onClick={() => setMagnetMode(!magnetMode)}
          title="자석 모드 (봉 고가/저가 스냅)"
        >
          🧲 자석
        </button>
        <button
          className="draw-btn clear"
          onClick={clearTrendLines}
          title="모두 삭제"
        >
          🗑️ 삭제
        </button>
        <button
          className={`draw-btn ${isFullscreen ? 'active' : ''}`}
          onClick={() => setIsFullscreen(!isFullscreen)}
          title={isFullscreen ? "축소" : "확대"}
        >
          {isFullscreen ? '🗗 축소' : '🔍 확대'}
        </button>
        {drawMode && <span className="draw-hint">차트 위에서 드래그하여 그리기</span>}
      </div>

      {/* 차트 컨테이너 */}
      <div className="chart-wrapper" style={{ position: 'relative' }}>
        <div
          ref={chartContainerRef}
          style={{
            width: '100%',
            height: isFullscreen ? fullscreenHeight : height
          }}
        />

        {/* 그리기 캔버스 오버레이 */}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            zIndex: drawMode ? 100 : 1,
            pointerEvents: drawMode ? 'auto' : 'none',
            cursor: drawMode ? 'crosshair' : 'default',
            background: 'transparent',
          }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
        />

        {/* 거래량 높이 조절 핸들 */}
        {indicators.volume && (
          <div
            className="volume-resize-handle"
            style={{
              position: 'absolute',
              left: 0,
              right: 50,
              bottom: `${volumeRatio * 100}%`,
              height: '8px',
              cursor: 'ns-resize',
              zIndex: 50,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
            onMouseDown={handleVolumeResizeStart}
          >
            <div
              style={{
                width: '60px',
                height: '4px',
                backgroundColor: isResizingVolume ? '#667eea' : 'rgba(102, 126, 234, 0.5)',
                borderRadius: '2px',
              }}
            />
          </div>
        )}
      </div>

      {/* 사용 안내 */}
      <div className="chart-help">
        <span>🖱️ 스크롤: 확대/축소</span>
        <span>👆 드래그: 이동</span>
        <span>⌨️ Shift+드래그: 시간축 확대</span>
      </div>
    </>
  )

  // 전체 화면 모드
  if (isFullscreen) {
    return (
      <div className="trading-chart-fullscreen">
        <div className="trading-chart-wrapper fullscreen">
          {chartContent}
        </div>
      </div>
    )
  }

  return (
    <div className="trading-chart-wrapper">
      {chartContent}
    </div>
  )
}

export default TradingChart

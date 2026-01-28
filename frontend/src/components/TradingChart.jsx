import { useEffect, useRef, useState } from 'react'
import { createChart, CrosshairMode } from 'lightweight-charts'

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

  // 차트 초기화
  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return

    // 기존 차트 제거
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    // 차트 생성
    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: height,
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
    const candleSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    })

    // 데이터 변환 및 설정
    const candleData = data.map(d => ({
      time: d.date,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))
    candleSeries.setData(candleData)
    candleSeriesRef.current = candleSeries

    // 거래량 시리즈 추가
    if (indicators.volume) {
      const volumeSeries = chart.addHistogramSeries({
        color: '#667eea',
        priceFormat: {
          type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
          top: 0.85,
          bottom: 0,
        },
      })
      const volumeData = data.map(d => ({
        time: d.date,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(34, 197, 94, 0.5)' : 'rgba(239, 68, 68, 0.5)',
      }))
      volumeSeries.setData(volumeData)
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
          .map(d => ({ time: d.date, value: d[key] }))

        if (maData.length > 0) {
          const maSeries = chart.addLineSeries({
            color: color,
            lineWidth: 1,
            priceLineVisible: false,
            lastValueVisible: false,
          })
          maSeries.setData(maData)
          lineSeriesRefs.current[key] = maSeries
        }
      }
    })

    // 볼린저 밴드 추가
    if (indicators.bollinger) {
      const bbUpperData = data
        .filter(d => d.bb_upper !== null)
        .map(d => ({ time: d.date, value: d.bb_upper }))
      const bbLowerData = data
        .filter(d => d.bb_lower !== null)
        .map(d => ({ time: d.date, value: d.bb_lower }))
      const bbMiddleData = data
        .filter(d => d.bb_middle !== null)
        .map(d => ({ time: d.date, value: d.bb_middle }))

      if (bbUpperData.length > 0) {
        const bbUpperSeries = chart.addLineSeries({
          color: '#4ecdc4',
          lineWidth: 1,
          lineStyle: 0,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbUpperSeries.setData(bbUpperData)

        const bbLowerSeries = chart.addLineSeries({
          color: '#4ecdc4',
          lineWidth: 1,
          lineStyle: 0,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbLowerSeries.setData(bbLowerData)

        const bbMiddleSeries = chart.addLineSeries({
          color: '#4ecdc4',
          lineWidth: 1,
          lineStyle: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbMiddleSeries.setData(bbMiddleData)
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

    // 리사이즈 핸들러
    const handleResize = () => {
      if (chartContainerRef.current && chart) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chart) {
        chart.remove()
      }
    }
  }, [data, indicators, supportResistance, height])

  // 추세선 그리기 핸들러
  const handleCanvasMouseDown = (e) => {
    if (!drawMode || !chartRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setIsDrawing(true)
    setCurrentLine({ startX: x, startY: y, endX: x, endY: y })
  }

  const handleCanvasMouseMove = (e) => {
    if (!isDrawing || !currentLine) return

    const rect = canvasRef.current.getBoundingClientRect()
    const x = e.clientX - rect.left
    const y = e.clientY - rect.top

    setCurrentLine(prev => ({ ...prev, endX: x, endY: y }))
    drawLines()
  }

  const handleCanvasMouseUp = () => {
    if (!isDrawing || !currentLine) return

    setTrendLines(prev => [...prev, { ...currentLine, mode: drawMode }])
    setCurrentLine(null)
    setIsDrawing(false)
    drawLines()
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

  useEffect(() => {
    drawLines()
  }, [trendLines, currentLine])

  return (
    <div className="trading-chart-wrapper">
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
          className="draw-btn clear"
          onClick={clearTrendLines}
          title="모두 삭제"
        >
          🗑️ 삭제
        </button>
        {drawMode && <span className="draw-hint">차트 위에서 드래그하여 그리기</span>}
      </div>

      {/* 차트 컨테이너 */}
      <div className="chart-wrapper" style={{ position: 'relative' }}>
        <div ref={chartContainerRef} style={{ width: '100%', height: height }} />

        {/* 그리기 캔버스 오버레이 */}
        <canvas
          ref={canvasRef}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            pointerEvents: drawMode ? 'auto' : 'none',
            cursor: drawMode ? 'crosshair' : 'default',
          }}
          onMouseDown={handleCanvasMouseDown}
          onMouseMove={handleCanvasMouseMove}
          onMouseUp={handleCanvasMouseUp}
          onMouseLeave={handleCanvasMouseUp}
        />
      </div>

      {/* 사용 안내 */}
      <div className="chart-help">
        <span>🖱️ 스크롤: 확대/축소</span>
        <span>👆 드래그: 이동</span>
        <span>⌨️ Shift+드래그: 시간축 확대</span>
      </div>
    </div>
  )
}

export default TradingChart

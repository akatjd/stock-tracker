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

// 데이터 정렬 헬퍼 함수
const sortByTime = (arr) => arr.sort((a, b) => {
  if (typeof a.time === 'string' && typeof b.time === 'string') {
    return a.time.localeCompare(b.time)
  }
  return a.time - b.time
})

// 캔들 데이터 변환 함수
const parseChartData = (rawData) => {
  const candleMap = new Map()
  rawData.forEach(d => {
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
  return Array.from(candleMap.values()).sort((a, b) => {
    if (typeof a.time === 'string' && typeof b.time === 'string') {
      return a.time.localeCompare(b.time)
    }
    return a.time - b.time
  })
}

const TradingChart = ({
  data,
  indicators = {},
  supportResistance = {},
  height = 400,
  period = '6mo',
  interval = '1d',
  onPeriodChange = null,
  onIntervalChange = null,
  isLoading = false
}) => {
  const chartContainerRef = useRef(null)
  const chartRef = useRef(null)
  const candleSeriesRef = useRef(null)
  const volumeSeriesRef = useRef(null)
  const lineSeriesRefs = useRef({})

  // RSI 차트 refs
  const rsiContainerRef = useRef(null)
  const rsiChartRef = useRef(null)
  const rsiSeriesRef = useRef(null)

  // 추세선 그리기 상태
  const [isDrawing, setIsDrawing] = useState(false)
  const [drawMode, setDrawMode] = useState(null) // 'trendline', 'horizontal', 'ray'
  const [trendLines, setTrendLines] = useState([])
  const [currentLine, setCurrentLine] = useState(null)
  const canvasRef = useRef(null)
  const [magnetMode, setMagnetMode] = useState(true) // 자석 모드 (봉 고가/저가 스냅)
  const candleDataRef = useRef([]) // 캔들 데이터 참조용
  const [magnetPreview, setMagnetPreview] = useState(null) // 자석 미리보기 { x, y, price, isHigh }
  const [scaleVersion, setScaleVersion] = useState(0) // 차트 스케일 변경 감지용

  // 전체 화면 상태
  const [isFullscreen, setIsFullscreen] = useState(false)

  // 거래량 차트 높이 비율 (0.05 ~ 0.5)
  const [volumeRatio, setVolumeRatio] = useState(0.15)
  const [isResizingVolume, setIsResizingVolume] = useState(false)

  // 전체 화면 차트 높이 계산 (상단 도구바 ~60px + RSI ~150px + 하단 도움말 ~40px + 여유)
  const [fullscreenHeight, setFullscreenHeight] = useState(window.innerHeight - 180)

  // 전체 화면 높이 업데이트
  useEffect(() => {
    if (!isFullscreen) return

    const rsiSpace = indicators.rsi ? 160 : 0
    const updateHeight = () => {
      setFullscreenHeight(window.innerHeight - 180 - rsiSpace)
    }

    updateHeight()
    window.addEventListener('resize', updateHeight)
    return () => window.removeEventListener('resize', updateHeight)
  }, [isFullscreen])

  // 차트 생성 (한 번만)
  useEffect(() => {
    if (!chartContainerRef.current) return

    const containerWidth = chartContainerRef.current.clientWidth
    if (containerWidth === 0) {
      const timer = setTimeout(() => {
        if (chartContainerRef.current) {
          chartContainerRef.current.dispatchEvent(new Event('resize'))
        }
      }, 100)
      return () => clearTimeout(timer)
    }

    // 차트가 이미 있으면 생성하지 않음
    if (chartRef.current) return

    // 차트 생성
    const chart = createChart(chartContainerRef.current, {
      width: containerWidth || 800,
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
        candleSeriesRef.current = null
        volumeSeriesRef.current = null
        lineSeriesRefs.current = {}
      }
    }
  }, [height])

  // 전체화면/높이 변경 시 차트 크기 조정
  useEffect(() => {
    if (!chartRef.current || !chartContainerRef.current) return

    // CSS 변경이 적용될 때까지 약간 대기
    const resizeChart = () => {
      if (!chartRef.current || !chartContainerRef.current) return
      const containerWidth = chartContainerRef.current.clientWidth
      chartRef.current.applyOptions({
        width: containerWidth,
        height: isFullscreen ? fullscreenHeight : height,
      })
      // 데이터가 전체 너비에 맞게 표시되도록 fitContent 호출
      chartRef.current.timeScale().fitContent()
    }

    // 즉시 한 번 실행
    resizeChart()

    // CSS 전환 완료 후 여러 번 리사이즈 (DOM 업데이트 타이밍 보장)
    const timer1 = setTimeout(resizeChart, 0)
    const timer2 = setTimeout(resizeChart, 100)
    const timer3 = setTimeout(resizeChart, 200)

    return () => {
      clearTimeout(timer1)
      clearTimeout(timer2)
      clearTimeout(timer3)
    }
  }, [isFullscreen, fullscreenHeight, height])

  // 데이터 업데이트 (차트 재생성 없이 시리즈만 업데이트)
  useEffect(() => {
    if (!chartRef.current || !data || data.length === 0) return

    const chart = chartRef.current

    // 기존 시리즈 모두 제거
    try {
      if (candleSeriesRef.current) {
        chart.removeSeries(candleSeriesRef.current)
        candleSeriesRef.current = null
      }
      if (volumeSeriesRef.current) {
        chart.removeSeries(volumeSeriesRef.current)
        volumeSeriesRef.current = null
      }
      Object.values(lineSeriesRefs.current).forEach(series => {
        try { chart.removeSeries(series) } catch (e) {}
      })
      lineSeriesRefs.current = {}
    } catch (e) {
      console.log('Series cleanup:', e)
    }

    // 캔들스틱 시리즈 추가
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderDownColor: '#ef4444',
      borderUpColor: '#22c55e',
      wickDownColor: '#ef4444',
      wickUpColor: '#22c55e',
    })

    const candleData = parseChartData(data)
    if (candleData.length === 0) {
      console.error('No valid candle data')
      return
    }

    candleSeries.setData(candleData)
    candleSeriesRef.current = candleSeries
    candleDataRef.current = candleData

    // 거래량 시리즈 추가
    if (indicators.volume) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        color: '#667eea',
        priceFormat: { type: 'volume' },
        priceScaleId: 'volume',
      })
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 1 - volumeRatio, bottom: 0 },
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
        lineSeriesRefs.current['bbUpper'] = bbUpperSeries

        const bbLowerSeries = chart.addSeries(LineSeries, {
          color: '#4ecdc4',
          lineWidth: 1,
          lineStyle: 0,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbLowerSeries.setData(sortByTime(bbLowerData))
        lineSeriesRefs.current['bbLower'] = bbLowerSeries

        const bbMiddleSeries = chart.addSeries(LineSeries, {
          color: '#4ecdc4',
          lineWidth: 1,
          lineStyle: 2,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        bbMiddleSeries.setData(sortByTime(bbMiddleData))
        lineSeriesRefs.current['bbMiddle'] = bbMiddleSeries
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

  }, [data, indicators, supportResistance, volumeRatio])

  // RSI 차트 생성/업데이트 (전체화면일 때만)
  useEffect(() => {
    if (!isFullscreen || !indicators.rsi || !data || data.length === 0) {
      // 전체화면이 아니거나 RSI 꺼졌으면 제거
      if (rsiChartRef.current) {
        rsiChartRef.current.remove()
        rsiChartRef.current = null
        rsiSeriesRef.current = null
      }
      return
    }

    if (!rsiContainerRef.current) return

    // RSI 차트가 없으면 생성
    if (!rsiChartRef.current) {
      const rsiChart = createChart(rsiContainerRef.current, {
        width: rsiContainerRef.current.clientWidth || 800,
        height: 120,
        layout: {
          background: { type: 'solid', color: 'transparent' },
          textColor: '#a0a0a0',
        },
        grid: {
          vertLines: { color: 'rgba(255, 255, 255, 0.05)' },
          horzLines: { color: 'rgba(255, 255, 255, 0.05)' },
        },
        rightPriceScale: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
          scaleMargins: { top: 0.05, bottom: 0.05 },
        },
        timeScale: {
          borderColor: 'rgba(255, 255, 255, 0.2)',
          timeVisible: true,
          visible: false,
        },
        crosshair: {
          mode: CrosshairMode.Normal,
        },
        handleScroll: { mouseWheel: true, pressedMouseMove: true },
        handleScale: { mouseWheel: true },
      })
      rsiChartRef.current = rsiChart
    }

    const rsiChart = rsiChartRef.current

    // 기존 시리즈 제거
    if (rsiSeriesRef.current) {
      try { rsiChart.removeSeries(rsiSeriesRef.current) } catch (e) {}
      rsiSeriesRef.current = null
    }

    // RSI 라인 시리즈
    const rsiSeries = rsiChart.addSeries(LineSeries, {
      color: '#f59e0b',
      lineWidth: 1.5,
      priceLineVisible: false,
      lastValueVisible: true,
    })

    const rsiData = data
      .filter(d => d.rsi !== null && d.rsi !== undefined)
      .map(d => ({ time: parseDate(d.date), value: d.rsi }))
      .filter(d => d.time !== null && d.time !== undefined)

    if (rsiData.length > 0) {
      rsiSeries.setData(sortByTime(rsiData))

      // 과매도(30) / 과매수(70) 기준선
      rsiSeries.createPriceLine({ price: 70, color: '#ef4444', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: '' })
      rsiSeries.createPriceLine({ price: 30, color: '#22c55e', lineWidth: 1, lineStyle: 2, axisLabelVisible: true, title: '' })
      rsiSeries.createPriceLine({ price: 50, color: 'rgba(255,255,255,0.2)', lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: '' })
    }

    rsiSeriesRef.current = rsiSeries

    // 차트 크기 조정
    rsiChart.applyOptions({ width: rsiContainerRef.current.clientWidth })
    rsiChart.timeScale().fitContent()

    // 메인 차트와 시간축 동기화
    if (chartRef.current) {
      const syncTimeScale = (sourceChart, targetChart) => {
        const logicalRange = sourceChart.timeScale().getVisibleLogicalRange()
        if (logicalRange) {
          targetChart.timeScale().setVisibleLogicalRange(logicalRange)
        }
      }

      const onMainRangeChange = () => syncTimeScale(chartRef.current, rsiChart)
      const onRsiRangeChange = () => syncTimeScale(rsiChart, chartRef.current)

      chartRef.current.timeScale().subscribeVisibleLogicalRangeChange(onMainRangeChange)
      rsiChart.timeScale().subscribeVisibleLogicalRangeChange(onRsiRangeChange)

      // 초기 동기화
      syncTimeScale(chartRef.current, rsiChart)

      return () => {
        try {
          chartRef.current?.timeScale().unsubscribeVisibleLogicalRangeChange(onMainRangeChange)
          rsiChart.timeScale().unsubscribeVisibleLogicalRangeChange(onRsiRangeChange)
        } catch (e) {}
      }
    }
  }, [isFullscreen, indicators.rsi, data])

  // RSI 차트 리사이즈
  useEffect(() => {
    if (!rsiChartRef.current || !rsiContainerRef.current) return

    const handleResize = () => {
      if (rsiContainerRef.current && rsiChartRef.current) {
        rsiChartRef.current.applyOptions({ width: rsiContainerRef.current.clientWidth })
      }
    }
    window.addEventListener('resize', handleResize)
    // 전체화면 전환 후 리사이즈
    const timer = setTimeout(handleResize, 200)
    return () => {
      window.removeEventListener('resize', handleResize)
      clearTimeout(timer)
    }
  }, [isFullscreen])

  // 전체화면 해제 시 RSI 차트 정리
  useEffect(() => {
    if (!isFullscreen && rsiChartRef.current) {
      rsiChartRef.current.remove()
      rsiChartRef.current = null
      rsiSeriesRef.current = null
    }
  }, [isFullscreen])

  // 캔버스 크기 초기화
  useEffect(() => {
    if (!canvasRef.current || !chartContainerRef.current) return

    const resizeCanvas = () => {
      if (!canvasRef.current || !chartContainerRef.current) return
      const rect = chartContainerRef.current.getBoundingClientRect()
      canvasRef.current.width = rect.width
      canvasRef.current.height = rect.height
      drawLines() // 리사이즈 후 다시 그리기
    }

    resizeCanvas()
    // CSS 전환 후 다시 실행
    const timer = setTimeout(resizeCanvas, 50)

    window.addEventListener('resize', resizeCanvas)
    return () => {
      window.removeEventListener('resize', resizeCanvas)
      clearTimeout(timer)
    }
  }, [data, height, trendLines, isFullscreen])

  // 차트 스케일/이동 변경 시 선 다시 그리기
  useEffect(() => {
    if (!chartRef.current) return

    const chart = chartRef.current

    // 스케일/이동 변경 구독 - 상태 업데이트로 리렌더 트리거
    const handleScaleChange = () => {
      setScaleVersion(v => v + 1)
    }

    // 논리적 범위 변경 (줌)
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleScaleChange)
    // 시간 범위 변경 (스크롤/이동)
    chart.timeScale().subscribeVisibleTimeRangeChange(handleScaleChange)
    // 크로스헤어 이동 (마우스 이동 시 - 더 즉각적인 반응)
    chart.subscribeCrosshairMove(handleScaleChange)

    return () => {
      try {
        chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleScaleChange)
        chart.timeScale().unsubscribeVisibleTimeRangeChange(handleScaleChange)
        chart.unsubscribeCrosshairMove(handleScaleChange)
      } catch (e) {
        // 차트가 이미 제거된 경우 무시
      }
    }
  }, [])

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

  // 픽셀 좌표를 차트 좌표(시간/가격)로 변환
  const pixelToChartCoords = (x, y) => {
    if (!chartRef.current || !candleSeriesRef.current) return null
    try {
      const chart = chartRef.current
      const series = candleSeriesRef.current
      const time = chart.timeScale().coordinateToTime(x)
      const price = series.coordinateToPrice(y)
      if (time === null || price === null) return null
      return { time, price }
    } catch (e) {
      return null
    }
  }

  // 차트 좌표를 픽셀 좌표로 변환
  const chartToPixelCoords = (time, price) => {
    if (!chartRef.current || !candleSeriesRef.current) return null
    try {
      const chart = chartRef.current
      const series = candleSeriesRef.current
      const x = chart.timeScale().timeToCoordinate(time)
      const y = series.priceToCoordinate(price)
      if (x === null || y === null) return null
      return { x, y }
    } catch (e) {
      return null
    }
  }

  // 자석 기능: 마우스 좌표를 가장 가까운 봉의 고가/저가로 스냅
  const snapToCandle = (x, y) => {
    if (!magnetMode || !chartRef.current || !candleSeriesRef.current || candleDataRef.current.length === 0) {
      return { x, y, snapped: false, time: null, price: null }
    }

    try {
      const chart = chartRef.current
      const series = candleSeriesRef.current
      const timeScale = chart.timeScale()

      // x 좌표를 시간으로 변환
      const time = timeScale.coordinateToTime(x)
      if (!time) return { x, y, snapped: false }

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

      if (!nearestCandle) return { x, y, snapped: false }

      // 캔들의 시간 좌표
      const candleX = timeScale.timeToCoordinate(nearestCandle.time)
      if (candleX === null) return { x, y, snapped: false }

      // 고가와 저가의 y 좌표 계산
      const highY = series.priceToCoordinate(nearestCandle.high)
      const lowY = series.priceToCoordinate(nearestCandle.low)

      if (highY === null || lowY === null) return { x, y, snapped: false }

      // 마우스 y 좌표와 더 가까운 쪽으로 스냅
      const distToHigh = Math.abs(y - highY)
      const distToLow = Math.abs(y - lowY)

      const isHigh = distToHigh < distToLow
      const snappedY = isHigh ? highY : lowY
      const price = isHigh ? nearestCandle.high : nearestCandle.low

      return { x: candleX, y: snappedY, snapped: true, isHigh, price, time: nearestCandle.time }
    } catch (e) {
      console.error('Snap error:', e)
      return { x, y, snapped: false, time: null, price: null }
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
    const snapResult = snapToCandle(rawX, rawY)
    const { x, y } = snapResult

    // 차트 좌표도 저장 (자석 모드면 스냅된 값, 아니면 변환)
    let startTime, startPrice
    if (snapResult.snapped) {
      startTime = snapResult.time
      startPrice = snapResult.price
    } else {
      const coords = pixelToChartCoords(x, y)
      startTime = coords?.time
      startPrice = coords?.price
    }

    console.log('Drawing started at:', x, y, 'time:', startTime, 'price:', startPrice)
    setIsDrawing(true)
    setCurrentLine({
      startX: x, startY: y, endX: x, endY: y,
      startTime, startPrice, endTime: startTime, endPrice: startPrice
    })
  }

  const handleCanvasMouseMove = (e) => {
    if (!canvasRef.current) return

    const rect = canvasRef.current.getBoundingClientRect()
    const rawX = e.clientX - rect.left
    const rawY = e.clientY - rect.top

    // 자석 모드 적용 및 미리보기 업데이트
    const snapResult = snapToCandle(rawX, rawY)

    if (magnetMode && snapResult.snapped) {
      setMagnetPreview({
        x: snapResult.x,
        y: snapResult.y,
        price: snapResult.price,
        isHigh: snapResult.isHigh
      })
    } else {
      setMagnetPreview(null)
    }

    // 그리기 중일 때 선 업데이트
    if (isDrawing && currentLine) {
      e.preventDefault()
      e.stopPropagation()

      // 차트 좌표도 업데이트
      let endTime, endPrice
      if (snapResult.snapped) {
        endTime = snapResult.time
        endPrice = snapResult.price
      } else {
        const coords = pixelToChartCoords(snapResult.x, snapResult.y)
        endTime = coords?.time
        endPrice = coords?.price
      }

      setCurrentLine(prev => ({
        ...prev,
        endX: snapResult.x,
        endY: snapResult.y,
        endTime,
        endPrice
      }))
    }
  }

  // 캔버스에서 마우스가 나갈 때 미리보기 숨김
  const handleCanvasMouseLeave = (e) => {
    setMagnetPreview(null)
    if (isDrawing && currentLine) {
      handleCanvasMouseUp(e)
    }
  }

  const handleCanvasMouseUp = (e) => {
    // 오른쪽 클릭은 무시 (contextmenu에서 처리)
    if (e.button !== 0) return
    if (!isDrawing || !currentLine) return

    e.preventDefault()
    e.stopPropagation()

    console.log('Drawing ended, line:', currentLine)
    setTrendLines(prev => [...prev, { ...currentLine, mode: drawMode }])
    setCurrentLine(null)
    setIsDrawing(false)
  }

  // 오른쪽 클릭 시 그리기 모드 해제
  const handleCanvasContextMenu = (e) => {
    e.preventDefault()
    if (drawMode) {
      setDrawMode(null)
      setCurrentLine(null)
      setIsDrawing(false)
      setMagnetPreview(null)
    }
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
    trendLines.forEach((line) => {
      // 차트 좌표가 있으면 현재 스케일에 맞게 픽셀 좌표 계산
      let startX = line.startX
      let startY = line.startY
      let endX = line.endX
      let endY = line.endY

      // 수평선은 가격만으로 Y 좌표 계산 (시간과 무관)
      if (line.mode === 'horizontal' && line.startPrice !== undefined && candleSeriesRef.current) {
        const priceY = candleSeriesRef.current.priceToCoordinate(line.startPrice)
        if (priceY !== null) {
          startY = priceY
        }
      } else {
        // 일반 선/반직선은 시간+가격으로 좌표 계산
        if (line.startTime !== undefined && line.startPrice !== undefined) {
          const startCoords = chartToPixelCoords(line.startTime, line.startPrice)
          if (startCoords) {
            startX = startCoords.x
            startY = startCoords.y
          }
        }
        if (line.endTime !== undefined && line.endPrice !== undefined) {
          const endCoords = chartToPixelCoords(line.endTime, line.endPrice)
          if (endCoords) {
            endX = endCoords.x
            endY = endCoords.y
          }
        }
      }

      ctx.beginPath()
      ctx.strokeStyle = '#f59e0b'
      ctx.lineWidth = 2
      ctx.moveTo(startX, startY)

      if (line.mode === 'horizontal') {
        ctx.lineTo(rect.width, startY)
        ctx.moveTo(0, startY)
        ctx.lineTo(startX, startY)
      } else if (line.mode === 'ray') {
        const dx = endX - startX
        const dy = endY - startY
        const length = Math.sqrt(dx * dx + dy * dy)
        if (length > 0) {
          const unitX = dx / length
          const unitY = dy / length
          ctx.lineTo(startX + unitX * 3000, startY + unitY * 3000)
        }
      } else {
        ctx.lineTo(endX, endY)
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

    // 자석 미리보기 표시
    if (magnetPreview && drawMode) {
      const { x, y, price, isHigh } = magnetPreview

      // 원형 마커
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, Math.PI * 2)
      ctx.fillStyle = isHigh ? '#22c55e' : '#ef4444' // 고가: 녹색, 저가: 빨간색
      ctx.fill()
      ctx.strokeStyle = '#fff'
      ctx.lineWidth = 2
      ctx.stroke()

      // 가격 라벨
      const label = `${isHigh ? '고' : '저'} ${price?.toLocaleString()}`
      ctx.font = 'bold 12px sans-serif'
      const textWidth = ctx.measureText(label).width
      const labelX = x + 10
      const labelY = y - 10

      // 라벨 배경
      ctx.fillStyle = isHigh ? 'rgba(34, 197, 94, 0.9)' : 'rgba(239, 68, 68, 0.9)'
      ctx.fillRect(labelX - 4, labelY - 12, textWidth + 8, 16)

      // 라벨 텍스트
      ctx.fillStyle = '#fff'
      ctx.fillText(label, labelX, labelY)
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
  }, [trendLines, currentLine, drawMode, magnetPreview, scaleVersion])

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

        {/* 기간/봉 타입 선택 (전체화면에서만 표시) */}
        {isFullscreen && onPeriodChange && onIntervalChange && (
          <>
            <span className="toolbar-divider"></span>
            <div className="toolbar-group">
              <span className="toolbar-label">기간:</span>
              {[
                { value: '1mo', label: '1M' },
                { value: '3mo', label: '3M' },
                { value: '6mo', label: '6M' },
                { value: '1y', label: '1Y' },
                { value: '2y', label: '2Y' },
                { value: '5y', label: '5Y' }
              ].map(p => (
                <button
                  key={p.value}
                  className={`draw-btn small ${period === p.value ? 'active' : ''}`}
                  onClick={() => onPeriodChange(p.value)}
                  disabled={isLoading}
                >
                  {p.label}
                </button>
              ))}
            </div>
            <div className="toolbar-group">
              <span className="toolbar-label">봉:</span>
              {[
                { value: '1h', label: '1H' },
                { value: '1d', label: '1D' },
                { value: '1wk', label: '1W' },
                { value: '1mo', label: '1Mo' }
              ].map(i => (
                <button
                  key={i.value}
                  className={`draw-btn small ${interval === i.value ? 'active' : ''}`}
                  onClick={() => onIntervalChange(i.value)}
                  disabled={isLoading}
                >
                  {i.label}
                </button>
              ))}
            </div>
            {isLoading && <span className="loading-hint">로딩...</span>}
          </>
        )}

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
          onMouseLeave={handleCanvasMouseLeave}
          onContextMenu={handleCanvasContextMenu}
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

      {/* RSI 서브차트 (전체화면일 때만) */}
      {isFullscreen && indicators.rsi && (
        <div className="rsi-subchart">
          <div className="sub-chart-title">RSI (14)</div>
          <div ref={rsiContainerRef} style={{ width: '100%', height: 120 }} />
        </div>
      )}

      {/* 사용 안내 */}
      <div className="chart-help">
        <span>🖱️ 스크롤: 확대/축소</span>
        <span>👆 드래그: 이동</span>
        <span>⌨️ Shift+드래그: 시간축 확대</span>
      </div>
    </>
  )

  // 단일 구조로 렌더링 (전체화면은 CSS 클래스로 처리)
  return (
    <div className={isFullscreen ? 'trading-chart-fullscreen' : ''}>
      <div className={`trading-chart-wrapper ${isFullscreen ? 'fullscreen' : ''}`}>
        {chartContent}
      </div>
    </div>
  )
}

export default TradingChart

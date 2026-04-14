import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Camera, CameraOff, Info } from 'lucide-react'
import { API_BASE, apiHealth, apiPredict, LABEL_INDEX } from '../lib/api'

const CAPTURE_SIZE = 320          // canvas size sent to the server (jpeg-encoded)
const CLASS_INFO = [
  { label: 'With Mask',      icon: '🥸', color: '#22c55e', bg: 'rgba(34,197,94,0.15)',   key: 'wm'  },
  { label: 'Without Mask',   icon: '😑', color: '#ef4444', bg: 'rgba(239,68,68,0.15)',   key: 'nm'  },
  { label: 'Mask Incorrect', icon: '⚠️', color: '#f59e0b', bg: 'rgba(245,158,11,0.15)',  key: 'inc' },
]

function probsFromResponse(json) {
  const p = json.probabilities || {}
  const arr = [0, 0, 0]
  for (const [name, idx] of Object.entries(LABEL_INDEX)) arr[idx] = +p[name] || 0
  return arr
}

export default function WebcamPage() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const timerRef  = useRef(null)
  const ctx2d     = useRef(null)
  const inflight  = useRef(false)

  const [modelState, setModelState] = useState('loading') // loading | ready | error
  const [loadMsg,    setLoadMsg]    = useState('Connecting to backend…')
  const [modelName,  setModelName]  = useState('–')
  const [running,    setRunning]    = useState(false)
  const [probs,      setProbs]      = useState([0, 0, 0])
  const [bestIdx,    setBestIdx]    = useState(0)
  const [fps,        setFps]        = useState(0)
  const [avgMs,      setAvgMs]      = useState(0)
  const [stats,      setStats]      = useState({ frames: 0, wm: 0, nm: 0, inc: 0, totalMs: 0 })
  const [intervalMs, setIntervalMs] = useState(200)

  const fpsRef = useRef({ count: 0, last: performance.now() })
  const msRef  = useRef({ total: 0, frames: 0 })

  // ── Check backend is up ───────────────────────────────────
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const h = await apiHealth()
        if (cancelled) return
        if (h.status !== 'ok') throw new Error(h.status || 'unknown status')
        setModelName(h.model || 'loaded')
        setModelState('ready')
        setLoadMsg('Backend ready')
      } catch (err) {
        console.error('[Webcam] backend unreachable:', err)
        if (cancelled) return
        setModelState('error')
        setLoadMsg(`Backend not reachable at ${API_BASE}. Start it with:  uvicorn backend.app:app --port 8000`)
      }
    })()
    return () => { cancelled = true }
  }, [])

  // ── Inference loop (calls the backend) ─────────────────────
  const runInference = useCallback(async () => {
    if (inflight.current) return                 // drop frame if previous request still pending
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas || video.readyState < 2) return

    if (!ctx2d.current) ctx2d.current = canvas.getContext('2d', { willReadFrequently: true })
    canvas.width  = CAPTURE_SIZE
    canvas.height = CAPTURE_SIZE
    ctx2d.current.drawImage(video, 0, 0, CAPTURE_SIZE, CAPTURE_SIZE)

    const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', 0.82))
    if (!blob) return

    inflight.current = true
    const t0 = performance.now()
    try {
      const json = await apiPredict(blob)
      const elapsed = performance.now() - t0
      const p = probsFromResponse(json)
      const best = LABEL_INDEX[json.label] ?? p.indexOf(Math.max(...p))

      setProbs(p)
      setBestIdx(best)

      // FPS
      fpsRef.current.count++
      const now = performance.now()
      if (now - fpsRef.current.last >= 1000) {
        setFps(fpsRef.current.count)
        fpsRef.current = { count: 0, last: now }
      }
      msRef.current.total  += elapsed
      msRef.current.frames += 1
      setAvgMs(+(msRef.current.total / msRef.current.frames).toFixed(1))

      setStats(prev => ({
        frames: prev.frames + 1,
        wm:  prev.wm  + (best === 0 ? 1 : 0),
        nm:  prev.nm  + (best === 1 ? 1 : 0),
        inc: prev.inc + (best === 2 ? 1 : 0),
        totalMs: prev.totalMs + elapsed,
      }))
    } catch (err) {
      console.error('[Webcam] /predict failed:', err)
    } finally {
      inflight.current = false
    }
  }, [])

  // ── Camera start/stop ──────────────────────────────────────
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } })
      videoRef.current.srcObject = stream
      await videoRef.current.play()
      setRunning(true)
      setStats({ frames: 0, wm: 0, nm: 0, inc: 0, totalMs: 0 })
      msRef.current = { total: 0, frames: 0 }
      fpsRef.current = { count: 0, last: performance.now() }
      timerRef.current = setInterval(runInference, intervalMs)
    } catch (err) {
      alert(err.name === 'NotAllowedError' ? 'Camera permission denied.' : 'Camera error: ' + err.message)
    }
  }

  const stopCamera = () => {
    clearInterval(timerRef.current)
    videoRef.current?.srcObject?.getTracks().forEach(t => t.stop())
    if (videoRef.current) videoRef.current.srcObject = null
    setRunning(false)
    setProbs([0, 0, 0])
    setFps(0)
    setAvgMs(0)
  }

  useEffect(() => {
    if (running) {
      clearInterval(timerRef.current)
      timerRef.current = setInterval(runInference, intervalMs)
    }
  }, [intervalMs, running, runInference])

  useEffect(() => () => { clearInterval(timerRef.current); stopCamera() }, [])

  const cls = CLASS_INFO[bestIdx]

  return (
    <div className="min-h-screen bg-[#050505] pt-14">
      {/* Hero */}
      <div className="py-12 px-5 text-center bg-[radial-gradient(ellipse_60%_50%_at_50%_0%,#0a1a0a,transparent)]">
        <motion.h1
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className="text-3xl md:text-5xl font-bold tracking-tight text-gradient-green mb-2"
        >
          Live Mask Detection
        </motion.h1>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2 }}
          className="text-white/40 text-base"
        >
          Real-time AI — server inference via FastAPI backend
        </motion.p>
      </div>

      <div className="max-w-6xl mx-auto px-5 pb-20 grid md:grid-cols-[1fr_320px] gap-6 items-start">

        {/* ── Left: Video ── */}
        <div>
          <div className="relative aspect-video bg-zinc-900 rounded-2xl overflow-hidden border border-white/[0.06]">

            {/* Placeholder / loading */}
            <AnimatePresence>
              {!running && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="absolute inset-0 flex flex-col items-center justify-center gap-4 z-10"
                >
                  {modelState === 'loading' && (
                    <>
                      <div className="w-12 h-12 rounded-full border-2 border-white/10 border-t-blue-400 animate-spin" />
                      <p className="text-white/50 text-sm">{loadMsg}</p>
                    </>
                  )}
                  {modelState === 'ready' && (
                    <>
                      <Camera size={48} className="text-white/20" />
                      <p className="text-white/30 text-sm">Click Start Camera below</p>
                    </>
                  )}
                  {modelState === 'error' && (
                    <>
                      <CameraOff size={40} className="text-red-400/60" />
                      <p className="text-red-400 text-sm max-w-xs text-center whitespace-pre-wrap">{loadMsg}</p>
                    </>
                  )}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Video element (always mounted, hidden when not running) */}
            <video
              ref={videoRef}
              className={`w-full h-full object-cover [transform:scaleX(-1)] ${running ? 'block' : 'hidden'}`}
              playsInline
              muted
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* Live result overlay */}
            <AnimatePresence>
              {running && (
                <>
                  {/* Top status */}
                  <motion.div
                    initial={{ opacity: 0, y: -8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute top-0 left-0 right-0 px-4 py-3 flex justify-between items-center z-20"
                    style={{ background: 'linear-gradient(to bottom, rgba(0,0,0,0.7), transparent)' }}
                  >
                    <div className="flex items-center gap-2 text-sm font-semibold">
                      <span className="w-2 h-2 rounded-full bg-red-500 animate-pulse" />
                      LIVE
                    </div>
                    <span className="text-xs text-white/50">{fps} fps · {avgMs}ms</span>
                  </motion.div>

                  {/* Bottom result */}
                  <motion.div
                    key={bestIdx}
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute bottom-0 left-0 right-0 p-5 z-20"
                    style={{ background: 'linear-gradient(to top, rgba(0,0,0,0.85), transparent)' }}
                  >
                    <p className="text-2xl font-bold mb-0.5" style={{ color: cls.color }}>
                      {cls.icon}  {cls.label}
                    </p>
                    <p className="text-sm text-white/50 mb-3">
                      {(probs[bestIdx] * 100).toFixed(1)}% confidence
                    </p>
                    {/* Confidence bar */}
                    <div className="h-1 bg-white/10 rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full"
                        style={{ background: cls.color }}
                        animate={{ width: (probs[bestIdx] * 100) + '%' }}
                        transition={{ duration: 0.2 }}
                      />
                    </div>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>

          {/* Start/stop button */}
          <div className="mt-4 flex gap-3">
            <motion.button
              whileTap={{ scale: 0.97 }}
              onClick={running ? stopCamera : startCamera}
              disabled={modelState !== 'ready'}
              className={`flex-1 py-3.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition-all disabled:opacity-30 disabled:cursor-not-allowed ${
                running
                  ? 'bg-red-500/10 text-red-400 border border-red-500/20 hover:bg-red-500/20'
                  : 'bg-green-500 text-black hover:bg-green-400'
              }`}
            >
              {running ? <><CameraOff size={16} /> Stop Camera</> : <><Camera size={16} /> Start Camera</>}
            </motion.button>

            <select
              value={intervalMs}
              onChange={e => setIntervalMs(+e.target.value)}
              className="px-3 py-2 rounded-xl bg-white/[0.05] border border-white/[0.08] text-sm text-white/70 focus:outline-none"
            >
              <option value={500}>2 fps</option>
              <option value={200}>5 fps</option>
              <option value={100}>10 fps</option>
            </select>
          </div>

          {/* How it works */}
          <div className="mt-4 flex items-start gap-3 p-4 rounded-xl bg-white/[0.03] border border-white/[0.05] text-sm text-white/40">
            <Info size={15} className="mt-0.5 flex-shrink-0 text-blue-400/60" />
            <p>
              Each frame is captured at {CAPTURE_SIZE}×{CAPTURE_SIZE}, JPEG-encoded, and POSTed to{' '}
              <code className="text-blue-300/70 bg-blue-500/10 px-1 rounded text-xs">{API_BASE}/predict</code>.
              The FastAPI backend runs the trained Keras model server-side and returns the
              class probabilities. Override the URL via <code className="text-blue-300/70 bg-blue-500/10 px-1 rounded text-xs">VITE_API_URL</code> in <code className="text-blue-300/70 bg-blue-500/10 px-1 rounded text-xs">.env.local</code>.
            </p>
          </div>
        </div>

        {/* ── Right: Side panel ── */}
        <div className="space-y-4">

          {/* Class probabilities */}
          <div className="glass rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">Class Probabilities</h3>
            <div className="space-y-4">
              {CLASS_INFO.map((c, i) => (
                <div key={c.key}>
                  <div className="flex items-center justify-between mb-1.5">
                    <div className="flex items-center gap-2 text-sm">
                      <span>{c.icon}</span>
                      <span className="text-white/70">{c.label}</span>
                    </div>
                    <span className="text-sm font-bold" style={{ color: c.color }}>
                      {(probs[i] * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 bg-white/[0.07] rounded-full overflow-hidden">
                    <motion.div
                      className="h-full rounded-full"
                      style={{ background: c.color }}
                      animate={{ width: (probs[i] * 100) + '%' }}
                      transition={{ duration: 0.2 }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Session stats */}
          <div className="glass rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">Session Stats</h3>
            <div className="space-y-2">
              {[
                { label: 'Frames analyzed', val: stats.frames, color: 'text-white' },
                { label: 'With Mask',       val: stats.wm,     color: 'text-green-400' },
                { label: 'Without Mask',    val: stats.nm,     color: 'text-red-400' },
                { label: 'Mask Incorrect',  val: stats.inc,    color: 'text-yellow-400' },
                { label: 'Avg round-trip',  val: avgMs ? avgMs + 'ms' : '—', color: 'text-blue-400' },
              ].map(s => (
                <div key={s.label} className="flex justify-between py-2 border-b border-white/[0.04] text-sm">
                  <span className="text-white/40">{s.label}</span>
                  <span className={`font-semibold ${s.color}`}>{s.val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Backend info */}
          <div className="glass rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">Backend</h3>
            <div className="space-y-2 text-sm">
              {[
                ['API URL',    API_BASE],
                ['Model',      modelName],
                ['Transport',  'multipart/form-data'],
                ['Capture',    `${CAPTURE_SIZE}×${CAPTURE_SIZE} JPEG`],
                ['Endpoint',   'POST /predict'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between py-2 border-b border-white/[0.04] gap-4">
                  <span className="text-white/35 flex-shrink-0">{k}</span>
                  <span className="text-white/70 font-medium text-right truncate">{v}</span>
                </div>
              ))}
            </div>
          </div>

          <Link
            to="/"
            className="flex items-center justify-center gap-2 py-3 rounded-xl glass text-sm text-white/50 hover:text-white transition-colors"
          >
            ← Back to Dashboard
          </Link>
        </div>
      </div>
    </div>
  )
}

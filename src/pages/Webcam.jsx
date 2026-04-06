import { useState, useEffect, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import { Camera, CameraOff, Info } from 'lucide-react'

const BASE = import.meta.env.BASE_URL
const MODEL_PATH = `${BASE}model/model.onnx`
const IMG_SIZE = 224
const CLASS_INFO = [
  { label: 'With Mask',      icon: '🥸', color: '#22c55e', bg: 'rgba(34,197,94,0.15)',   key: 'wm'  },
  { label: 'Without Mask',   icon: '😑', color: '#ef4444', bg: 'rgba(239,68,68,0.15)',   key: 'nm'  },
  { label: 'Mask Incorrect', icon: '⚠️', color: '#f59e0b', bg: 'rgba(245,158,11,0.15)',  key: 'inc' },
]

function softmax(arr) {
  const max = Math.max(...arr)
  const exps = arr.map(x => Math.exp(x - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map(x => x / sum)
}

export default function WebcamPage() {
  const videoRef  = useRef(null)
  const canvasRef = useRef(null)
  const ortRef    = useRef(null)
  const timerRef  = useRef(null)
  const ctx2d     = useRef(null)

  const [modelState, setModelState] = useState('loading') // loading | ready | error
  const [loadMsg,    setLoadMsg]    = useState('Initializing…')
  const [loadPct,    setLoadPct]    = useState(0)
  const [running,    setRunning]    = useState(false)
  const [probs,      setProbs]      = useState([0, 0, 0])
  const [bestIdx,    setBestIdx]    = useState(0)
  const [fps,        setFps]        = useState(0)
  const [avgMs,      setAvgMs]      = useState(0)
  const [stats,      setStats]      = useState({ frames: 0, wm: 0, nm: 0, inc: 0, totalMs: 0 })
  const [intervalMs, setIntervalMs] = useState(100)

  const fpsRef   = useRef({ count: 0, last: performance.now() })
  const msRef    = useRef({ total: 0, frames: 0 })

  // ── Load ORT + model ───────────────────────────────────────
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        setLoadMsg('Loading ONNX Runtime…')
        setLoadPct(10)
        const ort = await import('onnxruntime-web')
        if (cancelled) return

        // Serve WASM from same origin (avoids version mismatch + CORS).
        // Force single-threaded WASM (GitHub Pages doesn't set COOP/COEP for SharedArrayBuffer).
        ort.env.wasm.wasmPaths = `${BASE}ort/`
        ort.env.wasm.numThreads = 1
        ort.env.wasm.proxy = false

        setLoadMsg('Downloading model (9.9 MB)…')
        setLoadPct(30)

        // Fetch model as ArrayBuffer with progress tracking
        const resp = await fetch(MODEL_PATH)
        if (!resp.ok) throw new Error(`Model fetch failed: HTTP ${resp.status}`)
        const total  = +(resp.headers.get('content-length') || 0)
        const reader = resp.body.getReader()
        const chunks = []
        let received = 0
        while (true) {
          const { done, value } = await reader.read()
          if (done) break
          chunks.push(value)
          received += value.length
          if (total) setLoadPct(30 + Math.round((received / total) * 50))
        }
        if (cancelled) return
        const buf = new Uint8Array(received)
        let offset = 0
        for (const c of chunks) { buf.set(c, offset); offset += c.length }

        setLoadMsg('Initializing model…')
        setLoadPct(85)
        const session = await ort.InferenceSession.create(buf.buffer, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        })
        if (cancelled) return

        const iname = session.inputNames?.[0] ?? 'input_layer_5'

        // Warm-up
        setLoadMsg('Warming up…')
        setLoadPct(95)
        const dummy = new ort.Tensor(
          'float32',
          new Float32Array(1 * IMG_SIZE * IMG_SIZE * 3),
          [1, IMG_SIZE, IMG_SIZE, 3]
        )
        await session.run({ [iname]: dummy })
        if (cancelled) return

        ortRef.current = { session, ort, iname }
        setLoadPct(100)
        setModelState('ready')
        setLoadMsg('Model ready')
      } catch (err) {
        console.error('[Webcam] Model load failed:', err)
        if (cancelled) return
        setModelState('error')
        setLoadMsg(String(err.message ?? err).slice(0, 140))
      }
    })()
    return () => { cancelled = true }
  }, [])

  // ── Inference loop ─────────────────────────────────────────
  const runInference = useCallback(async () => {
    const { session, ort, iname } = ortRef.current ?? {}
    if (!session || !videoRef.current || !canvasRef.current) return
    const video = videoRef.current
    if (video.readyState < 2) return

    const canvas = canvasRef.current
    if (!ctx2d.current) ctx2d.current = canvas.getContext('2d', { willReadFrequently: true })
    canvas.width = IMG_SIZE
    canvas.height = IMG_SIZE
    ctx2d.current.drawImage(video, 0, 0, IMG_SIZE, IMG_SIZE)

    const pixels = ctx2d.current.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data
    const tensor = new Float32Array(1 * IMG_SIZE * IMG_SIZE * 3)
    for (let i = 0, t = 0; i < IMG_SIZE * IMG_SIZE; i++) {
      const b = i * 4
      tensor[t++] = pixels[b]     / 127.5 - 1
      tensor[t++] = pixels[b + 1] / 127.5 - 1
      tensor[t++] = pixels[b + 2] / 127.5 - 1
    }

    const t0 = performance.now()
    const out = await session.run({ [iname]: new ort.Tensor('float32', tensor, [1, IMG_SIZE, IMG_SIZE, 3]) })
    const elapsed = performance.now() - t0
    const logits  = Array.from(out[session.outputNames[0]].data)
    const p = softmax(logits)
    const best = p.indexOf(Math.max(...p))

    setProbs(p)
    setBestIdx(best)

    // FPS
    fpsRef.current.count++
    const now = performance.now()
    if (now - fpsRef.current.last >= 1000) {
      setFps(fpsRef.current.count)
      fpsRef.current = { count: 0, last: now }
    }
    // avg ms
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
  }, [])

  // ── Camera start/stop ───────────────────────────────────────
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
          Real-time AI — 89.6% accurate — runs entirely in your browser
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
                      <div className="w-48 h-1 bg-white/[0.07] rounded-full overflow-hidden">
                        <motion.div
                          className="h-full bg-blue-500 rounded-full"
                          animate={{ width: loadPct + '%' }}
                          transition={{ duration: 0.5 }}
                        />
                      </div>
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
                      <p className="text-red-400 text-sm max-w-xs text-center">{loadMsg}</p>
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
              <option value={200}>5 fps</option>
              <option value={100}>10 fps</option>
              <option value={50}>20 fps</option>
            </select>
          </div>

          {/* How it works */}
          <div className="mt-4 flex items-start gap-3 p-4 rounded-xl bg-white/[0.03] border border-white/[0.05] text-sm text-white/40">
            <Info size={15} className="mt-0.5 flex-shrink-0 text-blue-400/60" />
            <p>
              Each frame is resized to 224×224, normalized via <code className="text-blue-300/70 bg-blue-500/10 px-1 rounded text-xs">x / 127.5 − 1</code>,
              then fed to our <span className="text-white/60 font-medium">89.6% accurate MobileNetV2</span> ONNX model.
              Inference runs in your browser via ONNX Runtime Web — no data leaves your device.
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
                { label: 'Avg inference',   val: avgMs ? avgMs + 'ms' : '—', color: 'text-blue-400' },
              ].map(s => (
                <div key={s.label} className="flex justify-between py-2 border-b border-white/[0.04] text-sm">
                  <span className="text-white/40">{s.label}</span>
                  <span className={`font-semibold ${s.color}`}>{s.val}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Model info */}
          <div className="glass rounded-2xl p-5">
            <h3 className="text-sm font-semibold text-white/60 uppercase tracking-wider mb-4">Model Info</h3>
            <div className="space-y-2 text-sm">
              {[
                ['Architecture', 'MobileNetV2 Fine-Tuned'],
                ['Format', 'ONNX (opset 13)'],
                ['Input', '224 × 224 RGB'],
                ['Preprocessing', 'x / 127.5 − 1.0'],
                ['Output', '3-class softmax'],
                ['Accuracy', '89.58%'],
                ['Model size', '~9.9 MB'],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between py-2 border-b border-white/[0.04]">
                  <span className="text-white/35">{k}</span>
                  <span className="text-white/70 font-medium">{v}</span>
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

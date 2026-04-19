import { useState, useEffect, useRef } from 'react'
import { motion, AnimatePresence, useInView } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { ALL_MODELS, EXP1_MODELS, EXP2_MODELS, EXP3_MODELS } from '../data/models'

// ── Animated number counter ───────────────────────────────────────────────────
function Counter({ to, suffix = '', decimals = 0, duration = 1.4 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })
  useEffect(() => {
    if (!inView) return
    let start = null
    const step = ts => {
      if (!start) start = ts
      const p = Math.min((ts - start) / (duration * 1000), 1)
      const e = 1 - Math.pow(1 - p, 3)
      setVal(+(to * e).toFixed(decimals))
      if (p < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [inView, to, decimals, duration])
  return <span ref={ref}>{val}{suffix}</span>
}

// ── Scroll reveal ─────────────────────────────────────────────────────────────
function Reveal({ children, delay = 0, className = '' }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-60px' })
  return (
    <motion.div ref={ref} className={className}
      initial={{ opacity: 0, y: 28 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
    >{children}</motion.div>
  )
}

// ── Experiment model tabs ─────────────────────────────────────────────────────
function ModelTabs({ models, id }) {
  const [active, setActive] = useState(0)
  const m = models[active]
  return (
    <div>
      {/* tabs */}
      <div className="flex gap-1 p-1 bg-white/[0.04] rounded-xl border border-white/[0.06] mb-5 overflow-x-auto">
        {models.map((model, i) => (
          <button key={model.key} onClick={() => setActive(i)}
            className={`relative flex-1 min-w-max px-3 py-2 rounded-lg text-xs font-medium transition-all ${
              active === i ? 'text-white' : 'text-white/40 hover:text-white/70'
            }`}>
            {active === i && (
              <motion.div layoutId={`tab-${id}`}
                className="absolute inset-0 rounded-lg bg-white/[0.08]"
                transition={{ type: 'spring', bounce: 0.2, duration: 0.35 }} />
            )}
            <span className="relative">{model.label}</span>
            {model.badge && (
              <span className="relative ml-1.5 text-[10px] font-semibold px-1.5 py-0.5 rounded-full"
                style={{ background: `${model.color}25`, color: model.color }}>
                {model.badge}
              </span>
            )}
          </button>
        ))}
      </div>

      {/* content */}
      <AnimatePresence mode="wait">
        <motion.div key={active}
          initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }} transition={{ duration: 0.2 }}>
          {/* metrics */}
          <div className="flex gap-6 flex-wrap mb-4">
            {[
              { label: 'Accuracy', val: m.accuracy, color: m.color },
              { label: 'F1 Score', val: m.f1 },
              ...(m.params ? [{ label: 'Parameters', val: m.params }] : []),
              { label: 'Train Time', val: m.time },
            ].map(s => (
              <div key={s.label}>
                <p className="text-[10px] text-white/35 uppercase tracking-widest mb-1">{s.label}</p>
                <p className="text-xl font-bold" style={{ color: s.color || 'white' }}>{s.val}</p>
              </div>
            ))}
          </div>
          <p className="text-white/45 text-sm mb-5">{m.description}</p>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[
              { src: m.images.history, label: 'Training History' },
              { src: m.images.cm, label: 'Confusion Matrix' },
            ].map(img => (
              <div key={img.label} className="rounded-xl overflow-hidden bg-white/[0.03] border border-white/[0.06]">
                <p className="text-[10px] text-white/25 px-3 pt-2.5 uppercase tracking-wider">{img.label}</p>
                <img src={img.src} alt={img.label} className="w-full" />
              </div>
            ))}
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

// ── Custom colored bar shape (avoids deprecated Cell) ────────────────────────
function ColoredBar({ x, y, width, height, color }) {
  return <rect x={x} y={y} width={width} height={Math.max(0, height)}
    fill={color} fillOpacity={0.85} rx={5} ry={5} />
}

// ── Bar chart tooltip ─────────────────────────────────────────────────────────
function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-xl px-3 py-2 text-xs bg-[#111] border border-white/10 shadow-xl">
      <p className="text-white/50 mb-0.5">{label}</p>
      <p className="font-bold text-white">{payload[0].value.toFixed(2)}%</p>
    </div>
  )
}

// ── Experiment badge ──────────────────────────────────────────────────────────
function ExpBadge({ label, color }) {
  return (
    <span className="px-2 py-0.5 rounded-md text-[10px] font-semibold uppercase tracking-wide"
      style={{ background: `${color}18`, color }}>
      {label}
    </span>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
export default function Dashboard() {
  const barData = ALL_MODELS.map(m => ({
    name: m.short,
    accuracy: +(m.accuracy * 100).toFixed(2),
    color: m.color,
  }))

  return (
    <>
      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-5 overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_70%_55%_at_50%_-5%,#0c1a35,transparent)]" />
        <div className="absolute inset-0 opacity-[0.025]"
          style={{ backgroundImage: 'linear-gradient(rgba(255,255,255,0.6) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.6) 1px,transparent 1px)', backgroundSize: '72px 72px' }} />

        <div className="relative z-10 max-w-3xl w-full">
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.1 }}>
            <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full border border-white/10 bg-white/[0.04] text-xs font-medium text-blue-300 mb-7">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Deep Learning · 3 Experiments · 9 Models
            </div>
          </motion.div>

          <motion.h1 initial={{ opacity: 0, y: 28 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, delay: 0.2 }}
            className="text-gradient text-6xl md:text-8xl font-bold tracking-tight leading-tight mb-5">
            Face Mask<br />Detection
          </motion.h1>

          <motion.p initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.35 }}
            className="text-base md:text-lg text-white/45 max-w-lg mx-auto mb-9 leading-relaxed">
            Custom CNNs, transfer learning, and state-of-the-art architectures benchmarked
            on a 3-class face mask dataset.
          </motion.p>

          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.48 }}
            className="flex gap-3 justify-center mb-14">
            <a href="#results"
              className="px-6 py-2.5 rounded-full bg-blue-500 hover:bg-blue-400 text-white text-sm font-semibold transition-all hover:scale-105 active:scale-95 shadow-lg shadow-blue-500/20">
              View Results
            </a>
            <Link to="/webcam"
              className="px-6 py-2.5 rounded-full border border-white/10 bg-white/[0.05] hover:bg-white/[0.09] text-white text-sm font-semibold transition-all hover:scale-105 active:scale-95 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              Live Demo
            </Link>
          </motion.div>

          {/* stats */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.5, delay: 0.6 }}
            className="grid grid-cols-2 md:grid-cols-4 gap-px bg-white/[0.06] rounded-2xl overflow-hidden border border-white/[0.06]">
            {[
              { label: 'Best Accuracy', value: 91.4, suffix: '%', color: 'text-green-400' },
              { label: 'Models Trained', value: 9,   suffix: '',  color: 'text-blue-400' },
              { label: 'Experiments',   value: 3,    suffix: '',  color: 'text-purple-400' },
              { label: 'Face Images',   value: 4072, suffix: '',  color: 'text-orange-400' },
            ].map(s => (
              <div key={s.label} className="bg-[#0a0a0a] px-6 py-5 text-center">
                <p className={`text-2xl md:text-3xl font-bold ${s.color}`}>
                  <Counter to={s.value} suffix={s.suffix} decimals={s.value % 1 !== 0 ? 1 : 0} />
                </p>
                <p className="text-[10px] text-white/35 mt-1 uppercase tracking-wider">{s.label}</p>
              </div>
            ))}
          </motion.div>
        </div>

        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 1.4 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <motion.div animate={{ y: [0, 8, 0] }} transition={{ duration: 2, repeat: Infinity }}
            className="w-5 h-8 rounded-full border border-white/15 flex items-start justify-center pt-1.5">
            <div className="w-1 h-2 rounded-full bg-white/30" />
          </motion.div>
        </motion.div>
      </section>

      {/* ── DATASET ──────────────────────────────────────────────────────── */}
      <section className="py-24 px-5 bg-white">
        <div className="max-w-5xl mx-auto">
          <Reveal>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-blue-500 mb-3">Dataset</p>
            <h2 className="text-4xl font-bold text-gray-900 mb-4">3 Classes, 4 072 Face Crops</h2>
            <p className="text-gray-400 text-base mb-10 max-w-xl">
              Pascal VOC XML annotations parsed, faces cropped with 10 px padding, split 70/15/15 stratified.
            </p>
          </Reveal>
          <div className="grid md:grid-cols-3 gap-4 mb-10">
            {[
              { icon: '😷', label: 'With Mask',     pct: 79.4, count: '3 232', color: '#22c55e' },
              { icon: '😑', label: 'Without Mask',  pct: 17.6, count: '717',   color: '#ef4444' },
              { icon: '⚠️', label: 'Mask Incorrect',pct: 3.0,  count: '123',   color: '#f59e0b' },
            ].map((c, i) => (
              <Reveal key={c.label} delay={i * 0.08}>
                <div className="rounded-2xl p-7 text-center border"
                  style={{ background: `${c.color}09`, borderColor: `${c.color}22` }}>
                  <span className="text-4xl block mb-3">{c.icon}</span>
                  <p className="text-3xl font-bold mb-1" style={{ color: c.color }}>{c.pct}%</p>
                  <p className="text-gray-700 font-semibold mb-1">{c.label}</p>
                  <p className="text-gray-400 text-sm">{c.count} images</p>
                </div>
              </Reveal>
            ))}
          </div>
          <Reveal>
            <div className="grid md:grid-cols-2 gap-8 items-start">
              <img src="images/sample_images.png" alt="Sample images" className="rounded-2xl shadow-xl w-full" />
              <img src="images/class_distribution.png" alt="Class distribution" className="rounded-2xl shadow-xl w-full" />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── RESULTS LEADERBOARD ──────────────────────────────────────────── */}
      <section id="results" className="py-24 px-5">
        <div className="max-w-5xl mx-auto">
          <Reveal>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-blue-400 mb-3">All Results</p>
            <h2 className="text-4xl font-bold text-white mb-4">Model Leaderboard</h2>
            <p className="text-white/40 text-base mb-10">9 models ranked by test accuracy.</p>
          </Reveal>

          {/* table */}
          <Reveal>
            <div className="overflow-x-auto rounded-2xl border border-white/[0.07] mb-10">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06] bg-white/[0.02]">
                    {['#', 'Model', 'Experiment', 'Accuracy', 'F1', 'Precision', 'Time'].map(h => (
                      <th key={h} className="text-left px-4 py-3 text-[10px] font-semibold text-white/30 uppercase tracking-wider">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {ALL_MODELS.map((m, i) => (
                    <motion.tr key={m.key}
                      initial={{ opacity: 0, x: -12 }} whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }} transition={{ duration: 0.35, delay: i * 0.04 }}
                      className="border-b border-white/[0.04] hover:bg-white/[0.02] transition-colors">
                      <td className="px-4 py-3.5 text-white/25 font-mono text-xs">{m.rank}</td>
                      <td className="px-4 py-3.5">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: m.color }} />
                          <span className="font-medium text-white">{m.name}</span>
                          {m.isWinner && <span className="text-yellow-400 text-sm">🏆</span>}
                        </div>
                      </td>
                      <td className="px-4 py-3.5">
                        <ExpBadge label={m.experiment} color={m.expColor} />
                      </td>
                      <td className="px-4 py-3.5 font-bold" style={{ color: m.color }}>
                        {(m.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-3.5 text-white/60">{(m.f1 * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-white/60">{(m.precision * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-white/35 font-mono text-xs">
                        {m.time >= 3600 ? (m.time / 3600).toFixed(1) + ' h' : Math.round(m.time) + ' s'}
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>

          {/* accuracy bar chart */}
          <Reveal>
            <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-6 mb-10">
              <p className="text-sm font-semibold text-white mb-5">Accuracy by Model</p>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={barData} margin={{ top: 4, right: 8, bottom: 36, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                  <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }}
                    angle={-35} textAnchor="end" interval={0} />
                  <YAxis domain={[60, 100]} tick={{ fill: 'rgba(255,255,255,0.3)', fontSize: 10 }}
                    tickFormatter={v => v + '%'} />
                  <Tooltip content={<ChartTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
                  <Bar dataKey="accuracy" shape={<ColoredBar />} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Reveal>

          {/* winner card */}
          <Reveal>
            <div className="rounded-2xl border border-blue-500/20 p-8 md:p-10 relative overflow-hidden"
              style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.07), rgba(34,197,94,0.04))' }}>
              <div className="absolute top-0 right-0 w-56 h-56 rounded-full opacity-[0.07]"
                style={{ background: 'radial-gradient(circle, #3b82f6, transparent)', transform: 'translate(30%,-30%)' }} />
              <div className="relative">
                <p className="text-2xl mb-2">🏆</p>
                <h3 className="text-2xl font-bold text-white mb-1">MobileNetV2 Fine-Tuned</h3>
                <p className="text-white/40 text-sm mb-7 max-w-md">
                  Two-phase transfer learning achieves the highest accuracy with efficient training time.
                </p>
                <div className="flex gap-10 flex-wrap">
                  {[
                    { label: 'Accuracy',  val: '91.37%', color: '#22c55e' },
                    { label: 'F1 Score',  val: '90.63%', color: '#3b82f6' },
                    { label: 'Precision', val: '91.10%', color: '#a855f7' },
                    { label: 'Train Time',val: '108 s',  color: '#f59e0b' },
                  ].map(s => (
                    <div key={s.label}>
                      <p className="text-3xl font-bold" style={{ color: s.color }}>{s.val}</p>
                      <p className="text-[10px] text-white/35 mt-1 uppercase tracking-wider">{s.label}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── EXPERIMENTS ──────────────────────────────────────────────────── */}
      <section id="experiments" className="py-24 px-5 bg-[#060606]">
        <div className="max-w-5xl mx-auto">
          <Reveal>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-blue-400 mb-3">Experiments</p>
            <h2 className="text-4xl font-bold text-white mb-12">Three Approaches</h2>
          </Reveal>

          {/* Experiment 1 */}
          <Reveal className="mb-6">
            <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-7">
              <div className="flex items-center gap-3 mb-2">
                <span className="px-2.5 py-1 rounded-lg bg-blue-500/10 text-blue-400 text-[10px] font-bold uppercase tracking-wider">
                  Experiment 1
                </span>
                <h3 className="text-xl font-bold text-white">Custom CNN Architectures</h3>
              </div>
              <p className="text-white/40 text-sm mb-6">
                Three hand-designed CNNs of increasing depth, each tested at two learning rates — 6 total runs.
                Best result: <span className="text-white/70 font-medium">Arch-C at 89.09%</span>.
              </p>
              <ModelTabs models={EXP1_MODELS} id="exp1" />
            </div>
          </Reveal>

          {/* Experiment 2 */}
          <Reveal className="mb-6" delay={0.05}>
            <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-7">
              <div className="flex items-center gap-3 mb-2">
                <span className="px-2.5 py-1 rounded-lg bg-green-500/10 text-green-400 text-[10px] font-bold uppercase tracking-wider">
                  Experiment 2
                </span>
                <h3 className="text-xl font-bold text-white">Transfer Learning</h3>
              </div>
              <p className="text-white/40 text-sm mb-6">
                MobileNetV2 and VGG16 with frozen feature extraction and progressive fine-tuning.
                Best result: <span className="text-white/70 font-medium">MNv2 Fine-Tuned at 91.37%</span> — best overall.
              </p>
              <ModelTabs models={EXP2_MODELS} id="exp2" />
            </div>
          </Reveal>

          {/* Experiment 3 */}
          <Reveal delay={0.1}>
            <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-7">
              <div className="flex items-center gap-3 mb-2">
                <span className="px-2.5 py-1 rounded-lg bg-purple-500/10 text-purple-400 text-[10px] font-bold uppercase tracking-wider">
                  Experiment 3
                </span>
                <h3 className="text-xl font-bold text-white">State-of-the-Art Models</h3>
              </div>
              <p className="text-white/40 text-sm mb-6">
                EfficientNetB0 (scratch vs ImageNet pretrained) and a custom Vision Transformer.
                Best result: <span className="text-white/70 font-medium">EfficientNet Pretrained at 90.72%</span>.
              </p>
              <ModelTabs models={EXP3_MODELS} id="exp3" />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── KEY FINDINGS ─────────────────────────────────────────────────── */}
      <section className="py-24 px-5 bg-white">
        <div className="max-w-5xl mx-auto">
          <Reveal>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-blue-500 mb-3">Key Findings</p>
            <h2 className="text-4xl font-bold text-gray-900 mb-10">What We Learned</h2>
          </Reveal>
          <div className="grid md:grid-cols-2 gap-4">
            {[
              {
                n: '01', color: '#3b82f6',
                title: 'Pretrained > Scratch',
                body: 'ImageNet init consistently outperforms random init. EfficientNet pretrained matches scratch accuracy in 3.6× less time.',
              },
              {
                n: '02', color: '#22c55e',
                title: 'Fine-Tuning Pays Off',
                body: 'MobileNetV2 fine-tuning beats frozen feature extraction by 0.33% while training faster — unfreezing top layers matters.',
              },
              {
                n: '03', color: '#a855f7',
                title: 'LR is Critical for CNNs',
                body: 'The same architecture with LR 10× too small (0.0001) can lose 22 percentage points. lr=0.001 was the sweet spot.',
              },
              {
                n: '04', color: '#f59e0b',
                title: 'ViT Needs More Data',
                body: 'Custom ViT scored only 79.15% — transformers need large datasets to outperform CNNs. Our 4K dataset is too small.',
              },
            ].map((f, i) => (
              <Reveal key={f.n} delay={i * 0.07}>
                <div className="rounded-2xl p-7 border border-gray-100 h-full">
                  <p className="text-4xl font-black mb-4 opacity-15" style={{ color: f.color }}>{f.n}</p>
                  <h4 className="text-lg font-bold text-gray-900 mb-2">{f.title}</h4>
                  <p className="text-gray-500 text-sm leading-relaxed">{f.body}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      {/* ── LIVE DEMO CTA ─────────────────────────────────────────────────── */}
      <section className="py-24 px-5">
        <div className="max-w-3xl mx-auto text-center">
          <Reveal>
            <p className="text-[11px] font-semibold tracking-widest uppercase text-blue-400 mb-3">Live Demo</p>
            <h2 className="text-4xl font-bold text-white mb-4">Try It on Your Webcam</h2>
            <p className="text-white/40 text-base mb-8 max-w-md mx-auto">
              Real-time face mask detection powered by the best trained model via FastAPI.
            </p>
            <Link to="/webcam"
              className="inline-flex items-center gap-3 px-8 py-4 rounded-full bg-green-500 hover:bg-green-400 text-black font-bold text-base transition-all hover:scale-105 active:scale-95 shadow-xl shadow-green-500/20">
              <span className="w-2.5 h-2.5 rounded-full bg-black/40 animate-pulse" />
              Launch Webcam Detection
            </Link>
            <p className="text-white/25 text-xs mt-4">Requires the FastAPI backend running on localhost:8000</p>
          </Reveal>
        </div>
      </section>

      {/* ── FOOTER ────────────────────────────────────────────────────────── */}
      <footer className="border-t border-white/[0.05] py-8 px-5 text-center">
        <p className="text-white/20 text-xs">
          Face Mask Detection · Deep Learning Group Project · TensorFlow + React + FastAPI
        </p>
      </footer>
    </>
  )
}

import { useState, useEffect, useRef } from 'react'
import { motion, useInView, AnimatePresence } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  BarChart, Bar, RadarChart, Radar, PolarGrid, PolarAngleAxis,
  PolarRadiusAxis, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts'
import { MODELS, EXP2_MODELS, EXP3_MODELS } from '../data/models'
import { Trophy, Zap, Database, Brain, ChevronRight, ExternalLink } from 'lucide-react'

// ── Animated counter ──────────────────────────────────────────
function Counter({ to, suffix = '', decimals = 0, duration = 1.5 }) {
  const [val, setVal] = useState(0)
  const ref = useRef(null)
  const inView = useInView(ref, { once: true })

  useEffect(() => {
    if (!inView) return
    let start = null
    const step = ts => {
      if (!start) start = ts
      const prog = Math.min((ts - start) / (duration * 1000), 1)
      const ease = 1 - Math.pow(1 - prog, 3)
      setVal(+(to * ease).toFixed(decimals))
      if (prog < 1) requestAnimationFrame(step)
    }
    requestAnimationFrame(step)
  }, [inView, to, decimals, duration])

  return <span ref={ref}>{val}{suffix}</span>
}

// ── Scroll reveal wrapper ─────────────────────────────────────
function Reveal({ children, delay = 0, className = '' }) {
  const ref = useRef(null)
  const inView = useInView(ref, { once: true, margin: '-80px' })
  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 32 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.7, delay, ease: [0.25, 0.46, 0.45, 0.94] }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

// ── Section header ────────────────────────────────────────────
function SectionHeader({ eyebrow, title, subtitle, light }) {
  return (
    <div className="mb-12">
      <p className="text-xs font-semibold tracking-widest uppercase text-blue-400 mb-3">{eyebrow}</p>
      <h2 className={`text-4xl md:text-5xl font-bold tracking-tight mb-4 ${light ? 'text-gray-900' : 'text-white'}`}>
        {title}
      </h2>
      {subtitle && (
        <p className={`text-lg max-w-2xl leading-relaxed ${light ? 'text-gray-500' : 'text-white/50'}`}>
          {subtitle}
        </p>
      )}
    </div>
  )
}

// ── Experiment tab section ────────────────────────────────────
function ExpTabs({ models, accentIndex = 0 }) {
  const [active, setActive] = useState(0)
  const m = models[active]

  return (
    <div>
      {/* Tab bar */}
      <div className="flex gap-1 p-1 bg-white/[0.04] rounded-xl border border-white/[0.06] mb-6 overflow-x-auto">
        {models.map((model, i) => (
          <button
            key={model.key}
            onClick={() => setActive(i)}
            className={`relative flex-1 min-w-max px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
              active === i ? 'text-white' : 'text-white/40 hover:text-white/70'
            }`}
          >
            {active === i && (
              <motion.div
                layoutId={`tab-bg-${accentIndex}`}
                className="absolute inset-0 rounded-lg"
                style={{ background: 'rgba(255,255,255,0.08)' }}
                transition={{ type: 'spring', bounce: 0.2, duration: 0.4 }}
              />
            )}
            <span className="relative">{model.label}</span>
            {model.badge && (
              <span className="relative ml-1.5 text-xs text-yellow-400">{model.badge}</span>
            )}
          </button>
        ))}
      </div>

      {/* Content */}
      <AnimatePresence mode="wait">
        <motion.div
          key={active}
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.25 }}
        >
          {/* Metrics row */}
          <div className="flex gap-6 mb-6 flex-wrap">
            <div>
              <p className="text-xs text-white/40 uppercase tracking-wider mb-1">Accuracy</p>
              <p className="text-2xl font-bold" style={{ color: m.accent }}>{m.accuracy}</p>
            </div>
            <div>
              <p className="text-xs text-white/40 uppercase tracking-wider mb-1">F1 Score</p>
              <p className="text-2xl font-bold text-white">{m.f1}</p>
            </div>
            <div>
              <p className="text-xs text-white/40 uppercase tracking-wider mb-1">Training</p>
              <p className="text-2xl font-bold text-white">{m.time}</p>
            </div>
          </div>
          <p className="text-white/50 text-sm mb-6">{m.description}</p>

          {/* Images */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="glass rounded-xl overflow-hidden">
              <p className="text-xs text-white/30 px-3 pt-3 uppercase tracking-wider">Training History</p>
              <img
                src={m.images.history}
                alt="Training history"
                className="w-full object-cover"
              />
            </div>
            <div className="glass rounded-xl overflow-hidden">
              <p className="text-xs text-white/30 px-3 pt-3 uppercase tracking-wider">Confusion Matrix</p>
              <img
                src={m.images.cm}
                alt="Confusion matrix"
                className="w-full object-cover"
              />
            </div>
          </div>
        </motion.div>
      </AnimatePresence>
    </div>
  )
}

// ── Custom tooltip ────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div className="glass rounded-xl px-3 py-2 text-sm">
      <p className="text-white/60 text-xs mb-1">{label}</p>
      <p className="font-bold text-white">{payload[0].value.toFixed(2)}%</p>
    </div>
  )
}

// ── Main Dashboard ────────────────────────────────────────────
export default function Dashboard() {
  const barData = MODELS.map(m => ({
    name: m.short,
    accuracy: +(m.accuracy * 100).toFixed(2),
    color: m.color,
  }))

  const radarData = ['Accuracy', 'F1', 'Precision', 'Recall'].map(metric => {
    const key = { Accuracy: 'accuracy', F1: 'f1', Precision: 'precision', Recall: 'recall' }[metric]
    const row = { metric }
    MODELS.slice(0, 4).forEach(m => { row[m.short] = +(m[key] * 100).toFixed(1) })
    return row
  })

  return (
    <>
      {/* ── HERO ── */}
      <section className="relative min-h-screen flex flex-col items-center justify-center text-center px-5 overflow-hidden">
        {/* Background */}
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_-10%,#0d1f3c,transparent)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_40%_40%_at_75%_30%,rgba(59,130,246,0.08),transparent)]" />
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_35%_35%_at_25%_60%,rgba(34,197,94,0.06),transparent)]" />

        {/* Grid lines */}
        <div className="absolute inset-0 opacity-[0.03]"
          style={{backgroundImage:'linear-gradient(rgba(255,255,255,0.5) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,0.5) 1px,transparent 1px)',backgroundSize:'80px 80px'}} />

        <div className="relative z-10 max-w-4xl">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full glass text-xs font-medium text-blue-300 mb-8">
              <span className="w-1.5 h-1.5 rounded-full bg-green-400 animate-pulse" />
              Deep Learning Research &mdash; 6 Students &mdash; 3 Experiments
            </div>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.7, delay: 0.2 }}
            className="text-gradient text-6xl md:text-8xl font-bold tracking-tight leading-[1.02] mb-6"
          >
            Face Mask<br />Detection
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.35 }}
            className="text-lg md:text-xl text-white/50 max-w-xl mx-auto mb-10 leading-relaxed"
          >
            Comparing custom CNNs, transfer learning, and state-of-the-art
            architectures for real-time mask classification.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
            className="flex gap-3 justify-center flex-wrap mb-16"
          >
            <a href="#results"
              className="px-6 py-3 rounded-full bg-blue-500 hover:bg-blue-400 text-white text-sm font-semibold transition-all hover:scale-105 active:scale-95 shadow-lg shadow-blue-500/25">
              View Results
            </a>
            <Link to="/webcam"
              className="px-6 py-3 rounded-full glass hover:bg-white/[0.08] text-white text-sm font-semibold transition-all hover:scale-105 active:scale-95 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
              Live Demo
            </Link>
          </motion.div>

          {/* Stats */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.65 }}
            className="flex gap-8 md:gap-16 justify-center flex-wrap"
          >
            {[
              { label: 'Best Accuracy', value: 89.6, suffix: '%', color: 'text-green-400' },
              { label: 'Models Trained', value: 6, suffix: '', color: 'text-blue-400' },
              { label: 'Experiments', value: 3, suffix: '', color: 'text-purple-400' },
              { label: 'Face Crops', value: 4072, suffix: '', color: 'text-orange-400' },
            ].map(s => (
              <div key={s.label} className="text-center">
                <p className={`text-3xl md:text-4xl font-bold ${s.color}`}>
                  <Counter to={s.value} suffix={s.suffix} decimals={s.value % 1 !== 0 ? 1 : 0} />
                </p>
                <p className="text-xs text-white/40 mt-1 uppercase tracking-wider">{s.label}</p>
              </div>
            ))}
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.5 }}
          className="absolute bottom-8 left-1/2 -translate-x-1/2"
        >
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity }}
            className="w-5 h-8 rounded-full border border-white/20 flex items-start justify-center pt-1.5"
          >
            <div className="w-1 h-2 rounded-full bg-white/40" />
          </motion.div>
        </motion.div>
      </section>

      {/* ── THE CHALLENGE ── */}
      <section id="challenge" className="py-28 px-5">
        <div className="max-w-6xl mx-auto">
          <Reveal>
            <SectionHeader
              eyebrow="The Challenge"
              title="3 Classes. Severe Imbalance."
              subtitle="The minority class has only 3% representation — handling this imbalance was a key challenge."
            />
          </Reveal>
          <Reveal delay={0.1}>
            <div className="grid md:grid-cols-3 gap-5">
              {[
                { icon: '🥸', label: 'With Mask', pct: 79.4, count: 3232, color: '#22c55e', bg: 'rgba(34,197,94,0.08)', border: 'rgba(34,197,94,0.15)' },
                { icon: '😑', label: 'Without Mask', pct: 17.6, count: 717, color: '#ef4444', bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.15)' },
                { icon: '⚠️', label: 'Mask Incorrect', pct: 3.0, count: 123, color: '#f59e0b', bg: 'rgba(245,158,11,0.08)', border: 'rgba(245,158,11,0.15)' },
              ].map((c, i) => (
                <motion.div
                  key={c.label}
                  initial={{ opacity: 0, y: 24 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: true, margin: '-60px' }}
                  transition={{ duration: 0.5, delay: i * 0.1 }}
                  whileHover={{ y: -6, transition: { duration: 0.2 } }}
                  className="rounded-2xl p-8 text-center"
                  style={{ background: c.bg, border: `1px solid ${c.border}` }}
                >
                  <span className="text-5xl block mb-4">{c.icon}</span>
                  <p className="text-4xl font-bold mb-1" style={{ color: c.color }}>
                    {c.pct}%
                  </p>
                  <h3 className="text-lg font-semibold text-white mb-2">{c.label}</h3>
                  <p className="text-sm text-white/40">{c.count.toLocaleString()} images</p>
                </motion.div>
              ))}
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── DATASET ── */}
      <section id="dataset" className="py-28 px-5 bg-white">
        <div className="max-w-6xl mx-auto">
          <Reveal>
            <SectionHeader
              eyebrow="Dataset"
              title="Prepared from Scratch"
              subtitle="Pascal VOC XML annotations parsed, faces cropped with 10px padding, split 70/15/15 stratified."
              light
            />
          </Reveal>
          <div className="grid md:grid-cols-2 gap-10 items-start">
            <Reveal>
              <img
                src="images/sample_images.png"
                alt="Sample images"
                className="rounded-2xl shadow-2xl w-full"
              />
            </Reveal>
            <Reveal delay={0.15}>
              <div className="space-y-3">
                {[
                  { k: 'Source', v: 'andrewmvd/face-mask-detection' },
                  { k: 'Total Face Crops', v: '4,072' },
                  { k: 'Source Images', v: '853' },
                  { k: 'Classes', v: '3' },
                  { k: 'Train / Val / Test', v: '70% / 15% / 15%' },
                  { k: 'Class Weighting', v: 'Inverse-frequency' },
                  { k: 'Annotation Format', v: 'Pascal VOC XML' },
                ].map(r => (
                  <div key={r.k} className="flex justify-between items-center py-3 border-b border-black/[0.05]">
                    <span className="text-sm text-gray-400">{r.k}</span>
                    <span className="text-sm font-semibold text-gray-800">{r.v}</span>
                  </div>
                ))}
              </div>
              <div className="mt-6">
                <img src="images/class_distribution.png" alt="Class distribution" className="rounded-xl shadow-lg w-full" />
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ── EXPERIMENTS ── */}
      <section id="experiments" className="py-28 px-5">
        <div className="max-w-6xl mx-auto">
          <Reveal>
            <SectionHeader eyebrow="Experiments" title="Three Approaches" subtitle="From hand-crafted CNNs to Vision Transformers." />
          </Reveal>

          {/* Exp 1 */}
          <Reveal>
            <div className="mb-10 glass rounded-2xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <span className="px-3 py-1 rounded-lg bg-blue-500/10 text-blue-400 text-xs font-bold uppercase tracking-wider">Experiment 1</span>
                <h3 className="text-2xl font-bold">Custom CNN Architectures</h3>
              </div>
              <p className="text-white/50 text-sm mb-8">Supervised learning with three hand-designed architectures of increasing complexity, each tested at two learning rates (6 total runs).</p>
              <div className="grid md:grid-cols-3 gap-4">
                {[
                  { name: 'Architecture A', color: '#3b82f6', specs: ['2 conv blocks', 'Filters: 32 → 64', 'No regularization', 'Baseline model'] },
                  { name: 'Architecture B', color: '#22c55e', specs: ['3 conv blocks', 'Filters: 32 → 128', 'BatchNorm + Dropout(0.4)', 'Moderate complexity'] },
                  { name: 'Architecture C', color: '#a855f7', specs: ['4 conv blocks', 'Filters: 32 → 256', 'BN + Dropout(0.5) + L2', 'Full regularization'] },
                ].map((a, i) => (
                  <motion.div
                    key={a.name}
                    whileHover={{ scale: 1.02 }}
                    className="rounded-xl p-5"
                    style={{ background: `${a.color}10`, border: `1px solid ${a.color}25` }}
                  >
                    <h4 className="font-bold mb-3" style={{ color: a.color }}>{a.name}</h4>
                    <ul className="space-y-1.5">
                      {a.specs.map(s => (
                        <li key={s} className="text-xs text-white/50 flex items-center gap-1.5">
                          <ChevronRight size={10} style={{ color: a.color }} />
                          {s}
                        </li>
                      ))}
                    </ul>
                  </motion.div>
                ))}
              </div>
            </div>
          </Reveal>

          {/* Exp 2 */}
          <Reveal delay={0.05}>
            <div className="mb-10 glass rounded-2xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <span className="px-3 py-1 rounded-lg bg-green-500/10 text-green-400 text-xs font-bold uppercase tracking-wider">Experiment 2</span>
                <h3 className="text-2xl font-bold">Transfer Learning</h3>
              </div>
              <p className="text-white/50 text-sm mb-8">MobileNetV2 and VGG16 with frozen feature extraction and progressive fine-tuning strategies.</p>
              <ExpTabs models={EXP2_MODELS} accentIndex={2} />
            </div>
          </Reveal>

          {/* Exp 3 */}
          <Reveal delay={0.1}>
            <div className="glass rounded-2xl p-8">
              <div className="flex items-center gap-3 mb-6">
                <span className="px-3 py-1 rounded-lg bg-purple-500/10 text-purple-400 text-xs font-bold uppercase tracking-wider">Experiment 3</span>
                <h3 className="text-2xl font-bold">State-of-the-Art Models</h3>
              </div>
              <p className="text-white/50 text-sm mb-8">EfficientNetB0 (scratch vs pretrained) and a custom Vision Transformer to explore modern architectures.</p>
              <ExpTabs models={EXP3_MODELS} accentIndex={3} />
            </div>
          </Reveal>
        </div>
      </section>

      {/* ── RESULTS ── */}
      <section id="results" className="py-28 px-5 bg-[#070707]">
        <div className="max-w-6xl mx-auto">
          <Reveal>
            <SectionHeader eyebrow="All Results" title="Model Comparison" subtitle="Complete performance breakdown sorted by accuracy." />
          </Reveal>

          {/* Table */}
          <Reveal>
            <div className="overflow-x-auto rounded-2xl border border-white/[0.07] mb-10">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-white/[0.06]">
                    {['#', 'Model', 'Experiment', 'Accuracy', 'F1 Score', 'Precision', 'Training'].map(h => (
                      <th key={h} className="text-left px-4 py-3 text-xs font-semibold text-white/30 uppercase tracking-wider bg-white/[0.02]">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {MODELS.map((m, i) => (
                    <motion.tr
                      key={m.key}
                      initial={{ opacity: 0, x: -16 }}
                      whileInView={{ opacity: 1, x: 0 }}
                      viewport={{ once: true }}
                      transition={{ duration: 0.4, delay: i * 0.06 }}
                      className="border-b border-white/[0.04] hover:bg-white/[0.025] transition-colors"
                    >
                      <td className="px-4 py-3.5 text-white/25">{i + 1}</td>
                      <td className="px-4 py-3.5">
                        <div className="flex items-center gap-2">
                          <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ background: m.color }} />
                          <span className="font-medium text-white">{m.name}</span>
                          {m.isWinner && <span className="text-yellow-400 text-xs">🏆</span>}
                        </div>
                      </td>
                      <td className="px-4 py-3.5">
                        <span className={`px-2 py-0.5 rounded-md text-xs font-medium ${
                          m.expNum === 2
                            ? 'bg-green-500/10 text-green-400'
                            : 'bg-purple-500/10 text-purple-400'
                        }`}>
                          {m.experiment}
                        </span>
                      </td>
                      <td className="px-4 py-3.5 font-bold" style={{ color: m.color }}>
                        {(m.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="px-4 py-3.5 text-white/70">{(m.f1 * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-white/70">{(m.precision * 100).toFixed(2)}%</td>
                      <td className="px-4 py-3.5 text-white/40">
                        {m.time >= 3600 ? (m.time / 3600).toFixed(1) + 'h' : Math.round(m.time) + 's'}
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>

          {/* Charts */}
          <div className="grid md:grid-cols-2 gap-6 mb-10">
            <Reveal>
              <div className="glass rounded-2xl p-6">
                <h3 className="text-base font-semibold mb-5 text-white">Accuracy Comparison</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <BarChart data={barData} margin={{ top: 5, right: 10, bottom: 30, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="name" tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }}
                      angle={-35} textAnchor="end" interval={0} />
                    <YAxis domain={[0, 100]} tick={{ fill: 'rgba(255,255,255,0.35)', fontSize: 11 }}
                      tickFormatter={v => v + '%'} />
                    <Tooltip content={<CustomTooltip />} />
                    <Bar dataKey="accuracy" radius={[6, 6, 0, 0]}>
                      {barData.map((d, i) => <Cell key={i} fill={d.color} fillOpacity={0.85} />)}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Reveal>

            <Reveal delay={0.1}>
              <div className="glass rounded-2xl p-6">
                <h3 className="text-base font-semibold mb-5 text-white">Performance Radar (Top 4)</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.08)" />
                    <PolarAngleAxis dataKey="metric" tick={{ fill: 'rgba(255,255,255,0.45)', fontSize: 12 }} />
                    <PolarRadiusAxis domain={[60, 100]} tick={false} axisLine={false} />
                    {MODELS.slice(0, 4).map(m => (
                      <Radar key={m.key} name={m.short} dataKey={m.short}
                        stroke={m.color} fill={m.color} fillOpacity={0.12} strokeWidth={2} />
                    ))}
                    <Tooltip content={<CustomTooltip />} />
                  </RadarChart>
                </ResponsiveContainer>
                <div className="flex gap-4 flex-wrap mt-2">
                  {MODELS.slice(0, 4).map(m => (
                    <div key={m.key} className="flex items-center gap-1.5 text-xs text-white/40">
                      <div className="w-2.5 h-2.5 rounded-full" style={{ background: m.color }} />
                      {m.short}
                    </div>
                  ))}
                </div>
              </div>
            </Reveal>
          </div>

          {/* Winner card */}
          <Reveal>
            <motion.div
              whileHover={{ scale: 1.005 }}
              className="relative overflow-hidden rounded-2xl border border-blue-500/20 p-8 md:p-12"
              style={{ background: 'linear-gradient(135deg, rgba(59,130,246,0.08), rgba(34,197,94,0.05))' }}
            >
              <div className="absolute top-0 right-0 w-64 h-64 rounded-full opacity-10"
                style={{ background: 'radial-gradient(circle, #3b82f6, transparent)', transform: 'translate(30%, -30%)' }} />
              <div className="relative">
                <Trophy className="text-yellow-400 mb-4" size={40} />
                <h3 className="text-2xl md:text-3xl font-bold mb-2">Best Model: MobileNetV2 Fine-Tuned</h3>
                <p className="text-white/50 mb-8 max-w-lg">Two-phase transfer learning achieves the highest accuracy with efficient training — the right balance of accuracy and speed.</p>
                <div className="flex gap-10 flex-wrap">
                  {[
                    { label: 'Accuracy', val: '89.58%', color: '#22c55e' },
                    { label: 'F1 Score', val: '89.62%', color: '#3b82f6' },
                    { label: 'Precision', val: '89.67%', color: '#a855f7' },
                    { label: 'Training', val: '412.7s', color: '#f59e0b' },
                  ].map(s => (
                    <div key={s.label}>
                      <p className="text-3xl font-bold" style={{ color: s.color }}>{s.val}</p>
                      <p className="text-xs text-white/40 mt-1 uppercase tracking-wider">{s.label}</p>
                    </div>
                  ))}
                </div>
              </div>
            </motion.div>
          </Reveal>
        </div>
      </section>

      {/* ── DEMO ── */}
      <section className="py-28 px-5">
        <div className="max-w-6xl mx-auto text-center">
          <Reveal>
            <SectionHeader eyebrow="Demo" title="See It In Action" subtitle="Prediction results on test images — then try it live with your own face." />
          </Reveal>
          <Reveal>
            <img src="images/prediction_grid.png" alt="Predictions" className="rounded-2xl w-full max-w-3xl mx-auto shadow-2xl mb-10" />
          </Reveal>
          <Reveal>
            <Link
              to="/webcam"
              className="inline-flex items-center gap-3 px-8 py-4 rounded-full bg-green-500 hover:bg-green-400 text-black font-bold text-lg transition-all hover:scale-105 active:scale-95 shadow-xl shadow-green-500/25"
            >
              <span className="w-3 h-3 rounded-full bg-black animate-ping absolute" />
              <Zap size={20} />
              Launch Live Webcam Detection
            </Link>
            <p className="text-white/30 text-sm mt-4">Runs entirely in your browser · 89.6% accurate · No data sent anywhere</p>
          </Reveal>
        </div>
      </section>

      {/* ── FOOTER ── */}
      <footer className="border-t border-white/[0.06] py-10 px-5 text-center">
        <p className="text-white/25 text-sm">
          Face Mask Detection · Deep Learning Project · Built with TensorFlow + React + ONNX Runtime Web
        </p>
      </footer>
    </>
  )
}

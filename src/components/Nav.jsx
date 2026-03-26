import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Video } from 'lucide-react'

const links = [
  { label: 'Challenge', href: '#challenge' },
  { label: 'Dataset', href: '#dataset' },
  { label: 'Experiments', href: '#experiments' },
  { label: 'Results', href: '#results' },
]

export default function Nav() {
  const [scrolled, setScrolled] = useState(false)
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 20)
    window.addEventListener('scroll', onScroll, { passive: true })
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const isDashboard = pathname === '/'

  return (
    <motion.nav
      initial={{ y: -60, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-300 ${
        scrolled
          ? 'bg-black/80 backdrop-blur-2xl border-b border-white/[0.06] shadow-lg shadow-black/20'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-5 h-14 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center gap-2.5 group">
          <span className="text-2xl">🎭</span>
          <span className="font-semibold text-[15px] tracking-tight text-white group-hover:text-blue-400 transition-colors">
            FaceMask<span className="text-blue-400"> AI</span>
          </span>
        </Link>

        {/* Desktop links */}
        {isDashboard && (
          <ul className="hidden md:flex gap-1 items-center">
            {links.map(l => (
              <li key={l.label}>
                <a
                  href={l.href}
                  className="px-3 py-1.5 text-[13px] text-white/60 hover:text-white rounded-lg hover:bg-white/[0.06] transition-all duration-200"
                >
                  {l.label}
                </a>
              </li>
            ))}
          </ul>
        )}

        {/* Right actions */}
        <div className="flex items-center gap-2">
          {pathname !== '/webcam' ? (
            <Link
              to="/webcam"
              className="hidden sm:flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-[13px] font-medium bg-green-500/10 text-green-400 border border-green-500/20 hover:bg-green-500/20 transition-all"
            >
              <Video size={13} />
              Live Demo
            </Link>
          ) : (
            <Link
              to="/"
              className="hidden sm:flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-[13px] font-medium bg-blue-500/10 text-blue-400 border border-blue-500/20 hover:bg-blue-500/20 transition-all"
            >
              ← Dashboard
            </Link>
          )}

          {/* Mobile toggle */}
          <button
            className="md:hidden p-2 rounded-lg text-white/60 hover:text-white hover:bg-white/[0.06] transition-all"
            onClick={() => setOpen(v => !v)}
          >
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>

      {/* Mobile menu */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.15 }}
            className="md:hidden bg-black/95 backdrop-blur-2xl border-b border-white/[0.06] px-5 py-4 flex flex-col gap-1"
          >
            {isDashboard && links.map(l => (
              <a
                key={l.label}
                href={l.href}
                onClick={() => setOpen(false)}
                className="px-3 py-2 text-[14px] text-white/70 hover:text-white rounded-lg hover:bg-white/[0.06] transition-all"
              >
                {l.label}
              </a>
            ))}
            <Link
              to={pathname === '/webcam' ? '/' : '/webcam'}
              onClick={() => setOpen(false)}
              className="mt-2 px-3 py-2 text-[14px] font-medium text-green-400 rounded-lg bg-green-500/10 text-center"
            >
              {pathname === '/webcam' ? '← Dashboard' : '🎥 Live Demo'}
            </Link>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  )
}

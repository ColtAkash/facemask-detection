import { HashRouter, Routes, Route } from 'react-router-dom'
import { lazy, Suspense, Component } from 'react'
import Nav from './components/Nav'
import Dashboard from './pages/Dashboard'

// Lazy-load Webcam so onnxruntime-web is NOT bundled into the main chunk
const WebcamPage = lazy(() => import('./pages/Webcam'))

class ErrorBoundary extends Component {
  state = { error: null }
  static getDerivedStateFromError(e) { return { error: e } }
  render() {
    if (this.state.error) {
      return (
        <div style={{
          minHeight: '100vh', display: 'flex', alignItems: 'center',
          justifyContent: 'center', flexDirection: 'column', gap: 16,
          background: '#050505', color: '#fff', fontFamily: 'sans-serif', padding: 24
        }}>
          <div style={{ fontSize: 48 }}>⚠️</div>
          <h2 style={{ margin: 0 }}>Something went wrong</h2>
          <pre style={{
            background: '#111', padding: 16, borderRadius: 8, fontSize: 12,
            color: '#ef4444', maxWidth: 600, overflow: 'auto', whiteSpace: 'pre-wrap'
          }}>
            {this.state.error.message}
          </pre>
          <button
            onClick={() => this.setState({ error: null })}
            style={{ padding: '10px 24px', background: '#3b82f6', color: '#fff', border: 'none', borderRadius: 8, cursor: 'pointer' }}
          >
            Try again
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

export default function App() {
  return (
    <ErrorBoundary>
      <HashRouter>
        <Nav />
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/webcam" element={
            <Suspense fallback={
              <div style={{ minHeight: '100vh', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#050505' }}>
                <div style={{ width: 40, height: 40, border: '3px solid rgba(255,255,255,0.1)', borderTopColor: '#3b82f6', borderRadius: '50%', animation: 'spin 0.8s linear infinite' }} />
              </div>
            }>
              <WebcamPage />
            </Suspense>
          } />
        </Routes>
      </HashRouter>
    </ErrorBoundary>
  )
}

import { HashRouter, Routes, Route } from 'react-router-dom'
import Nav from './components/Nav'
import Dashboard from './pages/Dashboard'
import WebcamPage from './pages/Webcam'

export default function App() {
  return (
    <HashRouter>
      <Nav />
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/webcam" element={<WebcamPage />} />
      </Routes>
    </HashRouter>
  )
}

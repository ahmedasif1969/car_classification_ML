import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Zap, Car, AlertCircle } from 'lucide-react'
import ImageUploader from './components/ImageUploader'
import ModelSelector from './components/ModelSelector'
import ResultsCard from './components/ResultsCard'
import LoadingSpinner from './components/LoadingSpinner'
import './App.css'

const API_BASE = '/api'

export default function App() {
  const [selectedImage, setSelectedImage] = useState(null)
  const [preview, setPreview] = useState(null)
  const [model, setModel] = useState('resnet18')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handleImageSelect = useCallback((file) => {
    if (!file) {
      setSelectedImage(null)
      setPreview(null)
      setResult(null)
      setError(null)
      return
    }
    setSelectedImage(file)
    setPreview(URL.createObjectURL(file))
    setResult(null)
    setError(null)
  }, [])

  const handleClassify = useCallback(async () => {
    if (!selectedImage) return

    setLoading(true)
    setError(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', selectedImage)

    try {
      const response = await fetch(`${API_BASE}/predict?model=${model}`, {
        method: 'POST',
        body: formData,
      })

      let data
      try {
        data = await response.json()
      } catch {
        throw new Error(
          response.ok
            ? 'Server returned an invalid response. Please try again.'
            : `Server error (${response.status}). Is the backend running on port 8000?`
        )
      }

      if (!response.ok) {
        throw new Error(data?.detail || `Server error: ${response.status}`)
      }

      setResult(data)
    } catch (err) {
      if (err.name === 'TypeError' && err.message.includes('fetch')) {
        setError('Cannot reach the API server. Make sure the FastAPI backend is running on port 8000.')
      } else {
        setError(err.message || 'An unexpected error occurred.')
      }
    } finally {
      setLoading(false)
    }
  }, [selectedImage, model])

  return (
    <>
      {/* Ambient background */}
      <div className="bg-mesh" />

      {/* SVG gradient defs for ResultsCard ring */}
      <svg width="0" height="0" style={{ position: 'absolute' }}>
        <defs>
          <linearGradient id="ringGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#63b3ed" />
            <stop offset="100%" stopColor="#9f7aea" />
          </linearGradient>
        </defs>
      </svg>

      <div className="app-layout">
        {/* ── Header ─────────────────────────────────────────────── */}
        <motion.header
          className="app-header"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <div className="header-logo">
            <div className="logo-icon">
              <Car size={22} />
            </div>
            <div>
              <h1 className="logo-title">CarClassifier</h1>
              <span className="logo-subtitle">AI-Powered Vehicle Recognition</span>
            </div>
          </div>
          <div className="header-badge">
            <span className="badge-dot" />
            API Online
          </div>
        </motion.header>

        {/* ── Main ───────────────────────────────────────────────── */}
        <main className="app-main">
          {/* Left panel */}
          <motion.div
            className="panel glass-card"
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
          >
            <div className="panel-header">
              <h2 className="panel-title">Upload Image</h2>
              <p className="panel-desc">Upload any car photo and our AI will identify the make and model.</p>
            </div>

            <ImageUploader
              onImageSelect={handleImageSelect}
              selectedImage={selectedImage}
              preview={preview}
            />

            <div className="divider" />

            <ModelSelector selected={model} onChange={setModel} />

            <motion.button
              id="classify-btn"
              className="btn-primary classify-btn"
              onClick={handleClassify}
              disabled={!selectedImage || loading}
              whileHover={selectedImage && !loading ? { scale: 1.02 } : {}}
              whileTap={selectedImage && !loading ? { scale: 0.98 } : {}}
            >
              <Zap size={18} />
              {loading ? 'Classifying…' : 'Classify Car'}
            </motion.button>
          </motion.div>

          {/* Right panel */}
          <motion.div
            className="panel results-panel"
            initial={{ opacity: 0, x: 30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          >
            <AnimatePresence mode="wait">
              {loading ? (
                <motion.div key="loading" className="glass-card results-placeholder">
                  <LoadingSpinner model={model} />
                </motion.div>
              ) : error ? (
                <motion.div
                  key="error"
                  className="glass-card error-card"
                  initial={{ opacity: 0, scale: 0.95 }}
                  animate={{ opacity: 1, scale: 1 }}
                  exit={{ opacity: 0, scale: 0.95 }}
                >
                  <AlertCircle size={36} className="error-icon" />
                  <p className="error-title">Classification Failed</p>
                  <p className="error-message">{error}</p>
                </motion.div>
              ) : result ? (
                <motion.div key="result">
                  <ResultsCard result={result} />
                </motion.div>
              ) : (
                <motion.div
                  key="empty"
                  className="glass-card results-placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                >
                  <div className="empty-state">
                    <motion.div
                      className="empty-car-icon"
                      animate={{ y: [0, -10, 0] }}
                      transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                    >
                      <Car size={52} strokeWidth={1} />
                    </motion.div>
                    <p className="empty-title">Ready to Classify</p>
                    <p className="empty-desc">Upload a car image and hit <strong>Classify Car</strong> to see results here.</p>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </main>

        {/* ── Footer ─────────────────────────────────────────────── */}
        <footer className="app-footer">
          <p>Built with YOLOv8 &amp; ResNet-18 · FastAPI backend</p>
        </footer>
      </div>
    </>
  )
}

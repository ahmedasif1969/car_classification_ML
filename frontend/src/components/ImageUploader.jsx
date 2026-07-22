import React, { useCallback, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, ImageIcon, X, CheckCircle2 } from 'lucide-react'
import styles from './ImageUploader.module.css'

export default function ImageUploader({ onImageSelect, selectedImage, preview }) {
  const [isDragging, setIsDragging] = useState(false)

  const handleFile = useCallback((file) => {
    if (!file) return
    if (!file.type.startsWith('image/')) {
      alert('Please upload an image file.')
      return
    }
    onImageSelect(file)
  }, [onImageSelect])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    handleFile(file)
  }, [handleFile])

  const handleDragOver = useCallback((e) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback(() => {
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback((e) => {
    handleFile(e.target.files[0])
  }, [handleFile])

  const clearImage = useCallback((e) => {
    e.stopPropagation()
    onImageSelect(null)
  }, [onImageSelect])

  return (
    <div className={styles.wrapper}>
      <motion.label
        htmlFor="car-image-input"
        className={`${styles.dropzone} ${isDragging ? styles.dragging : ''} ${preview ? styles.hasImage : ''}`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        animate={isDragging ? { scale: 1.02 } : { scale: 1 }}
        transition={{ type: 'spring', stiffness: 300, damping: 20 }}
      >
        <AnimatePresence mode="wait">
          {preview ? (
            <motion.div
              key="preview"
              className={styles.previewContainer}
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.9 }}
              transition={{ duration: 0.3 }}
            >
              <img src={preview} alt="Car preview" className={styles.previewImage} />
              <div className={styles.previewOverlay}>
                <CheckCircle2 size={20} color="#68d391" />
                <span className={styles.fileName}>{selectedImage?.name}</span>
              </div>
              <button
                className={styles.clearBtn}
                onClick={clearImage}
                title="Remove image"
                type="button"
              >
                <X size={16} />
              </button>
            </motion.div>
          ) : (
            <motion.div
              key="placeholder"
              className={styles.placeholder}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3 }}
            >
              <motion.div
                className={styles.iconWrap}
                animate={isDragging
                  ? { y: -8, scale: 1.1 }
                  : { y: [0, -6, 0] }
                }
                transition={isDragging
                  ? { type: 'spring' }
                  : { duration: 3, repeat: Infinity, ease: 'easeInOut' }
                }
              >
                {isDragging ? (
                  <ImageIcon size={40} strokeWidth={1.5} />
                ) : (
                  <Upload size={40} strokeWidth={1.5} />
                )}
              </motion.div>
              <p className={styles.title}>
                {isDragging ? 'Drop your image here' : 'Drag & drop a car image'}
              </p>
              <p className={styles.subtitle}>or <span className={styles.browse}>browse files</span></p>
              <p className={styles.hint}>Supports JPG, PNG, WEBP, BMP</p>
            </motion.div>
          )}
        </AnimatePresence>

        <input
          id="car-image-input"
          type="file"
          accept="image/*"
          className={styles.hiddenInput}
          onChange={handleInputChange}
        />
      </motion.label>
    </div>
  )
}

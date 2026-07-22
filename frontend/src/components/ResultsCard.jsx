import { motion } from 'framer-motion'
import { Trophy, BarChart3, Hash, Cpu } from 'lucide-react'
import styles from './ResultsCard.module.css'

function ConfidenceBar({ label, value, rank, isTop }) {
  const percent = (value * 100).toFixed(2)

  return (
    <motion.div
      className={`${styles.barRow} ${isTop ? styles.topRow : ''}`}
      initial={{ opacity: 0, x: -20 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ delay: rank * 0.06 }}
    >
      <div className={styles.barLabel}>
        {isTop && <Trophy size={12} className={styles.trophyIcon} />}
        <span className={styles.barName}>{label}</span>
        <span className={styles.barPercent}>{percent}%</span>
      </div>
      <div className={styles.barTrack}>
        <motion.div
          className={`${styles.barFill} ${isTop ? styles.barFillTop : ''}`}
          initial={{ width: 0 }}
          animate={{ width: `${percent}%` }}
          transition={{ duration: 0.8, delay: rank * 0.06 + 0.2, ease: 'easeOut' }}
        />
      </div>
    </motion.div>
  )
}

export default function ResultsCard({ result }) {
  if (!result) return null

  const topProbs = Object.entries(result.probabilities)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 8)

  const confidencePct = (result.confidence * 100).toFixed(1)

  return (
    <motion.div
      className={styles.card}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: 'easeOut' }}
    >
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerIcon}>
          <Trophy size={20} />
        </div>
        <div>
          <p className={styles.headerLabel}>Top Prediction</p>
          <h2 className={styles.className}>{result.class_name}</h2>
        </div>
        <div className={styles.confidencePill}>
          <motion.span
            className={styles.confidenceValue}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.3 }}
          >
            {confidencePct}%
          </motion.span>
          <span className={styles.confidenceLabel}>confidence</span>
        </div>
      </div>

      {/* Circular confidence ring */}
      <div className={styles.ringWrap}>
        <svg className={styles.ring} viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="52" className={styles.ringTrack} />
          <motion.circle
            cx="60" cy="60" r="52"
            className={styles.ringFill}
            strokeDasharray={`${2 * Math.PI * 52}`}
            initial={{ strokeDashoffset: 2 * Math.PI * 52 }}
            animate={{ strokeDashoffset: 2 * Math.PI * 52 * (1 - result.confidence) }}
            transition={{ duration: 1.2, ease: 'easeOut', delay: 0.2 }}
          />
        </svg>
        <div className={styles.ringLabel}>
          <span className={styles.ringValue}>{confidencePct}%</span>
          <span className={styles.ringText}>match</span>
        </div>
      </div>

      {/* Metadata */}
      <div className={styles.meta}>
        <div className={styles.metaItem}>
          <Cpu size={14} className={styles.metaIcon} />
          <span className={styles.metaLabel}>Model</span>
          <span className={styles.metaValue}>{result.model === 'resnet18' ? 'ResNet-18' : 'YOLOv8'}</span>
        </div>
        <div className={styles.metaDivider} />
        <div className={styles.metaItem}>
          <Hash size={14} className={styles.metaIcon} />
          <span className={styles.metaLabel}>Class Index</span>
          <span className={styles.metaValue}>{result.class_index}</span>
        </div>
      </div>

      {/* Probability Bars */}
      <div className={styles.barsSection}>
        <div className={styles.barsHeader}>
          <BarChart3 size={16} />
          <span>Top Probabilities</span>
        </div>
        <div className={styles.bars}>
          {topProbs.map(([label, value], i) => (
            <ConfidenceBar
              key={label}
              label={label}
              value={value}
              rank={i}
              isTop={i === 0}
            />
          ))}
        </div>
      </div>
    </motion.div>
  )
}

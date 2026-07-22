import { motion } from 'framer-motion'
import styles from './ModelSelector.module.css'

const MODELS = [
  {
    id: 'resnet18',
    label: 'ResNet-18',
    tag: 'CNN',
    description: 'Deep residual network, high accuracy on fine-grained car classification.',
    color: '#63b3ed',
  },
  {
    id: 'yolo',
    label: 'YOLOv8',
    tag: 'Real-time',
    description: 'Ultra-fast object detection model optimized for speed.',
    color: '#9f7aea',
  },
]

export default function ModelSelector({ selected, onChange }) {
  return (
    <div className={styles.wrapper}>
      <p className={styles.label}>Choose Model</p>
      <div className={styles.grid}>
        {MODELS.map((m) => {
          const isActive = selected === m.id
          return (
            <motion.button
              key={m.id}
              id={`model-btn-${m.id}`}
              className={`${styles.card} ${isActive ? styles.active : ''}`}
              onClick={() => onChange(m.id)}
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              style={{ '--accent': m.color }}
            >
              {isActive && (
                <motion.div
                  className={styles.activeBg}
                  layoutId="model-active-bg"
                  transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                />
              )}
              <div className={styles.cardInner}>
                <div className={styles.top}>
                  <span className={styles.modelName}>{m.label}</span>
                  <span className={styles.tag} style={{ color: m.color, borderColor: `${m.color}40`, background: `${m.color}12` }}>
                    {m.tag}
                  </span>
                </div>
                <p className={styles.description}>{m.description}</p>
              </div>
              {isActive && (
                <motion.div
                  className={styles.activeDot}
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  style={{ background: m.color }}
                />
              )}
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}

import { motion } from 'framer-motion'
import { Cpu } from 'lucide-react'
import styles from './LoadingSpinner.module.css'

export default function LoadingSpinner({ model }) {
  return (
    <motion.div
      className={styles.wrapper}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.9 }}
      transition={{ duration: 0.3 }}
    >
      <div className={styles.spinnerWrap}>
        {/* Outer ring */}
        <motion.div
          className={styles.ringOuter}
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: 'linear' }}
        />
        {/* Inner ring */}
        <motion.div
          className={styles.ringInner}
          animate={{ rotate: -360 }}
          transition={{ duration: 1.5, repeat: Infinity, ease: 'linear' }}
        />
        {/* Center icon */}
        <div className={styles.centerIcon}>
          <motion.div
            animate={{ scale: [1, 1.15, 1] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
          >
            <Cpu size={22} />
          </motion.div>
        </div>
      </div>

      <div className={styles.text}>
        <motion.p
          className={styles.title}
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 1.5, repeat: Infinity }}
        >
          Running inference…
        </motion.p>
        <p className={styles.subtitle}>
          Using <strong>{model === 'resnet18' ? 'ResNet-18' : 'YOLOv8'}</strong> model
        </p>
      </div>

      {/* Animated dots */}
      <div className={styles.dots}>
        {[0, 1, 2].map((i) => (
          <motion.div
            key={i}
            className={styles.dot}
            animate={{ y: [0, -8, 0], opacity: [0.3, 1, 0.3] }}
            transition={{ duration: 0.8, repeat: Infinity, delay: i * 0.2 }}
          />
        ))}
      </div>
    </motion.div>
  )
}

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { TrendingUp, Target, CheckCircle, BarChart } from 'lucide-react'
import axios from 'axios'

export default function MetricsDisplay() {
  const [metrics, setMetrics] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchMetrics()
  }, [])

  const fetchMetrics = async () => {
    try {
      const response = await axios.get('http://localhost:5001/api/metrics')
      if (response.data.status === 'success') {
        setMetrics(response.data.metrics)
      }
    } catch (error) {
      console.error('Error fetching metrics:', error)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="glass-card p-6 text-center">
        <div className="animate-pulse">Loading metrics...</div>
      </div>
    )
  }

  if (!metrics) {
    return (
      <div className="glass-card p-6 text-center text-white/70">
        <p>Metrics not available. Train the model first.</p>
      </div>
    )
  }

  const metricItems = [
    { label: 'Accuracy', value: metrics.accuracy, icon: Target, color: 'blue' },
    { label: 'Precision', value: metrics.precision, icon: CheckCircle, color: 'green' },
    { label: 'Recall', value: metrics.recall, icon: TrendingUp, color: 'purple' },
    { label: 'F1-Score', value: metrics.f1_score, icon: BarChart, color: 'pink' },
  ]

  const colorClasses = {
    blue: 'text-blue-400 bg-blue-500/20',
    green: 'text-green-400 bg-green-500/20',
    purple: 'text-purple-400 bg-purple-500/20',
    pink: 'text-pink-400 bg-pink-500/20',
  }

  return (
    <div className="glass-card p-6">
      <h3 className="text-xl font-semibold mb-4">Model Performance</h3>
      
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {metricItems.map((item, index) => {
          const Icon = item.icon
          const colorClass = colorClasses[item.color]
          
          return (
            <motion.div
              key={item.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
              className="glass-card p-4 text-center"
            >
              <div className={`inline-flex p-3 rounded-full ${colorClass} mb-3`}>
                <Icon className="w-6 h-6" />
              </div>
              <div className="text-2xl font-bold mb-1">
                {(item.value * 100).toFixed(1)}%
              </div>
              <div className="text-sm text-white/70">{item.label}</div>
            </motion.div>
          )
        })}
      </div>
      
      <div className="mt-4 text-xs text-white/50 text-center">
        Metrics based on test set evaluation
      </div>
    </div>
  )
}

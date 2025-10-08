import { motion } from 'framer-motion'
import { CheckCircle, AlertCircle, TrendingUp } from 'lucide-react'

export default function ResultCard({ result }) {
  const isHealthy = result.prediction === 'Healthy'
  const confidence = (result.confidence * 100).toFixed(1)
  
  const Icon = isHealthy ? CheckCircle : AlertCircle
  const bgColor = isHealthy ? 'from-green-600/20 to-emerald-600/20' : 'from-red-600/20 to-orange-600/20'
  const borderColor = isHealthy ? 'border-green-500/50' : 'border-red-500/50'
  const iconColor = isHealthy ? 'text-green-400' : 'text-red-400'
  
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
      className={`glass-card p-8 bg-gradient-to-br ${bgColor} border-2 ${borderColor}`}
    >
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-4">
          <Icon className={`w-16 h-16 ${iconColor}`} />
          <div>
            <h2 className="text-3xl font-bold">{result.prediction}</h2>
            <p className="text-white/70">Prediction Result</p>
          </div>
        </div>
        
        <div className="text-right">
          <div className="text-4xl font-bold">{confidence}%</div>
          <p className="text-white/70 text-sm">Confidence</p>
        </div>
      </div>
      
      {/* Confidence Bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm mb-2">
          <span>Confidence Level</span>
          <span>{confidence}%</span>
        </div>
        <div className="w-full bg-white/10 rounded-full h-3 overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${confidence}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className={`h-full rounded-full ${
              isHealthy 
                ? 'bg-gradient-to-r from-green-500 to-emerald-500' 
                : 'bg-gradient-to-r from-red-500 to-orange-500'
            }`}
          />
        </div>
      </div>
      
      {/* Probabilities */}
      <div className="grid grid-cols-2 gap-4">
        <div className="glass-card p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-white/70">Healthy</span>
            <TrendingUp className="w-4 h-4 text-green-400" />
          </div>
          <div className="text-2xl font-bold text-green-400">
            {(result.probabilities.Healthy * 100).toFixed(1)}%
          </div>
        </div>
        
        <div className="glass-card p-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-white/70">Depressed</span>
            <TrendingUp className="w-4 h-4 text-red-400" />
          </div>
          <div className="text-2xl font-bold text-red-400">
            {(result.probabilities.Depressed * 100).toFixed(1)}%
          </div>
        </div>
      </div>
      
      {/* Interpretation */}
      <div className="mt-6 p-4 bg-white/5 rounded-lg">
        <h4 className="font-semibold mb-2">Interpretation:</h4>
        <p className="text-sm text-white/80 leading-relaxed">
          {isHealthy ? (
            "The analysis suggests typical speech patterns without strong indicators of depression. However, mental health is complex and multifaceted. If you have concerns, please consult a healthcare professional."
          ) : (
            "The analysis suggests potential indicators of depression in the speech patterns. This may include reduced vocal energy, monotone speech, or slower speech rate. Please note: this is a screening tool only. Consult a qualified healthcare professional for proper diagnosis and treatment."
          )}
        </p>
      </div>
    </motion.div>
  )
}

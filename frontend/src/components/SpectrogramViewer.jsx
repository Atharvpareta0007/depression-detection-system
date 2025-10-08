import { motion } from 'framer-motion'
import { Activity } from 'lucide-react'

export default function SpectrogramViewer({ spectrogram }) {
  if (!spectrogram) return null
  
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="glass-card p-6"
    >
      <div className="flex items-center space-x-3 mb-4">
        <Activity className="w-6 h-6 text-blue-400" />
        <h3 className="text-xl font-semibold">Audio Features</h3>
      </div>
      
      <div className="bg-white/5 rounded-lg p-4 overflow-hidden">
        <img 
          src={spectrogram} 
          alt="MFCC Spectrogram" 
          className="w-full h-auto rounded-lg"
        />
      </div>
      
      <p className="mt-4 text-sm text-white/70">
        <strong>MFCC (Mel-Frequency Cepstral Coefficients)</strong> features represent 
        the spectral characteristics of the speech signal. These features capture prosody, 
        pitch, energy patterns, and other acoustic properties used for depression detection.
      </p>
    </motion.div>
  )
}

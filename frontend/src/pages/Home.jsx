import { useState } from 'react'
import { motion } from 'framer-motion'
import { Loader, Sparkles, AlertTriangle, Brain, Zap, Shield, Target, Upload, Mic } from 'lucide-react'
import axios from 'axios'
import FileUploader from '../components/FileUploader'
import AudioRecorder from '../components/AudioRecorder'
import ResultCard from '../components/ResultCard'
import SpectrogramViewer from '../components/SpectrogramViewer'
import MetricsDisplay from '../components/MetricsDisplay'
import { API_ENDPOINTS } from '../config'

export default function Home() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleFileSelect = (file) => {
    setSelectedFile(file)
    setResult(null)
    setError(null)
  }

  const handleAudioReady = (audioBlob, filename) => {
    if (audioBlob && filename) {
      // Convert blob to File object
      const file = new File([audioBlob], filename, { type: audioBlob.type });
      setSelectedFile(file);
      setResult(null);
      setError(null);
    } else {
      // Clear if no audio
      setSelectedFile(null);
      setResult(null);
      setError(null);
    }
  }

  const handleClearFile = () => {
    setSelectedFile(null)
    setResult(null)
    setError(null)
  }

  const handlePredict = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)

    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post(API_ENDPOINTS.predict, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })

      if (response.data.status === 'success') {
        setResult(response.data)
      } else {
        setError(response.data.error || 'An error occurred during prediction')
      }
    } catch (err) {
      console.error('Error:', err)
      setError(
        err.response?.data?.error || 
        'Failed to connect to the server. Make sure the backend is running.'
      )
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-7xl mx-auto">
      {/* Hero Section */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1, ease: "easeOut" }}
        className="text-center mb-16 relative"
      >
        {/* Floating background elements */}
        <div className="absolute inset-0 -z-10">
          <motion.div
            animate={{ rotate: 360, scale: [1, 1.1, 1] }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="absolute top-10 left-1/4 w-32 h-32 bg-gradient-to-r from-purple-500/10 to-cyan-500/10 rounded-full blur-xl"
          />
          <motion.div
            animate={{ rotate: -360, scale: [1, 1.2, 1] }}
            transition={{ duration: 25, repeat: Infinity, ease: "linear" }}
            className="absolute top-20 right-1/4 w-24 h-24 bg-gradient-to-r from-pink-500/10 to-purple-500/10 rounded-full blur-xl"
          />
        </div>

        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.8 }}
          className="flex flex-col items-center justify-center mb-8"
        >
          <div className="relative mb-6">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
              className="absolute inset-0 bg-gradient-to-r from-purple-500 to-cyan-500 rounded-full blur-lg opacity-30"
            />
            <Brain className="relative w-24 h-24 text-transparent bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text drop-shadow-2xl" />
          </div>
          <div className="text-center">
            <h1 className="text-6xl font-bold bg-gradient-to-r from-purple-400 via-pink-400 to-cyan-400 bg-clip-text text-transparent mb-4 neon-text">
              Enhanced Depression Detection
            </h1>
            <p className="text-2xl text-gray-300 font-light">
              Cross-Modal Knowledge Distillation System
            </p>
          </div>
        </motion.div>
        
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.8 }}
          className="flex items-center justify-center space-x-12 text-sm"
        >
          <motion.div 
            whileHover={{ scale: 1.05, y: -2 }}
            className="flex items-center space-x-3 glass-card px-6 py-3 rounded-xl"
          >
            <Target className="w-5 h-5 text-green-400 pulse-glow" />
            <span className="text-white font-medium">75% Accuracy</span>
          </motion.div>
          <motion.div 
            whileHover={{ scale: 1.05, y: -2 }}
            className="flex items-center space-x-3 glass-card px-6 py-3 rounded-xl"
          >
            <Zap className="w-5 h-5 text-yellow-400 pulse-glow" />
            <span className="text-white font-medium">Real-time Analysis</span>
          </motion.div>
          <motion.div 
            whileHover={{ scale: 1.05, y: -2 }}
            className="flex items-center space-x-3 glass-card px-6 py-3 rounded-xl"
          >
            <Shield className="w-5 h-5 text-blue-400 pulse-glow" />
            <span className="text-white font-medium">Neural Network</span>
          </motion.div>
        </motion.div>
      </motion.div>

      {/* Upload and Recording Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-12">
        <motion.div
          initial={{ opacity: 0, x: -30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.8 }}
          className="glass-card-modern p-8 rounded-2xl card-3d hover:shadow-2xl hover:shadow-cyan-500/20"
        >
          <div className="flex items-center mb-6">
            <div className="relative">
              <Upload className="w-8 h-8 text-cyan-400 mr-4" />
              <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-md"></div>
            </div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
              Upload Audio
            </h2>
          </div>
          <FileUploader
            onFileSelect={handleFileSelect}
            selectedFile={selectedFile}
            onClear={handleClearFile}
          />
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 30 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4, duration: 0.8 }}
          className="glass-card-modern p-8 rounded-2xl card-3d hover:shadow-2xl hover:shadow-pink-500/20 float-3d"
        >
          <div className="flex items-center mb-6">
            <div className="relative">
              <Mic className="w-8 h-8 text-pink-400 mr-4" />
              <div className="absolute inset-0 bg-pink-400/20 rounded-full blur-md pulse-glow"></div>
            </div>
            <h2 className="text-3xl font-bold bg-gradient-to-r from-pink-400 to-purple-400 bg-clip-text text-transparent">
              Live Recording
            </h2>
          </div>
          <AudioRecorder
            onAudioReady={handleAudioReady}
            disabled={loading}
          />
        </motion.div>
      </div>

      {/* Predict Button */}
      {selectedFile && !result && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="flex justify-center mb-12"
        >
          <motion.button
            whileHover={{ scale: 1.05, rotateX: 5 }}
            whileTap={{ scale: 0.95 }}
            onClick={handlePredict}
            disabled={loading}
            className="btn-futuristic flex items-center space-x-3 px-12 py-4 text-xl disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden"
          >
            {loading ? (
              <>
                <Loader className="w-7 h-7 animate-spin spin-glow" />
                <span>Analyzing Neural Patterns...</span>
              </>
            ) : (
              <>
                <Sparkles className="w-7 h-7 neon-glow" />
                <span>Analyze Audio</span>
              </>
            )}
          </motion.button>
        </motion.div>
      )}

      {/* Error Message */}
      {error && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card-modern bg-red-500/10 border border-red-500/30 rounded-2xl p-6 text-red-200 card-3d mb-8"
        >
          <div className="flex items-start space-x-3">
            <AlertTriangle className="w-6 h-6 text-red-400 flex-shrink-0 mt-1 neon-glow" />
            <div>
              <h3 className="font-bold text-lg mb-2 text-red-400">System Error</h3>
              <p className="text-red-300">{error}</p>
            </div>
          </div>
        </motion.div>
      )}

      {/* Results Section */}
      {result && (
        <motion.div
          initial={{ opacity: 0, y: 30, scale: 0.95 }}
          animate={{ opacity: 1, y: 0, scale: 1 }}
          transition={{ duration: 0.8, ease: "easeOut" }}
          className="space-y-8 mb-12"
        >
          <ResultCard result={result} />
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3, duration: 0.6 }}
            className="glass-card-modern p-6 rounded-2xl card-3d"
          >
            <SpectrogramViewer spectrogram={result.spectrogram} />
          </motion.div>
        </motion.div>
      )}

      {/* Metrics Section */}
      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6, duration: 0.8 }}
        className="mb-12"
      >
        <MetricsDisplay />
      </motion.div>

      {/* Disclaimer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8, duration: 0.6 }}
        className="glass-card-modern p-8 border border-yellow-500/30 bg-yellow-500/5 rounded-2xl card-3d"
      >
        <div className="flex items-start space-x-4">
          <AlertTriangle className="w-8 h-8 text-yellow-400 flex-shrink-0 pulse-glow" />
          <div>
            <h3 className="font-bold text-2xl text-yellow-400 mb-3">Important Disclaimer</h3>
            <p className="text-gray-300 text-base leading-relaxed mb-3">
              <strong className="text-white">This is a research demo. Not a medical device. Not for clinical diagnosis.</strong>
            </p>
            <p className="text-gray-300 text-base leading-relaxed">
              This system is for research and demonstration purposes only. It is NOT a medical device and should NOT be used for:
              clinical diagnosis or medical decision-making, treatment recommendations, standalone medical device use, or legal/insurance purposes.
              Always consult qualified healthcare professionals for mental health concerns.
            </p>
          </div>
        </div>
      </motion.div>
    </div>
  )
}

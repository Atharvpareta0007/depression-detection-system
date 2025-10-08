import { useState, useRef } from 'react'
import { motion } from 'framer-motion'
import { Upload, X, FileAudio } from 'lucide-react'

export default function FileUploader({ onFileSelect, selectedFile, onClear }) {
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef(null)

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)

    const files = e.dataTransfer.files
    if (files && files[0]) {
      handleFileChange(files[0])
    }
  }

  const handleFileChange = (file) => {
    // Check file type
    const validTypes = ['audio/wav', 'audio/mpeg', 'audio/flac', 'audio/x-wav']
    const validExtensions = ['.wav', '.mp3', '.flac']
    
    const isValidType = validTypes.includes(file.type) || 
                       validExtensions.some(ext => file.name.toLowerCase().endsWith(ext))
    
    if (!isValidType) {
      alert('Please upload a valid audio file (.wav, .mp3, or .flac)')
      return
    }

    // Check file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
      alert('File size must be less than 16MB')
      return
    }

    onFileSelect(file)
  }

  const handleInputChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileChange(e.target.files[0])
    }
  }

  const handleClick = () => {
    fileInputRef.current?.click()
  }

  return (
    <div className="w-full">
      {!selectedFile ? (
        <motion.div
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={handleClick}
          className={`glass-card p-12 text-center cursor-pointer transition-all duration-300 ${
            isDragging ? 'border-blue-500 bg-blue-500/20' : 'border-white/20 hover:border-blue-400/50'
          }`}
        >
          <Upload className={`w-16 h-16 mx-auto mb-4 ${isDragging ? 'text-blue-400 animate-bounce' : 'text-white/60'}`} />
          <h3 className="text-xl font-semibold mb-2">Upload Audio File</h3>
          <p className="text-white/70 mb-4">
            Drag and drop your audio file here, or click to browse
          </p>
          <p className="text-sm text-white/50">
            Supported formats: WAV, MP3, FLAC (max 16MB)
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept=".wav,.mp3,.flac,audio/*"
            onChange={handleInputChange}
            className="hidden"
          />
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <FileAudio className="w-12 h-12 text-blue-400" />
              <div>
                <h3 className="font-semibold text-lg">{selectedFile.name}</h3>
                <p className="text-sm text-white/60">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <button
              onClick={onClear}
              className="p-2 hover:bg-red-500/20 rounded-lg transition-colors"
            >
              <X className="w-6 h-6 text-red-400" />
            </button>
          </div>
        </motion.div>
      )}
    </div>
  )
}

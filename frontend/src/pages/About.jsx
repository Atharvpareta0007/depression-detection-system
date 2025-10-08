import { motion } from 'framer-motion'
import { Brain, BookOpen, Cpu, Database, GitBranch, Award, ExternalLink } from 'lucide-react'

export default function About() {
  return (
    <div className="max-w-5xl mx-auto">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center mb-12"
      >
        <Brain className="w-20 h-20 mx-auto mb-4 text-blue-400" />
        <h1 className="text-4xl font-bold mb-4">About the Project</h1>
        <p className="text-xl text-white/70">
          Understanding cross-modal knowledge distillation for depression detection
        </p>
      </motion.div>

      {/* Research Background */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center space-x-3 mb-4">
          <BookOpen className="w-6 h-6 text-purple-400" />
          <h2 className="text-2xl font-bold">Research Background</h2>
        </div>
        
        <p className="text-white/80 leading-relaxed mb-4">
          This system implements the methodology from the research paper 
          <strong className="text-blue-400"> "Cross-modal Knowledge Distillation for Enhanced Depression Detection"</strong> 
          published in <em>Complex & Intelligent Systems</em> (2025).
        </p>
        
        <p className="text-white/80 leading-relaxed">
          The research introduces a novel approach that leverages knowledge from multimodal 
          (EEG + Speech) models to enhance the performance of speech-only depression detection systems, 
          making them more practical for real-world deployment.
        </p>
      </motion.div>

      {/* Methodology */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center space-x-3 mb-6">
          <GitBranch className="w-6 h-6 text-green-400" />
          <h2 className="text-2xl font-bold">Methodology</h2>
        </div>
        
        <div className="space-y-6">
          {/* TeacherNet */}
          <div className="border-l-4 border-blue-500 pl-4">
            <h3 className="text-xl font-semibold mb-2 text-blue-400">1. TeacherNet (Multimodal)</h3>
            <p className="text-white/80 leading-relaxed">
              Trained on both EEG and speech data using a dual-stream architecture with TCNNB 
              (Transformer + CNN) blocks. Features are fused using IFFB (Improved Feature Fusion Block) 
              for cross-modal integration. Achieves high accuracy by leveraging complementary information 
              from both modalities.
            </p>
          </div>

          {/* StudentNet */}
          <div className="border-l-4 border-purple-500 pl-4">
            <h3 className="text-xl font-semibold mb-2 text-purple-400">2. StudentNet (Speech-only)</h3>
            <p className="text-white/80 leading-relaxed">
              A unimodal model that processes only speech data. Uses the same TCNNB architecture 
              but operates on a single modality. Practical for real-world deployment as it doesn't 
              require expensive EEG equipment.
            </p>
          </div>

          {/* Knowledge Distillation */}
          <div className="border-l-4 border-pink-500 pl-4">
            <h3 className="text-xl font-semibold mb-2 text-pink-400">3. Knowledge Distillation</h3>
            <p className="text-white/80 leading-relaxed mb-3">
              The StudentNet learns from the TeacherNet through three complementary loss components:
            </p>
            <ul className="list-disc list-inside text-white/70 space-y-2 ml-4">
              <li><strong className="text-white">L<sub>CE</sub>:</strong> Cross-entropy loss with true labels</li>
              <li><strong className="text-white">L<sub>KD</sub>:</strong> KL divergence for soft logits transfer (temperature-scaled)</li>
              <li><strong className="text-white">L<sub>SK</sub>:</strong> MSE loss for intermediate feature alignment</li>
            </ul>
            <p className="text-white/80 mt-3">
              Combined Loss: <code className="bg-white/10 px-2 py-1 rounded">L = (1−α)L<sub>CE</sub> + αL<sub>KD</sub> + L<sub>SK</sub></code>
            </p>
          </div>
        </div>
      </motion.div>

      {/* Architecture */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center space-x-3 mb-4">
          <Cpu className="w-6 h-6 text-yellow-400" />
          <h2 className="text-2xl font-bold">Model Architecture</h2>
        </div>
        
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-semibold mb-2 text-blue-400">TCNNB Blocks</h3>
            <p className="text-white/80">
              Each TCNNB (Transformer + CNN Block) combines the strengths of both architectures:
            </p>
            <ul className="list-disc list-inside text-white/70 ml-4 mt-2">
              <li>Transformer encoder captures long-range temporal dependencies</li>
              <li>CNN layers extract local patterns and features</li>
              <li>3 stacked blocks for hierarchical feature learning</li>
            </ul>
          </div>
          
          <div>
            <h3 className="text-lg font-semibold mb-2 text-purple-400">Feature Extraction</h3>
            <p className="text-white/80">
              Speech features are extracted using MFCC (Mel-Frequency Cepstral Coefficients) 
              with 40 coefficients plus delta and delta-delta features, capturing prosody, 
              pitch, energy patterns, and other acoustic properties.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Dataset */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center space-x-3 mb-4">
          <Database className="w-6 h-6 text-green-400" />
          <h2 className="text-2xl font-bold">Dataset</h2>
        </div>
        
        <div>
          <h3 className="text-xl font-semibold mb-3 text-green-400">
            MODMA (Multi-modal Open Dataset for Mental Disorder Analysis)
          </h3>
          <ul className="space-y-2 text-white/80">
            <li className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <span><strong>EEG Data:</strong> 128-channel resting-state recordings</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <span><strong>Speech Data:</strong> Audio recordings from structured interviews</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <span><strong>Subjects:</strong> Healthy controls (HC) and major depressive disorder (MDD) patients</span>
            </li>
            <li className="flex items-start">
              <span className="text-blue-400 mr-2">•</span>
              <span><strong>Task:</strong> Binary classification (Healthy / Depressed)</span>
            </li>
          </ul>
        </div>
      </motion.div>

      {/* Applications */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-card p-8 mb-8"
      >
        <div className="flex items-center space-x-3 mb-4">
          <Award className="w-6 h-6 text-pink-400" />
          <h2 className="text-2xl font-bold">Applications</h2>
        </div>
        
        <div className="grid md:grid-cols-2 gap-4">
          <div className="border border-white/20 rounded-lg p-4">
            <h3 className="font-semibold mb-2 text-blue-400">Telemedicine</h3>
            <p className="text-sm text-white/70">Remote depression screening via voice calls</p>
          </div>
          <div className="border border-white/20 rounded-lg p-4">
            <h3 className="font-semibold mb-2 text-purple-400">Mobile Health</h3>
            <p className="text-sm text-white/70">Smartphone-based continuous monitoring</p>
          </div>
          <div className="border border-white/20 rounded-lg p-4">
            <h3 className="font-semibold mb-2 text-green-400">Early Intervention</h3>
            <p className="text-sm text-white/70">Regular mental health assessment</p>
          </div>
          <div className="border border-white/20 rounded-lg p-4">
            <h3 className="font-semibold mb-2 text-pink-400">Research</h3>
            <p className="text-sm text-white/70">Understanding depression biomarkers</p>
          </div>
        </div>
      </motion.div>

      {/* Citation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-card p-8 bg-gradient-to-br from-blue-600/20 to-purple-600/20 border-2 border-blue-500/30"
      >
        <div className="flex items-center space-x-3 mb-4">
          <ExternalLink className="w-6 h-6 text-blue-400" />
          <h2 className="text-2xl font-bold">Citation</h2>
        </div>
        
        <div className="bg-black/30 p-4 rounded-lg font-mono text-sm text-white/80">
          <p className="mb-2"><strong>Paper:</strong> Cross-modal Knowledge Distillation for Enhanced Depression Detection</p>
          <p className="mb-2"><strong>Journal:</strong> Complex & Intelligent Systems</p>
          <p className="mb-2"><strong>Year:</strong> 2025</p>
          <p><strong>Publisher:</strong> Springer</p>
        </div>
      </motion.div>

      {/* Ethical Considerations */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="mt-8 glass-card p-8 border-2 border-yellow-500/30 bg-yellow-500/10"
      >
        <h2 className="text-2xl font-bold mb-4 text-yellow-400">Ethical Considerations</h2>
        <ul className="space-y-2 text-white/80">
          <li className="flex items-start">
            <span className="text-yellow-400 mr-2">⚠️</span>
            <span>This is a screening tool, not a diagnostic instrument</span>
          </li>
          <li className="flex items-start">
            <span className="text-yellow-400 mr-2">⚠️</span>
            <span>Should be used in conjunction with professional assessment</span>
          </li>
          <li className="flex items-start">
            <span className="text-yellow-400 mr-2">⚠️</span>
            <span>Privacy and data security must be ensured</span>
          </li>
          <li className="flex items-start">
            <span className="text-yellow-400 mr-2">⚠️</span>
            <span>Bias and fairness in model predictions should be monitored</span>
          </li>
        </ul>
      </motion.div>
    </div>
  )
}

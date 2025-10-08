import React, { useState, useRef, useEffect } from 'react';
import { Mic, MicOff, Square, Play, Pause, Download, Trash2 } from 'lucide-react';

const AudioRecorder = ({ onAudioReady, disabled = false }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [recordingTime, setRecordingTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState(null);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const timerRef = useRef(null);
  const audioRef = useRef(null);
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      // Cleanup on unmount
      if (timerRef.current) {
        clearInterval(timerRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  const startRecording = async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: 16000
        } 
      });
      
      streamRef.current = stream;
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus'
      });
      
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { 
          type: 'audio/webm;codecs=opus' 
        });
        setAudioBlob(audioBlob);
        
        const url = URL.createObjectURL(audioBlob);
        setAudioUrl(url);
        
        // Convert to WAV and notify parent
        convertToWav(audioBlob);
        
        // Stop all tracks
        if (streamRef.current) {
          streamRef.current.getTracks().forEach(track => track.stop());
        }
      };

      mediaRecorder.start(100); // Collect data every 100ms
      setIsRecording(true);
      setIsPaused(false);
      setRecordingTime(0);

      // Start timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);

    } catch (err) {
      console.error('Error accessing microphone:', err);
      setError('Unable to access microphone. Please check permissions.');
    }
  };

  const pauseRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.pause();
      setIsPaused(true);
      clearInterval(timerRef.current);
    }
  };

  const resumeRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'paused') {
      mediaRecorderRef.current.resume();
      setIsPaused(false);
      
      // Resume timer
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => prev + 1);
      }, 1000);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      setIsPaused(false);
      clearInterval(timerRef.current);
    }
  };

  const convertToWav = async (webmBlob) => {
    try {
      // Create audio context
      const audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Convert blob to array buffer
      const arrayBuffer = await webmBlob.arrayBuffer();
      
      // Decode audio data
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
      
      // Convert to WAV
      const wavBlob = audioBufferToWav(audioBuffer);
      
      // Notify parent component
      if (onAudioReady) {
        onAudioReady(wavBlob, 'recorded_audio.wav');
      }
      
    } catch (err) {
      console.error('Error converting audio:', err);
      // Fallback: use original blob
      if (onAudioReady) {
        onAudioReady(webmBlob, 'recorded_audio.webm');
      }
    }
  };

  const audioBufferToWav = (audioBuffer) => {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;

    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;

    const buffer = new ArrayBuffer(44 + audioBuffer.length * bytesPerSample);
    const view = new DataView(buffer);

    // WAV header
    const writeString = (offset, string) => {
      for (let i = 0; i < string.length; i++) {
        view.setUint8(offset + i, string.charCodeAt(i));
      }
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + audioBuffer.length * bytesPerSample, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, audioBuffer.length * bytesPerSample, true);

    // Convert audio data
    const channelData = audioBuffer.getChannelData(0);
    let offset = 44;
    for (let i = 0; i < channelData.length; i++) {
      const sample = Math.max(-1, Math.min(1, channelData[i]));
      view.setInt16(offset, sample * 0x7FFF, true);
      offset += 2;
    }

    return new Blob([buffer], { type: 'audio/wav' });
  };

  const playRecording = () => {
    if (audioRef.current) {
      if (isPlaying) {
        audioRef.current.pause();
        setIsPlaying(false);
      } else {
        audioRef.current.play();
        setIsPlaying(true);
      }
    }
  };

  const deleteRecording = () => {
    setAudioBlob(null);
    setAudioUrl(null);
    setIsPlaying(false);
    setRecordingTime(0);
    if (onAudioReady) {
      onAudioReady(null, null);
    }
  };

  const downloadRecording = () => {
    if (audioBlob) {
      const url = URL.createObjectURL(audioBlob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'recorded_audio.wav';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="space-y-6">
      {error && (
        <div className="glass-card bg-red-500/10 border border-red-500/30 text-red-200 px-4 py-3 rounded-xl">
          <p className="font-medium">{error}</p>
        </div>
      )}

      <div className="space-y-4">
        {/* Recording Controls */}
        <div className="flex items-center gap-3 flex-wrap">
          {!isRecording && !audioBlob && (
            <button
              onClick={startRecording}
              disabled={disabled}
              className="flex items-center gap-2 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 disabled:from-gray-500 disabled:to-gray-600 text-white px-6 py-3 rounded-xl transition-all duration-300 font-semibold shadow-lg hover:shadow-red-500/50 hover:scale-105 disabled:hover:scale-100 disabled:cursor-not-allowed"
            >
              <Mic className="w-5 h-5" />
              Start Recording
            </button>
          )}

          {isRecording && (
            <>
              {!isPaused ? (
                <button
                  onClick={pauseRecording}
                  className="flex items-center gap-2 bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 text-white px-5 py-3 rounded-xl transition-all duration-300 font-semibold shadow-lg hover:shadow-yellow-500/50 hover:scale-105"
                >
                  <Pause className="w-5 h-5" />
                  Pause
                </button>
              ) : (
                <button
                  onClick={resumeRecording}
                  className="flex items-center gap-2 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-5 py-3 rounded-xl transition-all duration-300 font-semibold shadow-lg hover:shadow-green-500/50 hover:scale-105"
                >
                  <Play className="w-5 h-5" />
                  Resume
                </button>
              )}

              <button
                onClick={stopRecording}
                className="flex items-center gap-2 bg-gradient-to-r from-gray-600 to-gray-700 hover:from-gray-700 hover:to-gray-800 text-white px-5 py-3 rounded-xl transition-all duration-300 font-semibold shadow-lg hover:shadow-gray-500/50 hover:scale-105"
              >
                <Square className="w-5 h-5" />
                Stop
              </button>
            </>
          )}

          {/* Recording Timer */}
          {(isRecording || recordingTime > 0) && (
            <div className="flex items-center gap-3 glass-card px-5 py-3 rounded-xl">
              <div className={`w-4 h-4 rounded-full ${isRecording && !isPaused ? 'bg-red-500 animate-pulse shadow-lg shadow-red-500/50' : 'bg-gray-400'}`}></div>
              <span className="text-xl font-mono font-bold text-white">{formatTime(recordingTime)}</span>
            </div>
          )}
        </div>

        {/* Playback Controls */}
        {audioUrl && (
          <div className="space-y-4 glass-card p-5 rounded-xl border border-white/10">
            <audio
              ref={audioRef}
              src={audioUrl}
              onEnded={() => setIsPlaying(false)}
              className="w-full rounded-lg"
              controls
            />

            <div className="flex items-center gap-3 flex-wrap">
              <button
                onClick={playRecording}
                className="flex items-center gap-2 bg-gradient-to-r from-blue-500 to-cyan-500 hover:from-blue-600 hover:to-cyan-600 text-white px-4 py-2 rounded-xl transition-all duration-300 font-medium shadow-lg hover:shadow-blue-500/50 hover:scale-105"
              >
                {isPlaying ? <Pause className="w-5 h-5" /> : <Play className="w-5 h-5" />}
                {isPlaying ? 'Pause' : 'Play'}
              </button>

              <button
                onClick={downloadRecording}
                className="flex items-center gap-2 bg-gradient-to-r from-green-500 to-emerald-500 hover:from-green-600 hover:to-emerald-600 text-white px-4 py-2 rounded-xl transition-all duration-300 font-medium shadow-lg hover:shadow-green-500/50 hover:scale-105"
              >
                <Download className="w-5 h-5" />
                Download
              </button>

              <button
                onClick={deleteRecording}
                className="flex items-center gap-2 bg-gradient-to-r from-red-500 to-pink-500 hover:from-red-600 hover:to-pink-600 text-white px-4 py-2 rounded-xl transition-all duration-300 font-medium shadow-lg hover:shadow-red-500/50 hover:scale-105"
              >
                <Trash2 className="w-5 h-5" />
                Delete
              </button>
            </div>

            <div className="glass-card bg-green-500/10 border border-green-500/30 px-4 py-2 rounded-lg">
              <p className="text-green-300 font-medium flex items-center gap-2">
                <span className="text-green-400 text-xl">✓</span>
                Recording ready for analysis
              </p>
            </div>
          </div>
        )}

        {/* Instructions */}
        <div className="glass-card p-5 rounded-xl border border-purple-500/20">
          <h4 className="text-lg font-bold text-white mb-3 flex items-center gap-2">
            <span className="text-purple-400">💡</span>
            Quick Guide
          </h4>
          <ul className="space-y-2 text-gray-300">
            <li className="flex items-start gap-2">
              <span className="text-cyan-400 font-bold">→</span>
              <span>Click <strong className="text-white">"Start Recording"</strong> to begin</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400 font-bold">→</span>
              <span>Speak clearly for <strong className="text-white">5-30 seconds</strong></span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400 font-bold">→</span>
              <span>Use <strong className="text-white">"Pause/Resume"</strong> if needed</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400 font-bold">→</span>
              <span>Click <strong className="text-white">"Stop"</strong> when finished</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="text-cyan-400 font-bold">→</span>
              <span>Recording automatically processed for analysis</span>
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default AudioRecorder;

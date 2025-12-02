import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Camera, Square, Play, Mic, AlertCircle, CheckCircle } from 'lucide-react';
import { useVoice } from '../context/VoiceNavigationContext';
import toast from 'react-hot-toast';
import './VisionPage.css';

// REPLACE WITH YOUR CURRENT NGROK URL
const API_BASE_URL = 'https://f1c662134b53.ngrok-free.app'; 

const VisionPage = () => {
  const { speak } = useVoice();
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentDescription, setCurrentDescription] = useState('Camera not active');
  const [detections, setDetections] = useState([]);
  const [annotatedImage, setAnnotatedImage] = useState(null);
  const [error, setError] = useState(null);
  const [lang, setLang] = useState('en');

  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const intervalRef = useRef(null);
  const canvasRef = useRef(null);
  const latestTTS = useRef(null); 

  // --- LISTEN FOR VOICE COMMANDS ---
  useEffect(() => {
    const handleVoiceStart = () => {
        // This function runs when you say "Start camera" or "Describe surroundings"
        if (!streamRef.current) {
            startCamera();
        }
    };

    const handleVoiceStop = () => stopCamera();
    
    const handleVoiceCapture = () => {
        // If camera is off, turn it on. If on, it captures automatically in loop.
        if (!streamRef.current) {
            startCamera();
        }
    };

    window.addEventListener('voice-start-camera', handleVoiceStart);
    window.addEventListener('voice-stop-camera', handleVoiceStop);
    window.addEventListener('voice-capture', handleVoiceCapture);

    // Cleanup
    return () => {
      // If we are leaving the page and camera is on, say "Camera stopped"
      if (streamRef.current) {
         speak("Camera stopped.");
         stopCamera(true); // true = silent stop (don't speak twice)
      } else {
         stopCamera(true);
      }
      window.removeEventListener('voice-start-camera', handleVoiceStart);
      window.removeEventListener('voice-stop-camera', handleVoiceStop);
      window.removeEventListener('voice-capture', handleVoiceCapture);
    };
  }, []);

  const startCamera = async () => {
    if (streamRef.current) return; // Already running

    try {
      setError(null);
      speak("Starting live camera."); // Audio feedback
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
      });
      
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
      
      setIsStreaming(true);
      startPeriodicAnalysis(); // Starts the AI Loop immediately
    } catch (err) {
      console.error(err);
      setError('Could not access camera.');
      toast.error("Camera access denied");
      speak("I cannot access the camera.");
    }
  };

  const stopCamera = (silent = false) => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
    }
    if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;

    setIsStreaming(false);
    setIsAnalyzing(false);
    setCurrentDescription('Camera not active');
    setDetections([]);
    setAnnotatedImage(null);
    latestTTS.current = null;

    if (!silent) speak("Camera stopped.");
  };

  const captureFrame = () => {
    if (!videoRef.current || !canvasRef.current) return null;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    canvas.width = 480; 
    canvas.height = 360;
    ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    return new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', 0.7));
  };

  const analyzeFrame = async () => {
    if (isAnalyzing) return;
    try {
      setIsAnalyzing(true);
      const blob = await captureFrame();
      if (!blob) return;

      const formData = new FormData();
      formData.append('frame', blob, 'frame.jpg');
      formData.append('lang', lang);

      const response = await fetch(`${API_BASE_URL}/analyze_frame`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) throw new Error("Network error");
      
      const data = await response.json();
      if (data) {
        const desc = data.caption_translated || data.caption || 'No description available';
        setCurrentDescription(desc);
        setDetections(data.detections || []);
        if (data.annotated_image) setAnnotatedImage(`data:image/png;base64,${data.annotated_image}`);
        latestTTS.current = data.tts_audio || null;
      }
    } catch (err) {
      console.error(err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const startPeriodicAnalysis = () => {
    analyzeFrame(); // Run once immediately
    intervalRef.current = setInterval(analyzeFrame, 2500); // Then every 2.5s
  };

  const playAudio = () => {
    if (latestTTS.current) {
      new Audio(`data:audio/mp3;base64,${latestTTS.current}`).play();
    } else {
      speak(currentDescription); 
    }
  };

  return (
    <div className="webcam-component">
      <div className="webcam-header">
        <h2>Real-time Camera Analysis</h2>
        <p>Point your camera at objects to get live AI captions</p>
      </div>

      <div className="webcam-content">
        <div className="video-section">
          <div className="video-container">
            <video ref={videoRef} autoPlay playsInline muted className="webcam-video" />
            <canvas ref={canvasRef} style={{ display: 'none' }} />
            {!isStreaming && (
              <div className="video-placeholder">
                <Camera className="placeholder-icon" size={48} />
                <p>Camera not active</p>
              </div>
            )}
          </div>

          <div className="video-controls">
            <div className="control-block">
              {!isStreaming ? (
                <motion.button 
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={startCamera} 
                  className="control-button start-button"
                >
                  <Play size={20} /> Start Camera
                </motion.button>
              ) : (
                <motion.button 
                  whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
                  onClick={() => stopCamera(false)} 
                  className="control-button stop-button"
                >
                  <Square size={20} /> Stop Camera
                </motion.button>
              )}
            </div>
            <motion.button 
              whileHover={{ scale: 1.05 }} whileTap={{ scale: 0.95 }}
              onClick={playAudio} 
              disabled={!latestTTS.current && !currentDescription}
              className="control-button audio-button"
            >
              <Mic size={20} /> Play Audio
            </motion.button>
          </div>

          <div className="voice-select-wrapper control-block-select">
            <select value={lang} onChange={(e) => setLang(e.target.value)} className="voice-select">
              <option value="en">English</option>
              <option value="hi">Hindi</option>
              <option value="mr">Marathi</option>
            </select>
          </div>
        </div>

        <div className="analysis-section">
          {annotatedImage && <div className="annotated-image-container"><img src={annotatedImage} alt="AI Vision" className="annotated-image" /></div>}
          {detections.length > 0 && (
            <div className="detections-card">
              <div className="detections-list">
                {detections.map((det, idx) => (
                  <div key={idx} className="detection-item">
                    <CheckCircle className="detection-icon" size={16} />
                    <span>{det.class}</span>
                    <span className="detection-confidence">{(det.confidence * 100).toFixed(0)}%</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          <div className="description-card">
             <p className="description-text">{currentDescription}</p>
          </div>
          {error && <div className="error-card"><AlertCircle size={20} /><p>{error}</p></div>}
        </div>
      </div>
    </div>
  );
};

export default VisionPage;
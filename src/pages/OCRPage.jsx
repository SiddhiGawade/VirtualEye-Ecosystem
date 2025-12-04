import React, { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import { BookOpen, Camera, X } from 'lucide-react';
import { useVoice } from '../context/VoiceNavigationContext';
import toast from 'react-hot-toast';
import './VisionPage.css';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const OCRPage = () => {
  const { speak } = useVoice();
  
  const [isStreaming, setIsStreaming] = useState(false);
  const [ocrText, setOcrText] = useState('');
  const [ocrInProgress, setOcrInProgress] = useState(false);
  const [currentFrame, setCurrentFrame] = useState(null);
  
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  // Start camera
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        streamRef.current = stream;
        setIsStreaming(true);
        toast.success('Camera started');
      }
    } catch (err) {
      console.error('Error accessing camera:', err);
      toast.error('Could not access camera. Check permissions.');
    }
  };

  // Stop camera
  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop());
      streamRef.current = null;
      setIsStreaming(false);
      toast.success('Camera stopped');
    }
  };

  // Capture frame from video
  const captureFrame = () => {
    if (videoRef.current && canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      canvasRef.current.width = videoRef.current.videoWidth;
      canvasRef.current.height = videoRef.current.videoHeight;
      ctx.drawImage(videoRef.current, 0, 0);
      return canvasRef.current.toDataURL('image/jpeg', 0.9);
    }
    return null;
  };

  // Handle OCR
  const handleOCR = async () => {
    if (!isStreaming) {
      toast.error('Camera not active. Start camera first.');
      return;
    }

    setOcrInProgress(true);
    
    try {
      // Capture frame
      const frameData = captureFrame();
      if (!frameData) {
        throw new Error('Failed to capture frame');
      }

      // Convert to blob
      const response = await fetch(frameData);
      const blob = await response.blob();

      // Send to backend
      const formData = new FormData();
      formData.append('frame', blob);

      console.log('[OCR] Sending frame to backend...');
      const res = await fetch(`${API_BASE_URL}/ocr`, {
        method: 'POST',
        body: formData,
        timeout: 30000
      });

      console.log(`[OCR] Backend response: ${res.status}`);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ error: res.statusText }));
        throw new Error(`Backend error: ${errorData.error || res.statusText}`);
      }

      const data = await res.json();
      
      if (data.text && data.text.trim()) {
        setOcrText(data.text);
        speak(`Found text: ${data.text}`);
        toast.success(`Extracted ${data.line_count} lines of text`);
      } else {
        setOcrText('No text detected in the current frame.');
        speak('No text found in the image.');
        toast.info('No text detected');
      }
    } catch (err) {
      console.error('[OCR] Error:', err);
      const errorMsg = err.message || 'Unknown OCR error';
      toast.error(`OCR failed: ${errorMsg}`);
      setOcrText('');
    } finally {
      setOcrInProgress(false);
    }
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);

  return (
    <div className="vision-container">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="vision-content"
      >
        {/* Header */}
        <div className="vision-header">
          <div className="flex items-center gap-3">
            <div className="icon-badge">
              <BookOpen size={32} />
            </div>
            <div>
              <h1 className="vision-title">Read Text (OCR)</h1>
              <p className="vision-subtitle">Extract text from camera view using EasyOCR</p>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Camera Section */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.1 }}
            className="video-container"
          >
            <video
              ref={videoRef}
              autoPlay
              playsInline
              className="video-stream"
              style={{ display: isStreaming ? 'block' : 'none' }}
            />
            {!isStreaming && (
              <div className="video-placeholder">
                <Camera size={64} color="#cbd5e1" />
                <p>Camera not active</p>
              </div>
            )}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
          </motion.div>

          {/* OCR Section */}
          <motion.div
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5, delay: 0.2 }}
            className="ocr-section"
          >
            <h3>Text Extraction Controls</h3>
            
            <div className="ocr-controls">
              <button
                onClick={startCamera}
                disabled={isStreaming}
                className="control-button"
                style={{
                  background: isStreaming 
                    ? '#d1d5db' 
                    : 'linear-gradient(135deg, #3b82f6, #1d4ed8)',
                  color: isStreaming ? '#6b7280' : 'white'
                }}
              >
                <Camera size={18} />
                {isStreaming ? 'Camera Active' : 'Start Camera'}
              </button>

              <button
                onClick={stopCamera}
                disabled={!isStreaming}
                className="control-button"
                style={{
                  background: !isStreaming 
                    ? '#d1d5db' 
                    : 'linear-gradient(135deg, #ef4444, #dc2626)',
                  color: !isStreaming ? '#6b7280' : 'white'
                }}
              >
                <X size={18} />
                Stop Camera
              </button>
            </div>

            <button
              onClick={handleOCR}
              disabled={!isStreaming || ocrInProgress}
              className="ocr-action-button"
              style={{
                opacity: (!isStreaming || ocrInProgress) ? 0.5 : 1,
                cursor: (!isStreaming || ocrInProgress) ? 'not-allowed' : 'pointer'
              }}
            >
              <BookOpen size={20} />
              {ocrInProgress ? 'Extracting Text...' : 'Extract Text from Camera'}
            </button>

            {/* Results */}
            {ocrText ? (
              <div className="ocr-result-box">
                <h4 className="font-semibold text-slate-700 mb-2">Extracted Text:</h4>
                <div className="ocr-text-display">
                  {ocrText}
                </div>
              </div>
            ) : (
              <div className="ocr-hint-box">
                Click "Extract Text from Camera" to read text visible in your camera view.
              </div>
            )}
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default OCRPage;

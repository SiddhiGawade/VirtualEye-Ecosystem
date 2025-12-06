import React, { createContext, useContext, useState, useEffect, useRef } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import toast from 'react-hot-toast';

const VoiceContext = createContext();

const ROUTE_NAMES = {
  '/dashboard/vision': 'Opening Live Vision',
  '/dashboard/ocr': 'Opening Smart Reader',
  '/dashboard/chat': 'Opening Voice Chat',
  '/dashboard/settings': 'Opening Settings',
  '/dashboard/demopurpose': 'Opening Demo Section',
  '/dashboard': 'Entering Dashboard',
  '/': 'Exiting Application'
};

export const VoiceProvider = ({ children }) => {
  const [isListening, setIsListening] = useState(false);
  const [lastCommand, setLastCommand] = useState('');
  
  const navigate = useNavigate();
  const location = useLocation();
  
  const recognitionRef = useRef(null);
  const isListeningRef = useRef(isListening); 
  const processingRef = useRef(false);
  const [voice, setVoice] = useState(null);

  // --- VOICE LOADING ---
  useEffect(() => {
    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices();
      if (voices.length > 0) {
        const preferredVoice = voices.find(v => v.name.includes('Google US English')) || 
                               voices.find(v => v.name.includes('Zira')) || 
                               voices.find(v => v.lang.startsWith('en'));
        if (preferredVoice) setVoice(preferredVoice);
      }
    };
    window.speechSynthesis.onvoiceschanged = loadVoices;
    loadVoices();
  }, []);

  // --- AUTO-ANNOUNCE PAGE CHANGES ---
  useEffect(() => {
    const message = ROUTE_NAMES[location.pathname];
    if (message) setTimeout(() => speak(message), 200);
  }, [location.pathname]);

  // --- SPEECH RECOGNITION ---
  useEffect(() => {
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      const recognition = new SpeechRecognition();
      
      recognition.continuous = true;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => { setIsListening(true); isListeningRef.current = true; };
      recognition.onend = () => { if (isListeningRef.current) try { recognition.start(); } catch (e) {} else setIsListening(false); };
      recognition.onerror = () => { isListeningRef.current = false; setIsListening(false); };
      
      recognition.onresult = (event) => {
        const transcript = event.results[event.resultIndex][0].transcript.toLowerCase().trim();
        console.log("Heard:", transcript);
        setLastCommand(transcript);
        
        if (!processingRef.current) {
          processingRef.current = true;
          processCommand(transcript);
          setTimeout(() => { processingRef.current = false; }, 1500);
        }
      };
      
      recognitionRef.current = recognition;
    }
    return () => { if (recognitionRef.current) recognitionRef.current.stop(); };
  }, []);

  // --- COMMAND LOGIC ---
  const processCommand = (cmd) => {
    const matches = (keywords) => keywords.some(k => cmd.includes(k));

    // 1. CAMERA ACTIVATION COMMANDS
    if (matches(['start camera', 'turn on camera', 'describe my surrounding', 'describe surrounding', 'what is around me'])) {
        if (location.pathname.includes('vision')) {
            window.dispatchEvent(new CustomEvent('voice-start-camera'));
        } else {
            speak("Opening Vision to describe surroundings.");
            navigate('/dashboard/vision');
            setTimeout(() => window.dispatchEvent(new CustomEvent('voice-start-camera')), 1500);
        }
        return; 
    }

    // 2. STOP COMMANDS
    if (matches(['stop camera', 'turn off camera', 'close camera'])) {
        if (location.pathname.includes('vision')) {
            window.dispatchEvent(new CustomEvent('voice-stop-camera'));
        }
        return;
    }

    // 3. CALIBRATION COMMANDS
    if (matches(['calibrate', 'calibration', 'set distance', 'start calibrate'])) {
        if (location.pathname.includes('vision')) {
            window.dispatchEvent(new CustomEvent('voice-start-calibration'));
            speak("Calibration mode activated. Please enter the distance.");
        } else {
            speak("Opening Vision for calibration.");
            navigate('/dashboard/vision');
            setTimeout(() => window.dispatchEvent(new CustomEvent('voice-start-calibration')), 1500);
        }
        return;
    }

    // 4. Q&A COMMANDS
    if (matches(['ask', 'question', 'what is', 'tell me about', 'describe', 'ask ai'])) {
        if (location.pathname.includes('vision')) {
            window.dispatchEvent(new CustomEvent('voice-start-qa'));
            speak("Q&A mode activated. Please ask your question.");
        } else {
            speak("Opening Vision for question answering.");
            navigate('/dashboard/vision');
            setTimeout(() => window.dispatchEvent(new CustomEvent('voice-start-qa')), 1500);
        }
        return;
    }

    // 5. CAPTURE/SINGLE DESCRIBE
    else if (matches(['capture', 'take photo', 'snap', 'analyze']) && location.pathname.includes('vision')) {
        window.dispatchEvent(new CustomEvent('voice-capture'));
        return;
    }

    // 5b. OCR COMMANDS
    if (matches(['read text', 'extract text', 'read', 'scan text', 'ocr']) && location.pathname.includes('ocr')) {
        window.dispatchEvent(new CustomEvent('voice-read-text'));
        return;
    }
    
    if (matches(['auto read', 'continuous read', 'start auto read', 'auto scan']) && location.pathname.includes('ocr')) {
        window.dispatchEvent(new CustomEvent('voice-auto-read'));
        return;
    }

    // 6. NAVIGATION
    if (matches(['vision', 'live see', 'camera'])) navigate('/dashboard/vision');
    else if (matches(['read', 'ocr', 'text', 'smart reader'])) navigate('/dashboard/ocr');
    else if (matches(['chat', 'bot', 'voice chat'])) navigate('/dashboard/chat');
    else if (matches(['setting', 'settings'])) navigate('/dashboard/settings');
    else if (matches(['demo'])) navigate('/dashboard/demopurpose');
    else if (matches(['exit', 'logout', 'goodbye'])) navigate('/');
    else if (matches(['enter', 'dashboard', 'app', 'home'])) navigate('/dashboard');
    
    // 7. SOS
    else if (matches(['sos', 'help', 'emergency', 'urgent'])) {
      speak("Triggering Emergency Alert!");
      toast.error("SOS ALERT TRIGGERED");
    }
  };

  const toggleListening = () => {
    if (isListeningRef.current) {
      isListeningRef.current = false;
      recognitionRef.current.stop();
      speak("Voice paused.");
      setIsListening(false);
    } else {
      isListeningRef.current = true;
      try { recognitionRef.current.start(); speak("I am listening."); } catch(e) {}
      setIsListening(true);
    }
  };

  const speak = (text) => {
    window.speechSynthesis.cancel();
    const voices = window.speechSynthesis.getVoices();
    // Retry if voices not loaded
    if (voices.length === 0) { setTimeout(() => speak(text), 100); return; }

    let selectedVoice = voice || voices.find(v => v.name.includes('Google US English')) || voices.find(v => v.lang.startsWith('en'));
    
    const utterance = new SpeechSynthesisUtterance(text);
    if (selectedVoice) utterance.voice = selectedVoice;
    window.speechSynthesis.speak(utterance);
  };

  return (
    <VoiceContext.Provider value={{ isListening, toggleListening, lastCommand, speak }}>
      {children}
    </VoiceContext.Provider>
  );
};

export const useVoice = () => useContext(VoiceContext);
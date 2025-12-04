"""
Virtual Eye 3.0 Backend - YOLOv8 + ByteTrack + BLIP-2 + Voice
Flask server for real-time object detection, tracking, distance estimation, and scene captioning
"""

import cv2
import time
import json
import torch
import pyttsx3
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from io import BytesIO
import os

# ============== CONFIG ==============
MODEL_WEIGHTS = "yolov8n.pt"
CONF_THRESH = 0.35
LEFT_FRAC = 0.33
RIGHT_FRAC = 0.66
K_CALIB_FILE = "calib_K.json"
K_DEFAULT = None
TRACK_SPEAK_COOLDOWN = 1.5
CAPTION_INTERVAL = 5.0

# ============== FLASK APP ==============
app = Flask(__name__)
CORS(app)

# ============== GLOBAL STATE ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {device}")

# Load YOLO model
print("[INIT] Loading YOLO model...")
model = YOLO(MODEL_WEIGHTS)

# Load BLIP-2 model with error handling
print("[INIT] Loading BLIP-2 model...")
try:
    # Try using the smaller, more stable BLIP-2 variant
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", trust_remote_code=True)
    blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", trust_remote_code=True, torch_dtype=torch.float32).to(device)
except Exception as e:
    print(f"[WARNING] BLIP-2 OPT-2.7B failed to load: {e}")
    print("[INIT] Attempting to load BLIP-2 Flan-T5 variant...")
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True)
        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True, torch_dtype=torch.float32).to(device)
        print("[INIT] Successfully loaded BLIP-2 Flan-T5 variant")
    except Exception as e2:
        print(f"[ERROR] BLIP-2 models failed: {e2}")
        print("[INIT] BLIP-2 disabled - Q&A and captioning will be unavailable")
        blip_model = None
        processor = None

# Initialize TTS
tts = pyttsx3.init()
tts.setProperty("rate", 160)

# Track memory for persistent tracking
track_memory = {}
last_global_speak = 0.0
last_caption_time = 0.0

# ============== UTILITY FUNCTIONS ==============
def load_calib_K():
    """Load calibrated K value from file"""
    try:
        with open(K_CALIB_FILE, "r") as f:
            data = json.load(f)
            return data.get("K", None)
    except:
        return None

def save_calib_K(K):
    """Save calibrated K value to file"""
    with open(K_CALIB_FILE, "w") as f:
        json.dump({"K": K}, f)

def choose_side(cx, w):
    """Determine if object is on left/right/center"""
    if cx < w * LEFT_FRAC:
        return "left"
    if cx > w * RIGHT_FRAC:
        return "right"
    return "center"

def format_distance(d):
    """Format distance for human-readable output"""
    if d is None:
        return "?"
    if d >= 10:
        return f"{d:.0f} m"
    elif d >= 1:
        return f"{d:.1f} m"
    else:
        return f"{int(d*100)} cm"

def speak(text):
    """Text-to-speech"""
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

def blip_caption(frame):
    """Generate scene caption using BLIP-2"""
    try:
        if blip_model is None or processor is None:
            return "Scene analysis unavailable - BLIP-2 model not loaded"
        
        # Convert OpenCV BGR -> RGB
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img_rgb, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_length=50)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"BLIP caption error: {e}")
        return "Unable to generate caption due to model error"

def blip_qa(frame, question):
    """Answer question about frame using BLIP-2"""
    try:
        if blip_model is None or processor is None:
            return "Q&A unavailable - BLIP-2 model not loaded"
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inputs = processor(images=img_rgb, text=question, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_length=100)
        answer = processor.decode(output_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        print(f"BLIP QA error: {e}")
        return f"Unable to answer due to model error: {str(e)[:50]}"

def encode_image_base64(image_array):
    """Encode OpenCV image to base64"""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')

# ============== ENDPOINTS ==============

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok", "device": device})

@app.route('/analyze_frame', methods=['POST'])
def analyze_frame():
    """
    Analyze a frame for objects, distances, and generate captions
    
    Request:
    - frame: image file
    - lang: language code (en/hi/mr) - for future localization
    
    Response:
    - detections: list of {class, confidence, bbox, distance, side}
    - caption: BLIP-2 scene caption
    - has_objects: boolean
    - K_value: current calibration K
    """
    global track_memory, last_global_speak, last_caption_time, K_DEFAULT
    
    try:
        if 'frame' not in request.files:
            return jsonify({"error": "No frame provided"}), 400
        
        file = request.files['frame']
        lang = request.form.get('lang', 'en')
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        h, w = frame.shape[:2]
        
        # Run YOLO detection with ByteTrack
        results = model.track(frame, persist=True, conf=CONF_THRESH)[0]
        
        detections = []
        now = time.time()
        
        # Process detections
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = model.names[cls]
                track_id = int(box.id[0]) if box.id is not None else -1
                
                cx = (x1 + x2) // 2
                bh = (y2 - y1)
                dist_m = K_DEFAULT / (bh + 1e-6) if K_DEFAULT else None
                side = choose_side(cx, w)
                
                detections.append({
                    "class": name,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "track_id": track_id,
                    "distance": dist_m,
                    "distance_str": format_distance(dist_m),
                    "side": side,
                    "cx": cx,
                    "cy": (y1 + y2) // 2
                })
                
                # Draw on frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} {format_distance(dist_m)}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw center lines
        cv2.line(frame, (int(w * LEFT_FRAC), 0), (int(w * LEFT_FRAC), h), (255, 255, 0), 1)
        cv2.line(frame, (int(w * RIGHT_FRAC), 0), (int(w * RIGHT_FRAC), h), (255, 255, 0), 1)
        
        # Generate caption every CAPTION_INTERVAL
        caption = ""
        if now - last_caption_time > CAPTION_INTERVAL:
            caption = blip_caption(frame)
            last_caption_time = now
        
        # Encode annotated frame
        annotated_b64 = encode_image_base64(frame)
        
        return jsonify({
            "detections": detections,
            "caption": caption,
            "has_objects": len(detections) > 0,
            "K_value": K_DEFAULT,
            "annotated_image": annotated_b64
        })
    
    except Exception as e:
        print(f"Error in analyze_frame: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/question', methods=['POST'])
def ask_question():
    """
    Answer a question about the current frame using BLIP-2
    
    Request:
    - frame: image file
    - question: question string
    
    Response:
    - answer: BLIP-2 generated answer
    """
    try:
        if 'frame' not in request.files or 'question' not in request.form:
            return jsonify({"error": "Missing frame or question"}), 400
        
        file = request.files['frame']
        question = request.form['question']
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        answer = blip_qa(frame, question)
        
        return jsonify({
            "question": question,
            "answer": answer
        })
    
    except Exception as e:
        print(f"Error in question endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/calibrate', methods=['POST'])
def calibrate():
    """
    Calibrate K value using known distance and detected bbox height
    
    Request:
    - frame: image file
    - distance_m: known distance in meters
    
    Response:
    - K: calibrated K value
    - success: boolean
    """
    global K_DEFAULT
    
    try:
        if 'frame' not in request.files or 'distance_m' not in request.form:
            return jsonify({"error": "Missing frame or distance_m"}), 400
        
        file = request.files['frame']
        dist_m = float(request.form['distance_m'])
        
        # Read image
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400
        
        # Detect objects
        res = model(frame)[0]
        best_h = 0
        
        for r in res.boxes:
            x1, y1, x2, y2 = map(int, r.xyxy[0].tolist())
            h = y2 - y1
            if h > best_h:
                best_h = h
        
        if best_h <= 0:
            return jsonify({"error": "No object detected for calibration"}), 400
        
        K = dist_m * best_h
        save_calib_K(K)
        K_DEFAULT = K
        
        return jsonify({
            "success": True,
            "K": K,
            "bbox_height": best_h,
            "distance_m": dist_m
        })
    
    except Exception as e:
        print(f"Error in calibrate: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_calib_K', methods=['GET'])
def get_calib_K():
    """Get current calibration K value"""
    global K_DEFAULT
    return jsonify({
        "K": K_DEFAULT,
        "is_calibrated": K_DEFAULT is not None
    })

@app.route('/reset_calib', methods=['POST'])
def reset_calib():
    """Reset calibration"""
    global K_DEFAULT
    K_DEFAULT = None
    try:
        os.remove(K_CALIB_FILE)
    except:
        pass
    return jsonify({"success": True, "K": None})

# ============== INITIALIZATION ==============
if __name__ == '__main__':
    # Load saved calibration
    K_DEFAULT = load_calib_K()
    if K_DEFAULT:
        print(f"[INIT] Loaded calibrated K = {K_DEFAULT:.3f}")
    else:
        print("[INIT] No calibrated K found. Calibration required for distance estimation.")
    
    print("\n[SERVER] Starting Flask server...")
    print("[SERVER] Running on http://localhost:5000")
    
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

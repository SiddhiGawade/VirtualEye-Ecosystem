"""
Virtual Eye 3.0 Backend - YOLOv8 + ByteTrack + BLIP-2 + Voice
Flask server for real-time object detection, tracking, distance estimation, and scene captioning
"""

import cv2
import time
import json
import torch
import pyttsx3
import easyocr
import base64
import numpy as np
import requests
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
CAPTION_INTERVAL = 5.0
OCR_LANG_DEFAULT = ["en"]
OCR_PRELOAD_LANG_SETS = [
    OCR_LANG_DEFAULT,
    ["mr"],
    ["en", "mr"],
    ["hi"],
    ["en", "hi"],
]

# ============== FLASK APP ==============
app = Flask(__name__)
# Allow all origins for ngrok (you can restrict this in production)
CORS(app, resources={r"/*": {"origins": "*"}})

# Handle ngrok warning page bypass
@app.before_request
def handle_ngrok_warning():
    """Bypass ngrok warning page by checking for ngrok-skip-browser-warning header"""
    pass

# ============== GLOBAL STATE ==============
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Using device: {device}")
if torch.cuda.is_available():
    print(f"[INIT] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[INIT] CUDA Version: {torch.version.cuda}")
    print(f"[INIT] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("[INIT] ⚠️ Running on CPU - GPU recommended for better performance")
    print("[INIT] 💡 Tip: Use Google Colab with GPU runtime for free GPU access")

# Load YOLO model
print("[INIT] Loading YOLO model...")
model = YOLO(MODEL_WEIGHTS)

# Load BLIP-2 model with error handling
print("[INIT] Loading BLIP-2 model (lighter Flan-T5 XL first)...")
try:
    # Prefer lighter BLIP-2 Flan-T5 XL for lower VRAM usage
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl", trust_remote_code=True)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    blip_model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-flan-t5-xl",
        trust_remote_code=True,
        torch_dtype=dtype
    ).to(device)
    print("[INIT] Successfully loaded BLIP-2 Flan-T5 XL (lighter)")
except Exception as e:
    print(f"[WARNING] BLIP-2 Flan-T5 XL failed to load: {e}")
    print("[INIT] Attempting BLIP-2 OPT-2.7B fallback...")
    try:
        processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b", trust_remote_code=True)
        blip_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            trust_remote_code=True,
            torch_dtype=torch.float32
        ).to(device)
        print("[INIT] Successfully loaded BLIP-2 OPT-2.7B fallback")
    except Exception as e2:
        print(f"[ERROR] BLIP-2 models failed: {e2}")
        print("[INIT] BLIP-2 disabled - Q&A and captioning will be unavailable")
        blip_model = None
        processor = None

# Initialize OCR (EasyOCR) cache so OCR works from the main server as well
print("[INIT] Loading EasyOCR readers (this may take a moment)...")
ocr_readers = {}
for lang_set in OCR_PRELOAD_LANG_SETS:
    try:
        key = tuple(lang_set)
        ocr_readers[key] = easyocr.Reader(
            lang_set, gpu=torch.cuda.is_available(), verbose=False
        )
        print(f"[INIT] OCR {lang_set}: ready")
    except Exception as e:
        print(f"[WARNING] OCR failed to load for {lang_set}: {e}")

def get_ocr_reader(langs):
    """
    Get or create an EasyOCR reader for the requested language list.
    Caches readers to avoid repeated heavy initialisation.
    """
    lang_key = tuple(langs)
    if lang_key in ocr_readers:
        return ocr_readers[lang_key]
    try:
        reader = easyocr.Reader(langs, gpu=torch.cuda.is_available(), verbose=False)
        ocr_readers[lang_key] = reader
        print(f"[OCR] Loaded reader for languages: {langs}")
        return reader
    except Exception as e:
        print(f"[OCR] Failed to load reader for {langs}: {e}")
        return None

# Initialize TTS (optional - frontend handles TTS via Web Speech API)
tts = None
try:
    tts = pyttsx3.init()
    tts.setProperty("rate", 160)
    print("[INIT] TTS initialized (server-side)")
except Exception as e:
    print(f"[WARNING] TTS unavailable (eSpeak not installed): {e}")
    print("[INIT] TTS disabled - frontend will handle speech via Web Speech API")
    tts = None

# Track memory for persistent tracking
track_memory = {}
last_global_speak = 0.0
last_caption_time = 0.0
last_wall_alert = 0.0
WALL_ALERT_COOLDOWN = 3.0  # 3 seconds between wall alerts

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
    """Text-to-speech (optional - frontend handles TTS)"""
    if tts is None:
        return  # TTS disabled, frontend will handle speech
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print(f"TTS error: {e}")

def blip_caption(frame):
    """Generate a detailed scene description using BLIP-2, focusing on notes, text, and objects."""
    try:
        if blip_model is None or processor is None:
            return None

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Enhanced prompt to describe notes, text, objects, and walls in detail
        prompt = (
            "You are an assistive guide for a blind user. "
            "Describe everything you see in detail, especially: "
            "any walls, barriers, or obstacles ahead and their approximate distance; "
            "any notes, papers, documents, or text being held; "
            "objects in the scene and their positions (left/center/right); "
            "people and what they might be doing; "
            "any text visible on signs, labels, or documents. "
            "If you see a wall, mention it clearly and estimate if it's close or far. "
            "Be specific and descriptive. Use 2-3 sentences."
        )

        inputs = processor(images=img_rgb, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_length=120, num_beams=3, temperature=0.7)
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"BLIP caption error: {e}")
        return None

def blip_qa(frame, question):
    """Answer question about frame using BLIP-2 with detailed guidance for notes and text."""
    try:
        if blip_model is None or processor is None:
            return "Q&A unavailable - BLIP-2 model not loaded"
        
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        prompt = (
            "You are an assistive guide for a blind user. "
            "Look at the image carefully and answer the user's question in detail. "
            "If there are notes, papers, or documents, describe what's written on them. "
            "If the question is about what a person is doing, describe the visible action clearly. "
            "If there are objects, describe their appearance and position (left/center/right). "
            "Be very specific and descriptive. If you are unsure, say you are not sure instead of guessing. "
            f"Question: {question}"
        )
        inputs = processor(images=img_rgb, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = blip_model.generate(**inputs, max_length=150, num_beams=4, temperature=0.7)
        answer = processor.decode(output_ids[0], skip_special_tokens=True)
        return answer
    except Exception as e:
        print(f"BLIP QA error: {e}")
        return f"Unable to answer due to model error: {str(e)[:50]}"

def encode_image_base64(image_array):
    """Encode OpenCV image to base64"""
    _, buffer = cv2.imencode('.png', image_array)
    return base64.b64encode(buffer).decode('utf-8')

def detect_wall_and_distance(frame):
    """
    Detect walls using edge detection and estimate distance.
    Returns wall info: {detected: bool, distance: float, distance_str: str, position: str}
    """
    global K_DEFAULT
    
    if K_DEFAULT is None:
        return {"detected": False, "distance": None, "distance_str": "?", "position": "unknown"}
    
    h, w = frame.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Edge detection to find vertical structures (walls)
    edges = cv2.Canny(gray, 50, 150)
    
    # Detect vertical lines (walls are typically vertical)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    wall_detected = False
    wall_distance = None
    wall_position = "center"
    
    if lines is not None and len(lines) > 10:  # Many vertical lines suggest a wall
        # Analyze the center region of the image (where walls are most likely)
        center_region = frame[h//3:2*h//3, w//4:3*w//4]
        
        # Calculate average depth using the center region
        # Walls typically have uniform texture, so we estimate based on region size
        region_height = center_region.shape[0]
        
        if region_height > 50:  # Reasonable region size
            # Estimate distance: larger region = closer wall
            # This is a heuristic - adjust based on calibration
            estimated_pixels = region_height
            wall_distance = K_DEFAULT / (estimated_pixels + 1e-6)
            
            # Determine position based on where most lines are
            vertical_lines = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 20:  # Nearly vertical
                    vertical_lines.append((x1, y1, x2, y2))
            
            if len(vertical_lines) > 5:
                wall_detected = True
                
                # Check if wall is on left, center, or right
                center_x = w // 2
                left_count = sum(1 for (x1, y1, x2, y2) in vertical_lines if (x1 + x2) / 2 < center_x - w//6)
                right_count = sum(1 for (x1, y1, x2, y2) in vertical_lines if (x1 + x2) / 2 > center_x + w//6)
                
                if left_count > right_count:
                    wall_position = "left"
                elif right_count > left_count:
                    wall_position = "right"
                else:
                    wall_position = "center"
    
    # Also check BLIP caption for wall mentions
    blip_wall_check = blip_caption(frame)
    if blip_wall_check and ("wall" in blip_wall_check.lower() or "barrier" in blip_wall_check.lower()):
        if not wall_detected:
            wall_detected = True
            wall_position = "ahead"
        # If BLIP mentions close wall, adjust distance estimate
        if "close" in blip_wall_check.lower() or "near" in blip_wall_check.lower():
            if wall_distance is None or wall_distance > 2.0:
                wall_distance = 1.5  # Estimate close wall
    
    return {
        "detected": wall_detected,
        "distance": wall_distance,
        "distance_str": format_distance(wall_distance) if wall_distance else "?",
        "position": wall_position
    }

def navigation_caption_from_detections(detections):
    """
    Build a concise navigation sentence from YOLO detections.
    Prioritises nearby obstacles and uses left/center/right wording.
    """
    if not detections:
        return "I don't see clear obstacles right now."

    # Sort by bbox height as a proxy for closeness
    ordered = sorted(detections, key=lambda d: d.get("bbox_height", 0), reverse=True)

    phrases = []
    for det in ordered[:4]:  # limit to top 4 to keep it short
        cls = det.get("class", "object")
        side = det.get("side", "ahead")
        dist_str = det.get("distance_str", "").strip()
        if dist_str and dist_str != "?":
            phrases.append(f"{cls} {dist_str} on your {side}")
        else:
            phrases.append(f"{cls} on your {side}")

    summary = "; ".join(phrases)
    return f"Navigation: {summary}."

def simple_detection_facts(frame):
    """
    Run YOLO on a frame and return a normalized detections list used across endpoints.
    """
    global K_DEFAULT
    h, w = frame.shape[:2]
    results = model.track(frame, persist=True, conf=CONF_THRESH)[0]
    detections = []

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
                "cy": (y1 + y2) // 2,
                "bbox_height": bh,
            })

    return detections

def qa_from_detections(question, detections):
    """
    Lightweight Q&A using detection facts when BLIP-2 is unavailable or fails.
    Handles 'how many', 'what is here', and spatial queries.
    """
    if not detections:
        return "I don't see anything clearly in view."

    q = question.lower()

    # Object presence
    if any(k in q for k in ["what is here", "what do you see", "what is around", "describe"]):
        names = list(dict.fromkeys([d["class"] for d in detections]))
        return f"I see: {', '.join(names)}."

    # Counting
    for keyword in ["how many", "count", "number of"]:
        if keyword in q:
            target = None
            words = q.split()
            for w in words:
                if w.isalpha() and w not in ["how", "many", "count", "of"]:
                    target = w
                    break
            if target:
                count = sum(1 for d in detections if target in d["class"].lower())
                return f"I can see {count} {target}(s)."

    # Side queries
    if "left" in q or "right" in q or "front" in q:
        left = [d["class"] for d in detections if d["side"] == "left"]
        center = [d["class"] for d in detections if d["side"] == "center"]
        right = [d["class"] for d in detections if d["side"] == "right"]
        parts = []
        if left: parts.append(f"On your left: {', '.join(list(dict.fromkeys(left)))}")
        if center: parts.append(f"In front: {', '.join(list(dict.fromkeys(center)))}")
        if right: parts.append(f"On your right: {', '.join(list(dict.fromkeys(right)))}")
        return "; ".join(parts) if parts else "I don't have a clear side-based view yet."

    # Action questions fallback when we only have detections
    if "doing" in q or "action" in q or "activity" in q:
        people = [d for d in detections if d["class"].lower() == "person"]
        if people:
            return "I can see people, but I cannot determine their exact action from detections alone."
        return "I cannot tell the action; I only see objects, no clear people detected."

    # Fallback generic description
    return navigation_caption_from_detections(detections)

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
    - wall_alert: wall detection info with distance
    """
    global track_memory, last_global_speak, last_caption_time, K_DEFAULT, last_wall_alert
    
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
        detections = simple_detection_facts(frame)

        # Draw detections and guides
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            dist_str = det.get("distance_str", "")
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{det['class']} {dist_str}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.line(frame, (int(w * LEFT_FRAC), 0), (int(w * LEFT_FRAC), h), (255, 255, 0), 1)
        cv2.line(frame, (int(w * RIGHT_FRAC), 0), (int(w * RIGHT_FRAC), h), (255, 255, 0), 1)

        # Generate captions
        now = time.time()
        blip_text = None
        if now - last_caption_time > CAPTION_INTERVAL:
            blip_text = blip_caption(frame)
            last_caption_time = now

        nav_caption = navigation_caption_from_detections(detections)
        caption = blip_text or nav_caption
        if blip_text and nav_caption:
            caption = f"{blip_text} {nav_caption}"

        # Detect walls and estimate distance
        wall_info = detect_wall_and_distance(frame)
        wall_alert = None
        
        # Generate wall alert with 3-second cooldown
        now = time.time()
        if wall_info["detected"] and (now - last_wall_alert) >= WALL_ALERT_COOLDOWN:
            wall_distance = wall_info.get("distance")
            wall_pos = wall_info.get("position", "ahead")
            
            if wall_distance and wall_distance < 3.0:  # Alert if wall is within 3 meters
                if wall_distance < 1.0:
                    alert_msg = f"⚠️ WALL VERY CLOSE! {wall_info['distance_str']} ahead on your {wall_pos}. Stop immediately!"
                elif wall_distance < 2.0:
                    alert_msg = f"⚠️ Wall detected {wall_info['distance_str']} ahead on your {wall_pos}. Slow down."
                else:
                    alert_msg = f"Wall ahead {wall_info['distance_str']} on your {wall_pos}."
                
                wall_alert = {
                    "message": alert_msg,
                    "distance": wall_distance,
                    "distance_str": wall_info["distance_str"],
                    "position": wall_pos,
                    "urgent": wall_distance < 1.5
                }
                last_wall_alert = now
            elif wall_info["detected"]:
                # Wall detected but far away
                wall_alert = {
                    "message": f"Wall detected ahead on your {wall_pos}.",
                    "distance": wall_distance,
                    "distance_str": wall_info["distance_str"],
                    "position": wall_pos,
                    "urgent": False
                }
                last_wall_alert = now

        annotated_b64 = encode_image_base64(frame)

        return jsonify({
            "detections": detections,
            "caption": caption or "",
            "navigation_caption": nav_caption,
            "has_objects": len(detections) > 0,
            "K_value": K_DEFAULT,
            "wall_alert": wall_alert,
            "wall_info": wall_info,
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

        # Try BLIP-2 first; only fall back if BLIP truly unavailable
        answer = blip_qa(frame, question)
        blip_missing = answer is None or ("not loaded" in answer.lower())
        if blip_missing:
            detections = simple_detection_facts(frame)
            answer = qa_from_detections(question, detections)
        
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

@app.route('/ocr', methods=['POST'])
def ocr():
    """
    Extract text from a frame using EasyOCR with light preprocessing.
    """
    if 'frame' not in request.files:
        return jsonify({"error": "No frame provided"}), 400

    try:
        file = request.files['frame']
        langs_raw = request.form.get("langs", "")
        langs = [s.strip() for s in langs_raw.split(",") if s.strip()] or OCR_LANG_DEFAULT
        # Easy cap to prevent huge initialisation; adjust as needed
        langs = langs[:3]
        reader = get_ocr_reader(langs)
        if reader is None:
            return jsonify({"error": f"OCR unavailable for languages: {langs}"}), 503
        file_bytes = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return jsonify({"error": "Invalid image"}), 400

        # Downscale large images to speed up OCR while keeping readability
        max_side = max(frame.shape[:2])
        if max_side > 1280:
            scale = 1280 / max_side
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Improve contrast for small/low-light text
        gray = cv2.equalizeHist(gray)
        gray = cv2.medianBlur(gray, 3)

        results = reader.readtext(gray)
        lines = [r[1] for r in results if r and len(r) >= 2]
        text = "\n".join(lines).strip()

        return jsonify({
            "success": True,
            "text": text,
            "line_count": len(lines),
            "languages": langs
        })
    except Exception as e:
        print(f"OCR error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/ocr_url', methods=['POST'])
def ocr_url():
    """
    Extract text from an image URL (backend proxy to handle CORS).
    """
    if 'url' not in request.form:
        return jsonify({"error": "No URL provided"}), 400

    try:
        image_url = request.form['url']
        langs_raw = request.form.get("langs", "")
        langs = [s.strip() for s in langs_raw.split(",") if s.strip()] or OCR_LANG_DEFAULT
        langs = langs[:3]
        
        reader = get_ocr_reader(langs)
        if reader is None:
            return jsonify({"error": f"OCR unavailable for languages: {langs}"}), 503

        # Fetch image from URL using requests (backend can handle CORS)
        import requests
        
        # Set headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(image_url, timeout=15, stream=True, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('content-type', '').lower()
        if not content_type.startswith('image/'):
            # Try to decode anyway - sometimes content-type is wrong
            print(f"[OCR URL] Warning: Content-Type is '{content_type}', not image. Attempting to decode anyway...")
            # Check if it's HTML (common error)
            if 'text/html' in content_type or response.content[:100].startswith(b'<'):
                return jsonify({"error": "URL does not point to an image. It appears to be a webpage. Please use a direct image link (right-click image > Copy image address)."}), 400
        
        # Convert response to numpy array
        img_array = np.frombuffer(response.content, np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if frame is None:
            return jsonify({"error": "Invalid image format. Could not decode image. Please ensure the URL points to a valid image file (JPG, PNG, etc.)."}), 400

        # Downscale if too large
        max_side = max(frame.shape[:2])
        if max_side > 1280:
            scale = 1280 / max_side
            frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

        # Preprocess
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        gray = cv2.medianBlur(gray, 3)

        # OCR
        results = reader.readtext(gray)
        lines = [r[1] for r in results if r and len(r) >= 2]
        text = "\n".join(lines).strip()

        return jsonify({
            "success": True,
            "text": text,
            "line_count": len(lines),
            "languages": langs
        })
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if hasattr(e, 'response') else None
        if status_code == 403:
            return jsonify({"error": "Access denied (403 Forbidden). The URL may require authentication or block automated access. Try a public image hosting service."}), 403
        elif status_code == 404:
            return jsonify({"error": "Image not found (404). Please check the URL."}), 404
        else:
            return jsonify({"error": f"HTTP error {status_code}: {str(e)}"}), status_code or 400
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to fetch image from URL: {str(e)}"}), 400
    except Exception as e:
        print(f"OCR URL error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/ocr_pdf', methods=['POST'])
def ocr_pdf():
    """
    Extract text from PDF file by converting pages to images and using OCR.
    Note: Requires pdf2image library. For production, consider using pdf2image + OCR.
    """
    if 'pdf' not in request.files:
        return jsonify({"error": "No PDF file provided"}), 400

    try:
        pdf_file = request.files['pdf']
        langs_raw = request.form.get("langs", "")
        langs = [s.strip() for s in langs_raw.split(",") if s.strip()] or OCR_LANG_DEFAULT
        langs = langs[:3]
        
        reader = get_ocr_reader(langs)
        if reader is None:
            return jsonify({"error": f"OCR unavailable for languages: {langs}"}), 503

        # Read PDF file
        pdf_bytes = pdf_file.read()
        
        # Try to use pdf2image if available, otherwise return error
        try:
            from pdf2image import convert_from_bytes
            from PIL import Image
            
            print(f"[OCR PDF] Converting PDF to images...")
            # Convert PDF pages to images (first 5 pages for performance)
            images = convert_from_bytes(pdf_bytes, first_page=1, last_page=5, dpi=200)
            print(f"[OCR PDF] Converted {len(images)} page(s) to images")
            
            all_text = []
            for i, img in enumerate(images):
                print(f"[OCR PDF] Processing page {i+1}/{len(images)}...")
                # Convert PIL to OpenCV
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    frame = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                else:
                    frame = img_array
                
                # Preprocess
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                gray = cv2.equalizeHist(gray)
                gray = cv2.medianBlur(gray, 3)
                
                # OCR
                results = reader.readtext(gray)
                page_text = [r[1] for r in results if r and len(r) >= 2]
                if page_text:
                    all_text.append(f"--- Page {i+1} ---\n" + "\n".join(page_text))
            
            text = "\n\n".join(all_text).strip()
            
            return jsonify({
                "success": True,
                "text": text,
                "line_count": len([l for t in all_text for l in t.split('\n')]),
                "pages_processed": len(images),
                "languages": langs
            })
            
        except ImportError as e:
            print(f"[OCR PDF] Import error: {e}")
            return jsonify({
                "error": "PDF processing requires pdf2image library. Install with: pip install pdf2image pillow. Also install poppler-utils system package.",
                "text": "",
                "line_count": 0
            }), 503
        except Exception as e:
            print(f"PDF processing error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"error": f"PDF processing failed: {str(e)}"}), 500
            
    except Exception as e:
        print(f"OCR PDF error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

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

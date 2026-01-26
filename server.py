"""
NutriSnap Inference Server
Detects vegetables/ingredients from images using YOLOv8
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from ultralytics import YOLO
import base64
import numpy as np
import cv2
import os
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native

# =============================================
# CONFIGURATION
# =============================================

# Path to your trained model
MODEL_PATH = 'best.pt'

# Confidence threshold (0.5 = 50%)
CONFIDENCE_THRESHOLD = 0.7

# Your 13 detectable ingredients (MUST match your training classes)
CLASS_NAMES = [
    'cabbage',      # 0
    'capsicum',     # 1
    'carrot',       # 2
    'cauliflower',  # 3
    'eggplant',     # 4
    'garlic',       # 5
    'ginger',       # 6
    'onion',        # 7
    'peas',         # 8
    'potato',       # 9
    'pumpkin',      # 10
    'radish',       # 11
    'tomato'        # 12
]

# =============================================
# LOAD MODEL
# =============================================

logger.info("Loading YOLO model...")
if os.path.exists(MODEL_PATH):
    model = YOLO(MODEL_PATH)
    logger.info("✅ Model loaded successfully!")
else:
    logger.error(f"❌ Model not found at {MODEL_PATH}")
    model = None

# =============================================
# API ROUTES
# =============================================

@app.route('/', methods=['GET'])
def home():
    """Welcome endpoint"""
    return jsonify({
        'name': 'NutriSnap Inference API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'endpoints': {
            'GET /': 'This welcome message',
            'GET /health': 'Check server health',
            'GET /classes': 'List all detectable ingredients',
            'POST /detect': 'Detect ingredients from image'
        },
        'usage': {
            'detect': {
                'method': 'POST',
                'body': '{"image": "base64_encoded_image_string"}',
                'returns': '{"success": true, "detected_ingredients": ["tomato", "onion"]}'
            }
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat(),
        'classes_count': len(CLASS_NAMES)
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get all detectable ingredient classes"""
    return jsonify({
        'success': True,
        'total': len(CLASS_NAMES),
        'classes': CLASS_NAMES,
        'classes_with_index': {i: name for i, name in enumerate(CLASS_NAMES)}
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Main detection endpoint
    Accepts: JSON with base64 encoded image
    Returns: List of detected ingredients
    """
    
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded. Check if best.pt exists.'
        }), 500
    
    try:
        # Get JSON data
        data = request.json
        
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'No image provided',
                'usage': 'Send POST with {"image": "base64_string"}'
            }), 400
        
        image_data = data['image']
        
        # Remove data URL prefix if present (from React Native)
        if 'base64,' in image_data:
            image_data = image_data.split('base64,')[1]
        
        # Decode base64 to image
        try:
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Failed to decode image: {str(e)}'
            }), 400
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image. Make sure it is valid base64.'
            }), 400
        
        logger.info(f"Processing image of size: {image.shape}")
        
        # Run YOLO inference
        results = model(image, conf=CONFIDENCE_THRESHOLD)
        
        # Extract detected ingredients
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # Get class name safely
                if class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                else:
                    # Fallback to model's own class names
                    class_name = result.names.get(class_id, f'unknown_{class_id}')
                
                detections.append({
                    'name': class_name,
                    'confidence': round(confidence, 3),
                    'bounding_box': {
                        'x1': round(x1, 2),
                        'y1': round(y1, 2),
                        'x2': round(x2, 2),
                        'y2': round(y2, 2)
                    }
                })
        
        # Get unique ingredient names (for recipe matching)
        unique_ingredients = list(set([d['name'] for d in detections]))
        
        logger.info(f"Detected: {unique_ingredients}")
        
        return jsonify({
            'success': True,
            'detected_ingredients': unique_ingredients,
            'count': len(unique_ingredients),
            'detections': detections,
            'total_detections': len(detections)
        })
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/detect-file', methods=['POST'])
def detect_file():
    """
    Alternative endpoint - accepts image file upload
    Useful for testing with tools like Postman
    """
    
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided',
            'usage': 'Send POST with form-data, key="image", value=<file>'
        }), 400
    
    try:
        file = request.files['image']
        
        # Read image file
        image_bytes = file.read()
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Could not decode image file'
            }), 400
        
        # Run inference
        results = model(image, conf=CONFIDENCE_THRESHOLD)
        
        # Extract detections
        detections = []
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else 'unknown'
                
                detections.append({
                    'name': class_name,
                    'confidence': round(confidence, 3)
                })
        
        unique_ingredients = list(set([d['name'] for d in detections]))
        
        return jsonify({
            'success': True,
            'detected_ingredients': unique_ingredients,
            'count': len(unique_ingredients),
            'detections': detections
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================
# RUN SERVER
# =============================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🥗 NUTRISNAP INFERENCE SERVER")
    print("="*60)
    print(f"📦 Model: {MODEL_PATH}")
    print(f"📊 Classes: {len(CLASS_NAMES)} ingredients")
    print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    print("="*60)
    print("\n📡 Endpoints:")
    print("   GET  http://localhost:5000/         - Welcome")
    print("   GET  http://localhost:5000/health   - Health check")
    print("   GET  http://localhost:5000/classes  - List classes")
    print("   POST http://localhost:5000/detect   - Detect ingredients")
    print("\n" + "="*60 + "\n")
    
    # Run Flask server
    app.run(
        host='0.0.0.0',  # Accessible from other devices
        port=5000,
        debug=True
    )
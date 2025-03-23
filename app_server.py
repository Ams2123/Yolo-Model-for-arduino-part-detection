# app.py
from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import os
from PIL import Image

app = Flask(__name__)

# Load YOLO model
# model = YOLO('best.pt')  # Replace with your model path

model = YOLO(r'E:\Selvaganesh\Hackathons\SNUC\arduino_mega_pinout_diagram\aruduino_mega_v8.pt')

def process_image(image_data,model):
    # Decode base64 image
    encoded_data = image_data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 640))
    # Run inference
    height, width, channels = img.shape
    xsc = width/640
    ysc = height/640
    results = model.predict(img)
    
    # Process results
    detections = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = model.names[cls]
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class': name
            })
    
    return detections

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.json:
        return jsonify({'error': 'No image provided'}), 400
    
    image_data = request.json['image']
    detections = process_image(image_data,model)
    return jsonify({'detections': detections})

if __name__ == '__main__':
    app.run(debug=True)
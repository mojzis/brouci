"""
Flask app for capturing video frames and saving them as images
"""
from flask import Flask, render_template, jsonify, Response
import cv2
import os
from pathlib import Path
import time
import json
import threading

app = Flask(__name__, template_folder='../templates')

# Global variables
capturing = False
frame_count = 0
capture_interval = 0.5
last_capture_time = 0
cap = None
output_dir = Path("tests/video_frames_2")

def init_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

def generate_frames():
    global capturing, frame_count, last_capture_time, capture_interval
    
    camera = init_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        current_time = time.time()
        
        # Save frame if capturing and interval has passed
        if capturing and (current_time - last_capture_time) >= capture_interval:
            filename = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(filename), frame)
            frame_count += 1
            last_capture_time = current_time
        
        # Add capture indicator
        if capturing:
            cv2.putText(frame, "CAPTURING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('frame_capture.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_capture', methods=['POST'])
def toggle_capture():
    global capturing, last_capture_time
    capturing = not capturing
    if capturing:
        last_capture_time = time.time()
    return jsonify({'capturing': capturing, 'frame_count': frame_count})

@app.route('/reset', methods=['POST'])
def reset():
    global frame_count
    frame_count = 0
    
    # Clear directory
    for file in output_dir.glob("*.jpg"):
        file.unlink()
    
    return jsonify({'frame_count': frame_count})

@app.route('/set_interval/<float:interval>', methods=['POST'])
def set_interval(interval):
    global capture_interval
    capture_interval = max(0.1, min(2.0, interval))
    return jsonify({'capture_interval': capture_interval})

@app.route('/status')
def status():
    return jsonify({
        'capturing': capturing,
        'frame_count': frame_count,
        'capture_interval': capture_interval
    })

if __name__ == '__main__':
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
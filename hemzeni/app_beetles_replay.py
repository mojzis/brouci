"""
Flask app for replaying saved frames with beetle behavior for debugging
"""
from flask import Flask, render_template, jsonify, Response, request
import cv2
import numpy as np
from pathlib import Path
import json
import base64
from numpy.typing import NDArray

from hemzeni.beetle import Beetle
from hemzeni.pose_detector import PoseDetector
from hemzeni.controller import BeetleController

app = Flask(__name__, template_folder='../templates')

# Global variables
frame_dir = Path("tests/video_frames_2")
frames = []
frame_index = 0
playing = False
pose_detector = None
controller = None

def init_app():
    global frames, pose_detector, controller
    
    # Load frames
    frames = load_frames(frame_dir)
    
    # Initialize detector and controller
    pose_detector = PoseDetector()
    controller = BeetleController()

def load_frames(frame_dir: Path) -> list[NDArray[np.uint8]]:
    """Load all frames from directory"""
    frames = []
    frame_files = sorted(frame_dir.glob("*.jpg"))
    
    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is not None:
            frames.append(frame)
    
    return frames

def process_frame(index: int, show_beetles: bool = True, show_pose: bool = True) -> tuple[str, dict]:
    """Process a single frame and return base64 encoded image and debug info"""
    global frames, pose_detector, controller
    
    if not frames or index < 0 or index >= len(frames):
        return None, None
    
    frame = frames[index].copy()
    
    # Detect poses
    human_positions, human_bboxes, head_positions, keypoint_array, detected_keypoints = pose_detector.detect(frame)
    
    # Update beetles with detected humans
    controller._update_beetles(human_positions, human_bboxes)
    
    # Draw poses
    if show_pose and len(keypoint_array) > 0:
        # Draw keypoints from the keypoint array
        for person_keypoints in keypoint_array:
            for i in range(min(17, len(person_keypoints) // 3)):
                idx = i * 3
                if idx + 2 < len(person_keypoints):
                    x, y, conf = person_keypoints[idx], person_keypoints[idx + 1], person_keypoints[idx + 2]
                    if conf > 0.5 and x > 0 and y > 0:
                        cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    
    # Draw beetles
    if show_beetles:
        for beetle in controller.beetles:
            beetle.draw(frame)
    
    # Add frame info
    cv2.putText(frame, f"Frame: {index}/{len(frames)-1}", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Encode frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare debug info
    debug_info = {
        'frame_index': index,
        'total_frames': len(frames),
        'poses_count': len(human_positions),
        'beetles': []
    }
    
    for beetle in controller.beetles:
        debug_info['beetles'].append({
            'name': beetle.name,
            'x': round(beetle.x),
            'y': round(beetle.y),
            'state': beetle.state,
            'speed': round(beetle.speed, 1),
            'direction': round(beetle.direction, 2)
        })
    
    if human_positions:
        debug_info['poses'] = []
        for i, pos in enumerate(human_positions):
            debug_info['poses'].append({
                'id': i,
                'position': {'x': round(pos[0]), 'y': round(pos[1])}
            })
    
    return frame_base64, debug_info

@app.route('/')
def index():
    if not frames:
        return render_template('error.html', 
                             message=f"No frames found in {frame_dir}. Please run app_frame_capture.py first.")
    return render_template('beetle_replay.html', total_frames=len(frames))

@app.route('/frame/<int:index>')
def get_frame(index):
    show_beetles = request.args.get('show_beetles', 'true').lower() == 'true'
    show_pose = request.args.get('show_pose', 'true').lower() == 'true'
    
    frame_base64, debug_info = process_frame(index, show_beetles, show_pose)
    
    if frame_base64 is None:
        return jsonify({'error': 'Invalid frame index'}), 400
    
    return jsonify({
        'frame': frame_base64,
        'debug_info': debug_info
    })

@app.route('/reset', methods=['POST'])
def reset():
    global controller
    controller = BeetleController()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    init_app()
    app.run(debug=True, host='0.0.0.0', port=5002)
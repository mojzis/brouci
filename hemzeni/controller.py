import os
import time

import cv2
import streamlit as st
from aipose.plot import plot
from numpy.typing import NDArray
import numpy as np

from beetle import Beetle
from pose_detector import PoseDetector


class BeetleController:
    def __init__(self) -> None:
        self.camera: cv2.VideoCapture = cv2.VideoCapture(0)
        self.pose_detector: PoseDetector = PoseDetector(detection_scale=0.2, detection_interval=1)
        
        # Performance settings
        self.target_fps: int = 30
        self.frame_delay: float = 1.0 / self.target_fps
        self.last_frame_time: float = 0
        
        # Get initial frame dimensions
        ret: bool
        frame: NDArray[np.uint8]
        ret, frame = self.camera.read()
        if ret:
            self.frame_height: int
            self.frame_width: int
            self.frame_height, self.frame_width = frame.shape[:2]
        else:
            self.frame_height = 480  # Default values
            self.frame_width = 640
        
        self.beetles: list[Beetle] = []
        self.initialized: bool = False

    def initialize_beetles(self) -> None:
        """Initialize beetle objects"""
        if self.initialized:
            return
            
        try:
            kudlanka_path = os.path.join("img", "kudlanka.png")
            blecha_path = os.path.join("img", "blecha.png")

            self.beetles = [
                Beetle("kudlanka", kudlanka_path, self.frame_width, self.frame_height),
                Beetle("blecha", blecha_path, self.frame_width, self.frame_height),
            ]
            self.initialized = True
        except Exception as e:
            st.error(f"Failed to load beetle images: {e}")
            self.beetles = []
            self.initialized = True

    def process_frame(self) -> tuple[NDArray[np.uint8], list[tuple[float, float]]] | None:
        """Process a single frame and return the annotated image"""
        # Frame rate control
        current_time = time.time()
        if current_time - self.last_frame_time < self.frame_delay:
            time.sleep(0.01)
            return None
            
        self.last_frame_time = current_time
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            return None
            
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect humans
        human_positions, keypoint_array, raw_keypoints = self.pose_detector.detect(frame)
        
        # Update beetles
        self._update_beetles(human_positions)
        
        # Draw beetles
        for beetle in self.beetles:
            beetle.draw(frame)
        
        # Draw pose landmarks
        if len(keypoint_array) > 0:
            annotated_frame = plot(
                frame,
                keypoint_array,
                plot_image=False,
                return_img=True,
            )
        else:
            annotated_frame = frame.copy()
        
        # Update session state
        self._update_session_state(keypoint_array, raw_keypoints, annotated_frame)
        
        return annotated_frame, human_positions

    def _update_beetles(self, human_positions: list[tuple[float, float]]) -> None:
        """Update all beetle positions based on human positions"""
        for beetle in self.beetles:
            old_x, old_y = beetle.x, beetle.y

            if human_positions:
                beetle.update_fleeing(human_positions, self.beetles)
            else:
                beetle.update_roaming(self.beetles)

            # Debug beetle movement
            if abs(beetle.x - old_x) > 1 or abs(beetle.y - old_y) > 1:
                print(
                    f"Beetle {beetle.name} moved from ({old_x:.1f},{old_y:.1f}) to ({beetle.x:.1f},{beetle.y:.1f})"
                )

    def _update_session_state(self, keypoint_array: NDArray[np.float64], raw_keypoints: list, frame: NDArray[np.uint8]) -> None:
        """Update Streamlit session state with detection results"""
        if "pred_list" not in st.session_state:
            st.session_state.pred_list = []
            
        if len(keypoint_array) > 0:
            st.session_state.pred_list.append(keypoint_array)
            
        try:
            if len(raw_keypoints) > 0:
                st.session_state.hand = raw_keypoints[0].get_keypoint(9)
        except (IndexError, AttributeError):
            pass
            
        st.session_state.img = frame

    def get_status_message(self, human_positions: list[tuple[float, float]]) -> tuple[str, str]:
        """Get status message based on human detection"""
        if human_positions:
            return f"🏃 {len(human_positions)} human(s) detected - beetles are fleeing!", "success"
        else:
            return "🪲 No humans detected - beetles are roaming freely", "info"

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()
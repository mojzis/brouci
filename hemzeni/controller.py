import logging
import os
import time

import cv2
import numpy as np
from aipose.plot import plot
from numpy.typing import NDArray

from hemzeni.beetle import Beetle, HeadFollower
from hemzeni.pose_detector import PoseDetector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BeetleController:
    def __init__(self, debug: bool = False) -> None:
        self.camera: cv2.VideoCapture = cv2.VideoCapture(0)
        self.pose_detector: PoseDetector = PoseDetector(
            detection_scale=0.2, detection_interval=1, debug=debug
        )

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
        self.head_follower: HeadFollower | None = None
        self.initialized: bool = False
        self.debug: bool = debug

    def initialize_beetles(self) -> None:
        """Initialize beetle objects"""
        if self.initialized:
            return

        try:
            kudlanka_path = os.path.join("img", "kudlanka.png")
            blecha_path = os.path.join("img", "blecha.png")
            lachticek_path = os.path.join("img", "lachticek.png")

            self.beetles = [
                Beetle(
                    "kudlanka",
                    kudlanka_path,
                    self.frame_width,
                    self.frame_height,
                    self.debug,
                ),
                Beetle(
                    "blecha",
                    blecha_path,
                    self.frame_width,
                    self.frame_height,
                    self.debug,
                ),
            ]

            # Initialize head follower
            self.head_follower = HeadFollower(
                "lachticek",
                lachticek_path,
                self.frame_width,
                self.frame_height,
                self.debug,
            )

            self.initialized = True
            # print(f"☠️ {self.head_follower.name}")
        except Exception as e:
            logger.error(f"Failed to load beetle images: {e}")
            self.beetles = []
            self.head_follower = None
            self.initialized = True

    def process_frame(
        self,
    ) -> (
        tuple[
            NDArray[np.uint8],
            list[tuple[float, float]],
            list[tuple[float, float, float, float]],
        ]
        | None
    ):
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
            logger.error("Failed to capture frame from camera")
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect humans
        human_positions, human_bboxes, head_positions, keypoint_array, raw_keypoints = (
            self.pose_detector.detect(frame)
        )
        if self.debug:
            print(f"🔍 Detected: {len(human_positions)} humans, {len(head_positions)} heads")

        # Update beetles
        self._update_beetles(human_positions, human_bboxes)

        # Update head follower
        if self.head_follower:
            if self.debug:
                print(f"🎛️ Controller calling head_follower.update_following with {len(head_positions)} heads")
            self.head_follower.update_following(head_positions)
        else:
            if self.debug:
                print("❌ Controller: no head_follower to update")

        # Draw beetles
        for beetle in self.beetles:
            beetle.draw(frame)

        # Draw head follower
        if self.head_follower:
            if self.debug:
                print(f"🎛️ Controller calling head_follower.draw()")
            self.head_follower.draw(frame)
        else:
            if self.debug:
                print("❌ Controller: no head_follower to draw")

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
        # self._update_session_state(keypoint_array, raw_keypoints, annotated_frame)

        return annotated_frame, human_positions, human_bboxes

    def _update_beetles(
        self,
        human_positions: list[tuple[float, float]],
        human_bboxes: list[tuple[float, float, float, float]],
    ) -> None:
        """Update all beetle positions based on human positions"""
        for beetle in self.beetles:
            old_x, old_y = beetle.x, beetle.y

            if human_positions:
                beetle.update_fleeing(human_positions, human_bboxes, self.beetles)
            else:
                beetle.update_roaming(self.beetles)

            # Debug beetle movement
            if self.debug and (abs(beetle.x - old_x) > 1 or abs(beetle.y - old_y) > 1):
                print(
                    f"Beetle {beetle.name} moved from ({old_x:.1f},{old_y:.1f}) to ({beetle.x:.1f},{beetle.y:.1f})"
                )

    def _update_session_state(
        self,
        keypoint_array: NDArray[np.float64],
        raw_keypoints: list,
        frame: NDArray[np.uint8],
    ) -> None:
        """Update session state with detection results (Streamlit compatibility)"""
        # This method is for Streamlit compatibility only
        # When used with Flask, this does nothing
        try:
            import streamlit as st

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
        except ImportError:
            # Streamlit not available - skip session state updates
            pass

    def get_status_message(
        self, human_positions: list[tuple[float, float]]
    ) -> tuple[str, str]:
        """Get status message based on human detection"""
        if human_positions:
            return (
                f"🏃 {len(human_positions)} human(s) detected - beetles are fleeing!",
                "success",
            )
        else:
            return "🪲 No humans detected - beetles are roaming freely", "info"

    def cleanup(self) -> None:
        """Clean up resources"""
        if self.camera is not None:
            self.camera.release()

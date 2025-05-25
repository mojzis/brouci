import logging
import time

import cv2
import numpy as np
from aipose.models.yolov7.domain import YoloV7Pose
from numpy.typing import NDArray

# Suppress YOLOv7 model logging
logging.getLogger("models.common").setLevel(logging.ERROR)
logging.getLogger("models.experimental").setLevel(logging.ERROR)
logging.getLogger("models.yolo").setLevel(logging.ERROR)
logging.getLogger("aipose").setLevel(logging.WARNING)


class PoseDetector:
    def __init__(self, detection_scale: float = 0.2, detection_interval: int = 1) -> None:
        self.model: YoloV7Pose = YoloV7Pose()
        self.detection_scale: float = detection_scale
        self.detection_interval: int = detection_interval
        self.frame_count: int = 0
        self.last_human_positions: list[tuple[float, float]] = []
        self.last_keypoints: NDArray[np.float64] = np.array([])

    def detect(self, frame: NDArray[np.uint8]) -> tuple[list[tuple[float, float]], NDArray[np.float64], list]:
        """Run pose detection on frame and return human positions and keypoints"""
        self.frame_count += 1
        
        # Only run detection every N frames
        if self.frame_count % self.detection_interval == 0:
            original_height, original_width = frame.shape[:2]
            
            # Scale down for faster detection
            small_height = int(original_height * self.detection_scale)
            small_width = int(original_width * self.detection_scale)
            small_frame = cv2.resize(frame, (small_width, small_height))
            
            # Detect poses
            keypoints = self.model(small_frame)
            
            # Extract human positions
            raw_positions = self._extract_human_positions(keypoints)
            
            # Scale positions back to original frame size
            human_positions = []
            for pos in raw_positions:
                scaled_x = pos[0] / self.detection_scale
                scaled_y = pos[1] / self.detection_scale
                human_positions.append((scaled_x, scaled_y))
            
            # Prepare keypoint array for plotting
            keypoint_array = np.array([value.raw_keypoint for value in keypoints])
            if self.detection_scale != 1.0:
                keypoint_array = keypoint_array / self.detection_scale
            
            # Cache results
            self.last_human_positions = human_positions
            self.last_keypoints = keypoint_array
            
            return human_positions, keypoint_array, keypoints
        else:
            # Return cached results
            return self.last_human_positions, self.last_keypoints, []

    def _extract_human_positions(self, keypoints: list) -> list[tuple[float, float]]:
        """Extract human center positions from pose keypoints"""
        positions = []

        try:
            # Reduced debug output
            if len(keypoints) > 0 and int(time.time() * 2) % 10 == 0:  # Log every 5 seconds
                print(f"Number of keypoints detected: {len(keypoints)}")

            for person_keypoints in keypoints:
                # Get torso center from raw keypoints
                raw_kp = person_keypoints.raw_keypoint

                # Try different approaches based on the actual data structure
                valid_points = []

                # Approach 1: If it's a 2D array (17, 3)
                if (
                    isinstance(raw_kp, np.ndarray)
                    and len(raw_kp.shape) == 2
                    and raw_kp.shape[0] >= 17
                ):
                    for idx in [0, 5, 6, 11, 12]:  # nose, shoulders, hips
                        if idx < raw_kp.shape[0]:
                            point = raw_kp[idx]
                            if (
                                point[0] > 0
                                and point[1] > 0
                                and (len(point) < 3 or point[2] > 0.3)
                            ):
                                valid_points.append([point[0], point[1]])

                # Approach 2: If it's a 1D array with 51 elements (17 * 3)
                elif (
                    isinstance(raw_kp, np.ndarray)
                    and len(raw_kp.shape) == 1
                    and raw_kp.shape[0] >= 51
                ):
                    for idx in [0, 5, 6, 11, 12]:  # nose, shoulders, hips
                        base_idx = idx * 3
                        if base_idx + 2 < raw_kp.shape[0]:
                            x, y, conf = (
                                raw_kp[base_idx],
                                raw_kp[base_idx + 1],
                                raw_kp[base_idx + 2],
                            )
                            if x > 0 and y > 0 and conf > 0.3:
                                valid_points.append([x, y])

                # Calculate center if we have valid points
                if valid_points:
                    center_x = sum(p[0] for p in valid_points) / len(valid_points)
                    center_y = sum(p[1] for p in valid_points) / len(valid_points)
                    positions.append((center_x, center_y))

        except Exception as e:
            if int(time.time() * 2) % 20 == 0:  # Log errors every 10 seconds max
                print(f"Error processing keypoints: {e}")
        
        return positions
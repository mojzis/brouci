import logging
import time
import warnings

import cv2
import numpy as np
from aipose.models.yolov7.domain import YoloV7Pose
from numpy.typing import NDArray

# Suppress warnings and logging
warnings.filterwarnings("ignore")
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("torchvision").setLevel(logging.ERROR)
logging.getLogger("models.common").setLevel(logging.ERROR)
logging.getLogger("models.experimental").setLevel(logging.ERROR)
logging.getLogger("models.yolo").setLevel(logging.ERROR)
logging.getLogger("aipose").setLevel(logging.WARNING)


class PoseDetector:
    def __init__(
        self, detection_scale: float = 0.2, detection_interval: int = 1, debug: bool = False
    ) -> None:
        self.model: YoloV7Pose = YoloV7Pose()
        self.detection_scale: float = detection_scale
        self.detection_interval: int = detection_interval
        self.debug: bool = debug
        self.frame_count: int = 0
        self.last_human_positions: list[tuple[float, float]] = []
        self.last_human_bboxes: list[tuple[float, float, float, float]] = []
        self.last_head_positions: list[tuple[float, float]] = []
        self.last_keypoints: NDArray[np.float64] = np.array([])

    def detect(
        self, frame: NDArray[np.uint8]
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[float, float, float, float]],
        list[tuple[float, float]],
        NDArray[np.float64],
        list,
    ]:
        """Run pose detection on frame and return human positions, bounding boxes, head positions, and keypoints"""
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

            # Extract human positions, bounding boxes, and head positions
            raw_positions, raw_bboxes, raw_heads = self._extract_human_data(keypoints)

            # Scale positions, bboxes, and heads back to original frame size
            human_positions = []
            human_bboxes = []
            head_positions = []

            for pos in raw_positions:
                scaled_x = pos[0] / self.detection_scale
                scaled_y = pos[1] / self.detection_scale
                human_positions.append((scaled_x, scaled_y))

            for bbox in raw_bboxes:
                x, y, w, h = bbox
                scaled_bbox = (
                    x / self.detection_scale,
                    y / self.detection_scale,
                    w / self.detection_scale,
                    h / self.detection_scale,
                )
                human_bboxes.append(scaled_bbox)

            for head in raw_heads:
                scaled_x = head[0] / self.detection_scale
                scaled_y = head[1] / self.detection_scale
                head_positions.append((scaled_x, scaled_y))

            # Prepare keypoint array for plotting
            keypoint_array = np.array([value.raw_keypoint for value in keypoints])
            if self.detection_scale != 1.0:
                keypoint_array = keypoint_array / self.detection_scale

            # Cache results
            self.last_human_positions = human_positions
            self.last_human_bboxes = human_bboxes
            self.last_head_positions = head_positions
            self.last_keypoints = keypoint_array

            return (
                human_positions,
                human_bboxes,
                head_positions,
                keypoint_array,
                keypoints,
            )
        else:
            # Return cached results
            return (
                self.last_human_positions,
                self.last_human_bboxes,
                self.last_head_positions,
                self.last_keypoints,
                [],
            )

    def _extract_human_data(
        self, keypoints: list
    ) -> tuple[
        list[tuple[float, float]],
        list[tuple[float, float, float, float]],
        list[tuple[float, float]],
    ]:
        """Extract human center positions, bounding boxes, and head positions from pose keypoints"""
        positions = []
        bboxes = []
        heads = []

        try:
            # Reduced debug output
            if self.debug and (
                len(keypoints) > 0 and int(time.time() * 2) % 10 == 0
            ):  # Log every 5 seconds
                print(f"Number of keypoints detected: {len(keypoints)}")

            for person_idx, person_keypoints in enumerate(keypoints):
                # Get torso center from raw keypoints
                raw_kp = person_keypoints.raw_keypoint
                if self.debug:
                    print(
                        f"🔍 Person {person_idx}: raw_kp shape={getattr(raw_kp, 'shape', 'no shape')}, type={type(raw_kp)}"
                    )

                # Try different approaches based on the actual data structure
                valid_points = []
                all_points = []  # For bounding box calculation
                nose_pos = None  # For head tracking

                # Approach 1: If it's a 2D array (17, 3)
                if (
                    isinstance(raw_kp, np.ndarray)
                    and len(raw_kp.shape) == 2
                    and raw_kp.shape[0] >= 17
                ):
                    if self.debug:
                        print("   Using approach 1 (2D array)")
                    for idx in range(min(17, raw_kp.shape[0])):
                        point = raw_kp[idx]
                        if point[0] > 0 and point[1] > 0:
                            all_points.append([point[0], point[1]])
                            # Try nose keypoint for head tracking
                            if idx == 0:  # nose keypoint
                                confidence = point[2] if len(point) > 2 else 1.0
                                if self.debug:
                                    print(
                                        f"   👃 Nose keypoint: x={point[0]:.1f}, y={point[1]:.1f}, conf={confidence:.3f}"
                                    )
                                if len(point) < 3 or confidence > 0.3:
                                    nose_pos = (point[0], point[1])
                                    if self.debug:
                                        print("   ✅ Nose accepted!")
                                else:
                                    if self.debug:
                                        print(
                                            f"   ❌ Nose rejected (confidence {confidence:.3f} <= 0.3)"
                                        )
                            if idx in [0, 5, 6, 11, 12]:  # nose, shoulders, hips
                                if len(point) < 3 or point[2] > 0.3:
                                    valid_points.append([point[0], point[1]])

                # Approach 2: If it's a 1D array with keypoint data (17 * 3 = 51, or more)
                elif (
                    isinstance(raw_kp, np.ndarray)
                    and len(raw_kp.shape) == 1
                    and raw_kp.shape[0] >= 51
                ):
                    if self.debug:
                        print(
                            f"   Using approach 2 (1D array with {raw_kp.shape[0]} elements)"
                        )
                    for idx in range(17):
                        base_idx = idx * 3
                        if base_idx + 2 < raw_kp.shape[0]:
                            x, y, conf = (
                                raw_kp[base_idx],
                                raw_kp[base_idx + 1],
                                raw_kp[base_idx + 2],
                            )
                            if x > 0 and y > 0:
                                all_points.append([x, y])
                                if idx == 1:  # nose keypoint for head tracking
                                    if self.debug:
                                        print(
                                            f"   👃 Nose keypoint: x={x:.1f}, y={y:.1f}, conf={conf:.3f}"
                                        )
                                    if conf > 0.3:
                                        nose_pos = (x, y)
                                        if self.debug:
                                            print("   ✅ Nose accepted!")
                                    else:
                                        if self.debug:
                                            print(
                                                f"   ❌ Nose rejected (confidence {conf:.3f} <= 0.3)"
                                            )
                                if idx in [0, 5, 6, 11, 12] and conf > 0.3:
                                    valid_points.append([x, y])

                # Approach 3: If nothing matched, show debug info
                else:
                    if self.debug:
                        print(
                            f"   ❌ No matching approach for keypoint data: shape={getattr(raw_kp, 'shape', 'no shape')}, type={type(raw_kp)}"
                        )
                        if isinstance(raw_kp, np.ndarray):
                            print(
                                f"   Raw data sample: {raw_kp[:6] if len(raw_kp) > 6 else raw_kp}"
                            )

                # Calculate center and bounding box if we have valid points
                if valid_points:
                    center_x = sum(p[0] for p in valid_points) / len(valid_points)
                    center_y = sum(p[1] for p in valid_points) / len(valid_points)
                    positions.append((center_x, center_y))

                    # Add head position if nose was detected
                    if nose_pos:
                        heads.append(nose_pos)
                        if self.debug:
                            print(f"👃 Head detected at: {nose_pos}")
                    else:
                        if self.debug:
                            print("❌ No nose detected for this person")

                    # Calculate bounding box from all detected points
                    if all_points:
                        xs = [p[0] for p in all_points]
                        ys = [p[1] for p in all_points]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)

                        # Add some padding to the bounding box
                        padding = 20
                        width = max_x - min_x + 2 * padding
                        height = max_y - min_y + 2 * padding
                        bbox_x = min_x - padding
                        bbox_y = min_y - padding

                        bboxes.append((bbox_x, bbox_y, width, height))
                    else:
                        # Fallback: create a default box around the center
                        default_size = 150
                        bboxes.append(
                            (
                                center_x - default_size / 2,
                                center_y - default_size / 2,
                                default_size,
                                default_size,
                            )
                        )

        except Exception as e:
            # if int(time.time() * 2) % 20 == 0:  # Log errors every 10 seconds max
            print(f"Error processing keypoints: {e}")

        if self.debug:
            print(
                f"🔍 Final detection results: {len(positions)} humans, {len(heads)} heads"
            )
        return positions, bboxes, heads

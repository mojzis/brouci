import logging
from unittest.mock import MagicMock, Mock, patch

import cv2
import numpy as np
import pytest

from hemzeni.pose_detector import PoseDetector


class TestPoseDetector:
    @pytest.fixture
    def mock_yolo_model(self):
        """Mock YoloV7Pose model"""
        mock_model = Mock()
        mock_model.return_value = []  # Default empty keypoints
        return mock_model

    @patch('hemzeni.pose_detector.YoloV7Pose')
    def test_pose_detector_initialization(self, mock_yolo_class):
        """Test PoseDetector initialization with default parameters"""
        detector = PoseDetector()
        
        assert detector.detection_scale == 0.2
        assert detector.detection_interval == 1
        assert detector.frame_count == 0
        assert detector.last_human_positions == []
        assert detector.last_human_bboxes == []
        assert detector.last_head_positions == []
        assert detector.last_keypoints.size == 0
        mock_yolo_class.assert_called_once()

    @patch('hemzeni.pose_detector.YoloV7Pose')
    def test_pose_detector_initialization_custom_params(self, mock_yolo_class):
        """Test PoseDetector initialization with custom parameters"""
        detector = PoseDetector(detection_scale=0.5, detection_interval=3)
        
        assert detector.detection_scale == 0.5
        assert detector.detection_interval == 3

    @patch('hemzeni.pose_detector.YoloV7Pose')
    @patch('cv2.resize')
    def test_detect_runs_detection_on_interval(self, mock_resize, mock_yolo_class):
        """Test that detection runs on the specified interval"""
        mock_model = Mock()
        mock_model.return_value = []
        mock_yolo_class.return_value = mock_model
        
        mock_resize.return_value = np.zeros((96, 128, 3), dtype=np.uint8)
        
        detector = PoseDetector(detection_interval=2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First call - should run detection (frame_count = 1)
        detector.detect(frame)
        assert detector.frame_count == 1
        # Should not run detection on frame 1 (1 % 2 != 0)
        
        # Second call - should run detection (frame_count = 2)
        detector.detect(frame)
        assert detector.frame_count == 2
        mock_model.assert_called_once()  # Should be called on frame 2 (2 % 2 == 0)

    @patch('hemzeni.pose_detector.YoloV7Pose')
    @patch('cv2.resize')
    def test_detect_returns_cached_results(self, mock_resize, mock_yolo_class):
        """Test that detection returns cached results when not on interval"""
        mock_model = Mock()
        mock_model.return_value = []
        mock_yolo_class.return_value = mock_model
        
        detector = PoseDetector(detection_interval=2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Set some cached results
        detector.last_human_positions = [(100.0, 100.0)]
        detector.last_human_bboxes = [(80.0, 80.0, 40.0, 40.0)]
        detector.last_head_positions = [(100.0, 90.0)]
        detector.last_keypoints = np.array([[100, 100, 0.9]])
        
        # First call - should return cached results (frame_count = 1, 1 % 2 != 0)
        result = detector.detect(frame)
        human_positions, human_bboxes, head_positions, keypoint_array, raw_keypoints = result
        
        assert human_positions == [(100.0, 100.0)]
        assert human_bboxes == [(80.0, 80.0, 40.0, 40.0)]
        assert head_positions == [(100.0, 90.0)]
        np.testing.assert_array_equal(keypoint_array, np.array([[100, 100, 0.9]]))
        assert raw_keypoints == []

    @patch('hemzeni.pose_detector.YoloV7Pose')
    @patch('cv2.resize')
    def test_detect_scaling(self, mock_resize, mock_yolo_class):
        """Test frame scaling during detection"""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        small_frame = np.zeros((96, 128, 3), dtype=np.uint8)
        mock_resize.return_value = small_frame
        
        # Mock keypoints with raw_keypoint attribute
        mock_keypoint = Mock()
        mock_keypoint.raw_keypoint = np.array([[50, 60, 0.8]])  # Position in small frame
        mock_model.return_value = [mock_keypoint]
        
        detector = PoseDetector(detection_scale=0.2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(frame)
        human_positions, human_bboxes, head_positions, keypoint_array, raw_keypoints = result
        
        # Verify frame was resized correctly
        mock_resize.assert_called_with(frame, (128, 96))  # 640*0.2, 480*0.2
        
        # Verify keypoints were scaled back up (shape should be (1, 1, 3))
        expected_shape = (1, 1, 3)
        assert keypoint_array.shape == expected_shape
        # Check the actual scaled values 
        assert abs(keypoint_array[0, 0, 0] - 250) < 1  # 50/0.2
        assert abs(keypoint_array[0, 0, 1] - 300) < 1  # 60/0.2

    def test_extract_human_data_2d_array_format(self):
        """Test human data extraction from 2D keypoint array format"""
        detector = PoseDetector()
        
        # Create mock keypoint object with 2D array format (17, 3)
        mock_keypoint = Mock()
        keypoint_data = np.zeros((17, 3))
        # Set some key points (nose, shoulders, hips)
        keypoint_data[0] = [100, 50, 0.9]   # nose
        keypoint_data[5] = [80, 100, 0.8]   # left shoulder
        keypoint_data[6] = [120, 100, 0.8]  # right shoulder
        keypoint_data[11] = [90, 200, 0.7]  # left hip
        keypoint_data[12] = [110, 200, 0.7] # right hip
        mock_keypoint.raw_keypoint = keypoint_data
        
        positions, bboxes, heads = detector._extract_human_data([mock_keypoint])
        
        # Should have one person detected
        assert len(positions) == 1
        assert len(bboxes) == 1
        assert len(heads) == 1
        
        # Check center position (average of key points)
        expected_center_x = (100 + 80 + 120 + 90 + 110) / 5
        expected_center_y = (50 + 100 + 100 + 200 + 200) / 5
        assert abs(positions[0][0] - expected_center_x) < 0.1
        assert abs(positions[0][1] - expected_center_y) < 0.1
        
        # Check head position (nose)
        assert heads[0] == (100, 50)
        
        # Check bounding box
        bbox_x, bbox_y, bbox_w, bbox_h = bboxes[0]
        assert bbox_x < 100  # Should include padding
        assert bbox_y < 50
        assert bbox_w > 40   # Should span from 80 to 120 plus padding
        assert bbox_h > 150  # Should span from 50 to 200 plus padding

    def test_extract_human_data_1d_array_format(self):
        """Test human data extraction from 1D keypoint array format (51 elements)"""
        detector = PoseDetector()
        
        # Create mock keypoint object with 1D array format (17 * 3 = 51)
        mock_keypoint = Mock()
        keypoint_data = np.zeros(51)
        # Set some key points (nose, shoulders, hips)
        keypoint_data[0:3] = [100, 50, 0.9]    # nose
        keypoint_data[15:18] = [80, 100, 0.8]  # left shoulder (idx 5)
        keypoint_data[18:21] = [120, 100, 0.8] # right shoulder (idx 6)
        keypoint_data[33:36] = [90, 200, 0.7]  # left hip (idx 11)
        keypoint_data[36:39] = [110, 200, 0.7] # right hip (idx 12)
        mock_keypoint.raw_keypoint = keypoint_data
        
        positions, bboxes, heads = detector._extract_human_data([mock_keypoint])
        
        # Should have one person detected
        assert len(positions) == 1
        assert len(bboxes) == 1
        assert len(heads) == 1
        
        # Check head position (nose)
        assert heads[0] == (100, 50)

    def test_extract_human_data_low_confidence_points(self):
        """Test that low confidence keypoints are filtered out"""
        detector = PoseDetector()
        
        mock_keypoint = Mock()
        keypoint_data = np.zeros((17, 3))
        # Set points with low confidence
        keypoint_data[0] = [100, 50, 0.1]   # nose - low confidence
        keypoint_data[5] = [80, 100, 0.9]   # left shoulder - high confidence
        keypoint_data[6] = [120, 100, 0.2]  # right shoulder - low confidence
        mock_keypoint.raw_keypoint = keypoint_data
        
        positions, bboxes, heads = detector._extract_human_data([mock_keypoint])
        
        # Should still detect person but with limited keypoints
        assert len(positions) == 1
        # Head should not be detected due to low confidence nose
        assert len(heads) == 0

    def test_extract_human_data_no_valid_points(self):
        """Test handling of keypoints with no valid points"""
        detector = PoseDetector()
        
        mock_keypoint = Mock()
        keypoint_data = np.zeros((17, 3))  # All zeros = no valid points
        mock_keypoint.raw_keypoint = keypoint_data
        
        positions, bboxes, heads = detector._extract_human_data([mock_keypoint])
        
        # Should not detect any people
        assert len(positions) == 0
        assert len(bboxes) == 0
        assert len(heads) == 0

    def test_extract_human_data_multiple_people(self):
        """Test extraction with multiple people"""
        detector = PoseDetector()
        
        # Create two mock keypoint objects
        mock_keypoint1 = Mock()
        keypoint_data1 = np.zeros((17, 3))
        keypoint_data1[0] = [100, 50, 0.9]   # nose
        keypoint_data1[5] = [80, 100, 0.8]   # left shoulder
        mock_keypoint1.raw_keypoint = keypoint_data1
        
        mock_keypoint2 = Mock()
        keypoint_data2 = np.zeros((17, 3))
        keypoint_data2[0] = [300, 150, 0.9]  # nose
        keypoint_data2[6] = [320, 200, 0.8]  # right shoulder
        mock_keypoint2.raw_keypoint = keypoint_data2
        
        positions, bboxes, heads = detector._extract_human_data([mock_keypoint1, mock_keypoint2])
        
        # Should detect two people
        assert len(positions) == 2
        assert len(bboxes) == 2
        assert len(heads) == 2

    @patch('hemzeni.pose_detector.time.time')
    def test_extract_human_data_error_handling(self, mock_time):
        """Test error handling in human data extraction"""
        mock_time.return_value = 0  # Control time for logging
        
        detector = PoseDetector()
        
        # Create invalid keypoint data that will cause an exception
        mock_keypoint = Mock()
        mock_keypoint.raw_keypoint = "invalid_data"  # Not a numpy array
        
        with patch('builtins.print') as mock_print:
            positions, bboxes, heads = detector._extract_human_data([mock_keypoint])
        
        # Should handle error gracefully and return empty results
        assert positions == []
        assert bboxes == []
        assert heads == []
        mock_print.assert_called()  # Should log the error

    @patch('hemzeni.pose_detector.YoloV7Pose')
    @patch('cv2.resize')
    def test_detect_full_pipeline(self, mock_resize, mock_yolo_class):
        """Test complete detection pipeline with scaling"""
        mock_model = Mock()
        mock_yolo_class.return_value = mock_model
        
        # Mock keypoint with valid data
        mock_keypoint = Mock()
        keypoint_data = np.zeros((17, 3))
        keypoint_data[0] = [25, 15, 0.9]    # nose (in scaled frame)
        keypoint_data[5] = [20, 25, 0.8]    # left shoulder
        keypoint_data[6] = [30, 25, 0.8]    # right shoulder
        mock_keypoint.raw_keypoint = keypoint_data
        mock_model.return_value = [mock_keypoint]
        
        mock_resize.return_value = np.zeros((96, 128, 3), dtype=np.uint8)
        
        detector = PoseDetector(detection_scale=0.2)  # 5x scaling
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        result = detector.detect(frame)
        human_positions, human_bboxes, head_positions, keypoint_array, raw_keypoints = result
        
        # Check scaled results
        assert len(human_positions) == 1
        assert len(human_bboxes) == 1
        assert len(head_positions) == 1
        
        # Positions should be scaled up by 5x (1/0.2)
        expected_center_x = (25 + 20 + 30) / 3 * 5  # Average of x coordinates * 5
        expected_center_y = (15 + 25 + 25) / 3 * 5  # Average of y coordinates * 5
        
        assert abs(human_positions[0][0] - expected_center_x) < 1
        assert abs(human_positions[0][1] - expected_center_y) < 1
        
        # Head position should be scaled
        assert head_positions[0] == (125, 75)  # 25*5, 15*5
        
        # Keypoint array should be scaled - check shape and key values
        expected_shape = (1, 17, 3)
        assert keypoint_array.shape == expected_shape
        
        # Check that specific keypoints were scaled correctly
        assert abs(keypoint_array[0, 0, 0] - 125) < 1  # nose x: 25*5
        assert abs(keypoint_array[0, 0, 1] - 75) < 1   # nose y: 15*5
        assert abs(keypoint_array[0, 5, 0] - 100) < 1  # left shoulder x: 20*5
        assert abs(keypoint_array[0, 5, 1] - 125) < 1  # left shoulder y: 25*5

    @patch('hemzeni.pose_detector.YoloV7Pose')
    def test_caching_behavior(self, mock_yolo_class):
        """Test that results are properly cached and retrieved"""
        mock_model = Mock()
        mock_model.return_value = []
        mock_yolo_class.return_value = mock_model
        
        detector = PoseDetector(detection_interval=2)
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # First detection (frame_count = 1, should use cache - empty)
        detector.detect(frame)
        
        # Manually set cached results
        detector.last_human_positions = [(200.0, 200.0)]
        detector.last_human_bboxes = [(180.0, 180.0, 40.0, 40.0)]
        detector.last_head_positions = [(200.0, 190.0)]
        detector.last_keypoints = np.array([[200, 200, 0.9]])
        
        # Second detection (frame_count = 2, should run new detection)
        with patch('cv2.resize') as mock_resize:
            mock_resize.return_value = np.zeros((96, 128, 3), dtype=np.uint8)
            result2 = detector.detect(frame)
        
        # Third detection (frame_count = 3, should use cache from frame 2)
        result3 = detector.detect(frame)
        
        # Result3 should be same as result2 (both using updated cache)
        assert result2[0] == result3[0]  # human_positions
        assert result2[1] == result3[1]  # human_bboxes


class TestPoseDetectorLogging:
    """Test logging configuration and behavior"""
    
    def test_logging_configuration(self):
        """Test that logging is properly configured to suppress model output"""
        # Check that specific loggers are set to appropriate levels
        assert logging.getLogger("models.common").level == logging.ERROR
        assert logging.getLogger("models.experimental").level == logging.ERROR
        assert logging.getLogger("models.yolo").level == logging.ERROR
        assert logging.getLogger("aipose").level == logging.WARNING

    @patch('hemzeni.pose_detector.YoloV7Pose')
    @patch('hemzeni.pose_detector.time.time')
    def test_periodic_logging(self, mock_time, mock_yolo_class):
        """Test that keypoint detection logging happens periodically"""
        mock_model = Mock()
        mock_keypoint = Mock()
        mock_keypoint.raw_keypoint = np.zeros((17, 3))
        mock_model.return_value = [mock_keypoint]
        mock_yolo_class.return_value = mock_model
        
        # Mock time to trigger logging condition
        mock_time.return_value = 0  # int(0 * 2) % 10 == 0
        
        detector = PoseDetector()
        
        with patch('builtins.print') as mock_print:
            with patch('cv2.resize') as mock_resize:
                mock_resize.return_value = np.zeros((96, 128, 3), dtype=np.uint8)
                detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Should log the number of keypoints detected
        mock_print.assert_called_with("Number of keypoints detected: 1")

    # New tests for grace period and head detection logic
    @patch('hemzeni.pose_detector.YoloV7Pose')
    @patch('cv2.resize') # Mock resize as it's called in detect
    def test_grace_period_logic(self, mock_resize, mock_yolo_class):
        """Test the grace period logic for detections."""
        mock_model_instance = MagicMock()
        mock_yolo_class.return_value = mock_model_instance
        mock_resize.return_value = np.zeros((96, 128, 3), dtype=np.uint8) # Dummy small frame

        detector = PoseDetector(detection_interval=1, grace_period_frames=2)
        # Set a distinct detection_scale to test if it's preserved in cached results
        detector.detection_scale = 0.5 
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Mock keypoint object structure
        # raw_keypoint should be shape (N, 3) for Approach 1 in _extract_human_data
        # For simplicity, using a single keypoint that defines a person
        kp_data1 = np.array([[20, 20, 0.9], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [22,22,0.8]]) # simplified for position calculation
        mock_kp_obj1 = MagicMock()
        mock_kp_obj1.raw_keypoint = kp_data1
        
        kp_data2 = np.array([[40, 40, 0.9], [0,0,0], [0,0,0], [0,0,0], [0,0,0], [42,42,0.8]])
        mock_kp_obj2 = MagicMock()
        mock_kp_obj2.raw_keypoint = kp_data2


        # Frame 1: Valid detection
        mock_model_instance.return_value = [mock_kp_obj1]
        hp1, bb1, hdp1, kp_arr1, raw_kp1 = detector.detect(frame)
        
        assert len(hp1) > 0, "Frame 1 should have human positions"
        # raw_kp1 is a list of YoloV7Pose objects, check its content
        assert len(raw_kp1) == 1 and raw_kp1[0] == mock_kp_obj1, "Frame 1 should return raw keypoints"
        assert detector.frames_since_last_detection == 0
        assert detector.last_valid_detection_results is not None
        # Check that the scaled keypoints are stored in last_valid_detection_results
        # Example: first point x: 20 / 0.5 = 40
        assert detector.last_valid_detection_results[3][0,0,0] == 20 / 0.5


        # Frame 2: No detection (grace period active)
        mock_model_instance.return_value = [] # No keypoints detected
        hp2, bb2, hdp2, kp_arr2, raw_kp2 = detector.detect(frame)

        assert hp2 == hp1, "Frame 2 should return cached human positions from Frame 1"
        assert bb2 == bb1, "Frame 2 should return cached bboxes from Frame 1"
        assert hdp2 == hdp1, "Frame 2 should return cached head positions from Frame 1"
        np.testing.assert_array_equal(kp_arr2, kp_arr1, err_msg="Frame 2 should return cached keypoint array from Frame 1")
        assert raw_kp2 == [], "Frame 2 should return empty list for current raw keypoints"
        assert detector.frames_since_last_detection == 1
        
        # Frame 3: No detection (grace period active, last frame of grace)
        mock_model_instance.return_value = []
        hp3, bb3, hdp3, kp_arr3, raw_kp3 = detector.detect(frame)

        assert hp3 == hp1, "Frame 3 should return cached human positions from Frame 1"
        assert bb3 == bb1, "Frame 3 should return cached bboxes from Frame 1"
        assert hdp3 == hdp1, "Frame 3 should return cached head positions from Frame 1"
        np.testing.assert_array_equal(kp_arr3, kp_arr1, err_msg="Frame 3 should return cached keypoint array from Frame 1")
        assert raw_kp3 == [], "Frame 3 should return empty list for current raw keypoints"
        assert detector.frames_since_last_detection == 2

        # Frame 4: No detection (grace period expired)
        mock_model_instance.return_value = []
        hp4, bb4, hdp4, kp_arr4, raw_kp4 = detector.detect(frame)
        
        assert hp4 == [], "Frame 4 should return empty human positions"
        assert bb4 == [], "Frame 4 should return empty bboxes"
        assert hdp4 == [], "Frame 4 should return empty head positions"
        assert kp_arr4.size == 0, "Frame 4 should return empty keypoint array"
        assert raw_kp4 == [], "Frame 4 should return empty list for current raw keypoints"
        assert detector.frames_since_last_detection == 3 # Incremented beyond grace_period_frames
        assert detector.last_valid_detection_results is None # Should be cleared

        # Frame 5: New valid detection
        mock_model_instance.return_value = [mock_kp_obj2]
        hp5, bb5, hdp5, kp_arr5, raw_kp5 = detector.detect(frame)

        assert len(hp5) > 0, "Frame 5 should have new human positions"
        assert hp5 != hp1, "Frame 5 positions should be different from Frame 1"
        assert len(raw_kp5) == 1 and raw_kp5[0] == mock_kp_obj2, "Frame 5 should return new raw keypoints"
        assert detector.frames_since_last_detection == 0
        assert detector.last_valid_detection_results is not None
        assert detector.last_valid_detection_results[0] == hp5 
        # Check scaled keypoint from new detection
        assert detector.last_valid_detection_results[3][0,0,0] == 40 / 0.5


    def test_head_detection_ensures_human_position(self):
        """Test that a detected head (even if alone) results in a human position and bbox."""
        detector = PoseDetector(detection_scale=1.0, debug=False) # Use scale 1.0 for simplicity

        mock_kp_obj_head_only = MagicMock()
        keypoints_data_head_only = np.zeros((17, 3)) 
        keypoints_data_head_only[0] = [150, 160, 0.9] # Nose (idx 0 for YoloV7Pose)
        mock_kp_obj_head_only.raw_keypoint = keypoints_data_head_only
        
        # Call _extract_human_data directly as it contains the core logic
        # This method expects a list of keypoint objects from the model
        positions, bboxes, heads = detector._extract_human_data([mock_kp_obj_head_only])

        assert len(heads) == 1, "Should detect one head"
        assert heads[0] == (150, 160), "Head position should match nose keypoint"
        
        assert len(positions) == 1, "Should create one human position from the detected head"
        # The position calculation for a single nose point will be just the nose point itself
        # if only nose is in valid_points.
        # valid_points includes idx 0 (nose) if conf > 0.3
        # center_x = sum(p[0] for p in valid_points) / len(valid_points) -> 150/1 = 150
        assert positions[0] == (150, 160), "Human position should be based on the head's position"
        
        assert len(bboxes) == 1, "Should create one bounding box for the head-derived human"
        # Check default bbox around the head. From code: default_head_bbox_width = 50
        expected_bbox_x = 150 - 50 / 2
        expected_bbox_y = 160 - 50 / 2
        assert bboxes[0] == (expected_bbox_x, expected_bbox_y, 50, 50), "Bounding box should be the default for a head"

        # Scenario 2: Full person detected, and a separate head far away
        mock_person_kp_obj = MagicMock()
        person_kp_data = np.zeros((17,3))
        person_kp_data[0] = [100,100,0.9] # Nose
        person_kp_data[5] = [90,120,0.8]  # L Shoulder
        person_kp_data[6] = [110,120,0.8] # R Shoulder
        person_kp_data[11] = [95,150,0.8] # L Hip
        person_kp_data[12] = [105,150,0.8]# R Hip
        mock_person_kp_obj.raw_keypoint = person_kp_data
        # Expected center for person1: nose, shoulders, hips
        # x_coords = [100, 90, 110, 95, 105] -> sum = 500, avg = 100
        # y_coords = [100, 120, 120, 150, 150] -> sum = 640, avg = 128
        # Person1 center approx (100, 128)

        mock_far_head_kp_obj = MagicMock()
        far_head_kp_data = np.zeros((17,3))
        far_head_kp_data[0] = [300,300,0.9] # Far Nose
        mock_far_head_kp_obj.raw_keypoint = far_head_kp_data
        
        positions, bboxes, heads = detector._extract_human_data([mock_person_kp_obj, mock_far_head_kp_obj])
        
        assert len(heads) == 2 # Both noses detected
        assert len(positions) == 2 # Person + far head
        assert len(bboxes) == 2

        found_far_head_person = any(pos == (300,300) for pos in positions)
        assert found_far_head_person, "Far head should be added as a distinct human position"
        
        # Scenario 3: Full person detected, and another head detection close to this person
        # This "other head" should NOT create a new person due to proximity.
        mock_close_head_only_kp_obj = MagicMock()
        close_head_only_kp_data = np.zeros((17,3))
        # Person1 center approx (100, 128). min_distance_sq_threshold = 30*30 = 900
        # A head at (110, 130) would be: dx=10, dy=2. dist_sq = 100+4 = 104 < 900. So, too close.
        close_head_only_kp_data[0] = [110, 130, 0.9] 
        mock_close_head_only_kp_obj.raw_keypoint = close_head_only_kp_data

        positions, bboxes, heads = detector._extract_human_data([mock_person_kp_obj, mock_close_head_only_kp_obj])

        assert len(heads) == 2 # Nose from person1, and the new close head detection
        assert len(positions) == 1 # Should only be one person, as the second head is too close
        assert len(bboxes) == 1
        
        # Verify the single position is from mock_person_kp_obj (approx (100,128))
        main_person_pos = positions[0]
        assert abs(main_person_pos[0] - 100) < 1e-6 
        assert abs(main_person_pos[1] - 128) < 1e-6
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
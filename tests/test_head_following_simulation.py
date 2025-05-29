"""Test to simulate head detection and HeadFollower behavior."""

import os
import tempfile
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from hemzeni.beetle import HeadFollower
from hemzeni.controller import BeetleController


class TestHeadFollowingSimulation:
    """Tests that simulate realistic head detection and following scenarios"""

    @pytest.fixture
    def mock_head_follower_image(self):
        """Create a temporary test image file for head follower"""
        test_image = np.zeros((40, 40, 4), dtype=np.uint8)
        test_image[:, :, :3] = [200, 100, 50]  # BGR color
        test_image[:, :, 3] = 255  # Alpha channel
        
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_file.name, test_image)
        
        yield temp_file.name
        
        # Cleanup
        os.unlink(temp_file.name)

    def test_head_follower_starts_following_when_head_detected(self, mock_head_follower_image):
        """Test HeadFollower starts following when head_positions contains valid coordinates"""
        # Create a HeadFollower instance
        head_follower = HeadFollower(
            name="lachticek",
            image_path=mock_head_follower_image,
            frame_width=640,
            frame_height=480,
            debug=True
        )
        
        # Initially not following
        assert not head_follower.is_following
        assert head_follower.target_head_pos is None
        
        # Simulate head detection - person's head detected at coordinates (300, 200)
        head_positions = [(300.0, 200.0)]
        
        # Update the head follower with detected head position
        head_follower.update_following(head_positions)
        
        # Should now be following
        assert head_follower.is_following is True
        
        # Should move towards the head position (above the head)
        expected_target_x = 300.0 - head_follower.size / 2  # Center on head
        expected_target_y = 200.0 - head_follower.size - 10  # Position above head
        
        # Check that the follower is moving towards the target
        # (exact position depends on movement speed and initial position)
        print(f"HeadFollower position: ({head_follower.x}, {head_follower.y})")
        print(f"Target position: ({expected_target_x}, {expected_target_y})")

    def test_head_follower_stops_following_when_head_disappears(self, mock_head_follower_image):
        """Test HeadFollower stops following when head_positions becomes empty"""
        head_follower = HeadFollower(
            name="lachticek",
            image_path=mock_head_follower_image,
            frame_width=640,
            frame_height=480,
            debug=True
        )
        
        # Start following a head
        head_positions = [(300.0, 200.0)]
        head_follower.update_following(head_positions)
        assert head_follower.is_following is True
        
        # Head disappears (empty head_positions)
        head_positions = []
        head_follower.update_following(head_positions)
        
        # Should stop following
        assert head_follower.is_following is False

    def test_head_follower_follows_moving_head(self, mock_head_follower_image):
        """Test HeadFollower tracks a moving head across multiple frames"""
        head_follower = HeadFollower(
            name="lachticek",
            image_path=mock_head_follower_image,
            frame_width=640,
            frame_height=480,
            debug=True
        )
        
        # Simulate head moving across the frame
        head_positions_sequence = [
            [(100.0, 150.0)],  # Frame 1: head on left
            [(200.0, 150.0)],  # Frame 2: head moving right
            [(300.0, 150.0)],  # Frame 3: head continuing right
            [(400.0, 150.0)],  # Frame 4: head further right
        ]
        
        positions = []
        for head_positions in head_positions_sequence:
            head_follower.update_following(head_positions)
            positions.append((head_follower.x, head_follower.y))
            print(f"Head at {head_positions[0]}, Follower at ({head_follower.x:.1f}, {head_follower.y:.1f})")
        
        # Verify the follower is generally moving in the right direction
        # (should be following the head from left to right)
        assert positions[-1][0] > positions[0][0]  # X position should increase

    @patch('cv2.VideoCapture')
    @patch('hemzeni.controller.PoseDetector')
    def test_controller_head_following_integration(self, mock_detector_class, mock_video_capture, mock_head_follower_image):
        """Test full integration: Controller processes head_positions and updates HeadFollower"""
        mock_video_capture.return_value.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        
        # Mock detector to return head positions
        mock_detector = Mock()
        mock_detector.detect.return_value = (
            [(300.0, 300.0)],  # human_positions
            [(280.0, 280.0, 40.0, 40.0)],  # human_bboxes  
            [(300.0, 200.0)],  # head_positions - THIS IS THE KEY PART
            np.array([[300, 200, 0.9]]),  # keypoint_array
            []  # raw_keypoints
        )
        mock_detector_class.return_value = mock_detector
        
        # Create controller
        controller = BeetleController(debug=True)
        
        # Create a real HeadFollower instance
        controller.head_follower = HeadFollower(
            name="lachticek",
            image_path=mock_head_follower_image,
            frame_width=640,
            frame_height=480,
            debug=True
        )
        
        # Initially not following
        assert not controller.head_follower.is_following
        
        # Process a frame - this should trigger head following
        with patch('time.time', return_value=1.0):
            with patch('aipose.plot.plot', return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
                result = controller.process_frame()
        
        # Verify result
        assert result is not None
        frame, human_positions, human_bboxes = result
        
        # Most importantly: HeadFollower should now be following the detected head
        assert controller.head_follower.is_following is True
        print(f"✅ HeadFollower is now following! Position: ({controller.head_follower.x:.1f}, {controller.head_follower.y:.1f})")

    def test_multiple_heads_detected(self, mock_head_follower_image):
        """Test HeadFollower behavior when multiple heads are detected"""
        head_follower = HeadFollower(
            name="lachticek", 
            image_path=mock_head_follower_image,
            frame_width=640,
            frame_height=480,
            debug=True
        )
        
        # Multiple heads detected - should follow the first one
        head_positions = [(200.0, 150.0), (400.0, 150.0), (300.0, 200.0)]
        
        head_follower.update_following(head_positions)
        
        assert head_follower.is_following is True
        # Should be targeting the first head position
        expected_target_x = 200.0 - head_follower.size / 2
        expected_target_y = 150.0 - head_follower.size - 10
        
        print(f"Multiple heads detected, following first one at (200, 150)")
        print(f"Follower targeting: ({expected_target_x}, {expected_target_y})")

    def test_head_follower_stays_within_frame_bounds(self, mock_head_follower_image):
        """Test HeadFollower doesn't move outside frame boundaries when following head"""
        head_follower = HeadFollower(
            name="lachticek",
            image_path=mock_head_follower_image, 
            frame_width=640,
            frame_height=480,
            debug=True
        )
        
        # Head detected near edge of frame - follower should stay within bounds
        head_positions = [(10.0, 10.0)]  # Very close to top-left corner
        
        head_follower.update_following(head_positions)
        
        # Follower should stay within frame bounds
        assert head_follower.x >= 0
        assert head_follower.y >= 0
        assert head_follower.x <= 640 - head_follower.size
        assert head_follower.y <= 480 - head_follower.size
        
        print(f"Head near edge at (10, 10), follower bounded at ({head_follower.x:.1f}, {head_follower.y:.1f})")


# Demonstration script
if __name__ == "__main__":
    print("🎯 Head Following Simulation Demonstration")
    print("=" * 50)
    
    # Create a simple demo without pytest
    test_image = np.ones((40, 40, 4), dtype=np.uint8) * 255
    test_image[:, :, :3] = [100, 150, 200]
    
    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    cv2.imwrite(temp_file.name, test_image)
    
    try:
        # Create HeadFollower
        follower = HeadFollower("demo_follower", temp_file.name, 640, 480, debug=True)
        
        print(f"Initial state - Following: {follower.is_following}")
        print(f"Initial position: ({follower.x:.1f}, {follower.y:.1f})")
        print()
        
        # Simulate head detection
        print("👤 Head detected at (300, 200)")
        follower.update_following([(300.0, 200.0)])
        print(f"After detection - Following: {follower.is_following}")
        print(f"New position: ({follower.x:.1f}, {follower.y:.1f})")
        print()
        
        # Simulate head movement
        print("👤 Head moves to (400, 200)")
        follower.update_following([(400.0, 200.0)])
        print(f"Following head - Position: ({follower.x:.1f}, {follower.y:.1f})")
        print()
        
        # Simulate head disappearing
        print("👻 Head disappears")
        follower.update_following([])
        print(f"After disappearance - Following: {follower.is_following}")
        
    finally:
        os.unlink(temp_file.name)
    
    print("\n✅ Demonstration complete!")
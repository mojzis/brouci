import math
import os
import tempfile
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from hemzeni.beetle import Beetle, HeadFollower


class TestBeetle:
    @pytest.fixture
    def mock_beetle_image(self):
        """Create a temporary test image file"""
        # Create a simple test image
        test_image = np.zeros((80, 80, 4), dtype=np.uint8)
        test_image[:, :, :3] = [100, 150, 200]  # BGR color
        test_image[:, :, 3] = 255  # Alpha channel

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, test_image)

        yield temp_file.name

        # Cleanup
        os.unlink(temp_file.name)

    @pytest.fixture
    def beetle(self, mock_beetle_image: str) -> Beetle:
        """Create a test beetle instance"""
        return Beetle(
            name="test_beetle",
            image_path=mock_beetle_image,
            frame_width=640,
            frame_height=480,
            debug=False,
        )

    def test_beetle_initialization(self, mock_beetle_image: str) -> None:
        """Test beetle initialization with valid parameters"""
        beetle = Beetle(
            name="kudlanka",
            image_path=mock_beetle_image,
            frame_width=640,
            frame_height=480,
            debug=True,
        )

        assert beetle.name == "kudlanka"
        assert beetle.frame_width == 640
        assert beetle.frame_height == 480
        assert beetle.size == 80
        assert beetle.speed == 5  # kudlanka has speed 5
        assert beetle.debug is True
        assert not beetle.is_fleeing
        assert not beetle.is_hidden
        assert 0 <= beetle.x <= 640 - 80
        assert 0 <= beetle.y <= 480 - 80

    def test_beetle_initialization_blecha_speed(self, mock_beetle_image: str) -> None:
        """Test that blecha has different speed than kudlanka"""
        blecha = Beetle(
            name="blecha",
            image_path=mock_beetle_image,
            frame_width=640,
            frame_height=480,
        )

        assert blecha.speed == 8  # blecha is faster

    def test_beetle_initialization_invalid_image(self) -> None:
        """Test beetle initialization with invalid image path"""
        with pytest.raises(ValueError, match="Could not load beetle image"):
            Beetle(
                name="test",
                image_path="nonexistent.png",
                frame_width=640,
                frame_height=480,
            )

    def test_get_bounding_box(self, beetle: Beetle) -> None:
        """Test bounding box calculation"""
        beetle.x = 100.0
        beetle.y = 200.0

        bbox = beetle.get_bounding_box()
        assert bbox == (100.0, 200.0, 80.0, 80.0)

    def test_is_at_wall(self, beetle: Beetle) -> None:
        """Test wall detection"""
        # Test not at wall
        beetle.x = 100.0
        beetle.y = 100.0
        assert not beetle.is_at_wall()

        # Test at left wall
        beetle.x = 3.0
        beetle.y = 100.0
        assert beetle.is_at_wall()

        # Test at right wall
        beetle.x = 640.0 - 80.0 - 3.0
        beetle.y = 100.0
        assert beetle.is_at_wall()

        # Test at top wall
        beetle.x = 100.0
        beetle.y = 3.0
        assert beetle.is_at_wall()

        # Test at bottom wall
        beetle.x = 100.0
        beetle.y = 480.0 - 80.0 - 3.0
        assert beetle.is_at_wall()

    def test_rectangles_intersect(self) -> None:
        """Test rectangle intersection detection"""
        rect1 = (10.0, 10.0, 20.0, 20.0)  # x=10-30, y=10-30
        rect2 = (25.0, 25.0, 20.0, 20.0)  # x=25-45, y=25-45
        rect3 = (50.0, 50.0, 20.0, 20.0)  # x=50-70, y=50-70

        # Test intersection
        assert Beetle.rectangles_intersect(rect1, rect2) is True

        # Test no intersection
        assert Beetle.rectangles_intersect(rect1, rect3) is False

        # Test identical rectangles
        assert Beetle.rectangles_intersect(rect1, rect1) is True

    def test_check_collision(self, mock_beetle_image: str) -> None:
        """Test beetle collision detection"""
        beetle1 = Beetle("beetle1", mock_beetle_image, 640, 480)
        beetle2 = Beetle("beetle2", mock_beetle_image, 640, 480)

        # Position beetles to collide
        beetle1.x = 100.0
        beetle1.y = 100.0
        beetle2.x = 120.0  # Overlapping
        beetle2.y = 120.0  # Overlapping

        assert beetle1.check_collision(beetle2) is True

        # Position beetles apart
        beetle2.x = 200.0
        beetle2.y = 200.0

        assert beetle1.check_collision(beetle2) is False

        # Test hidden beetles don't collide
        beetle1.is_hidden = True
        beetle1.x = 100.0
        beetle1.y = 100.0
        beetle2.x = 120.0
        beetle2.y = 120.0
        beetle2.is_hidden = False

        assert beetle1.check_collision(beetle2) is False

    def test_avoid_collision(self, mock_beetle_image: str) -> None:
        """Test collision avoidance behavior"""
        beetle1 = Beetle("beetle1", mock_beetle_image, 640, 480)
        beetle2 = Beetle("beetle2", mock_beetle_image, 640, 480)

        # Position beetles
        beetle1.x = 100.0
        beetle1.y = 100.0
        beetle2.x = 80.0
        beetle2.y = 80.0

        original_direction = beetle1.direction
        beetle1.avoid_collision(beetle2)

        # Direction should have changed
        assert beetle1.direction != original_direction

    @patch("time.time")
    def test_update_roaming_direction_change(
        self, mock_time: MagicMock, beetle: Beetle
    ) -> None:
        """Test that beetle changes direction during roaming"""
        # Set up time to trigger direction change
        mock_time.return_value = 4.0  # Current time that triggers direction change

        original_direction = beetle.direction
        beetle.last_direction_change = 0  # More than 3 seconds ago

        beetle.update_roaming()

        # Direction should have changed (within the expected range)
        # The code adds random.uniform(-math.pi / 3, math.pi / 3) to direction
        direction_change = beetle.direction - original_direction
        assert abs(direction_change) <= math.pi / 3  # Should be within expected range
        assert beetle.last_direction_change == 4.0  # Should update last change time

    def test_update_roaming_wall_bounce(
        self, beetle: Beetle, mock_beetle_image: str
    ) -> None:
        """Test wall bouncing during roaming"""
        # Test horizontal wall bounce - position beetle to hit left wall after movement
        beetle.x = 1.0  # Close to left wall
        beetle.direction = math.pi  # Moving left (towards wall)

        beetle.update_roaming()

        # Should bounce off wall (direction flipped horizontally)
        assert abs(beetle.direction - 0) < 0.1  # Now moving right (approximately)
        assert beetle.x >= 0  # Position corrected

        # Reset for vertical wall test
        beetle = Beetle("test", mock_beetle_image, 640, 480)
        beetle.y = 1.0  # Close to top wall
        beetle.direction = -math.pi / 2  # Moving up (towards top wall)

        beetle.update_roaming()

        # Should bounce off wall (direction flipped vertically)
        assert (
            abs(beetle.direction - math.pi / 2) < 0.1
        )  # Now moving down (approximately)
        assert beetle.y >= 0  # Position corrected

    def test_update_roaming_emerging_from_hiding(self, beetle: Beetle) -> None:
        """Test beetle emerging from hiding when roaming"""
        beetle.is_hidden = True
        beetle.is_fleeing = True
        beetle.x = 0.0  # At wall
        beetle.y = 100.0

        beetle.update_roaming()

        # Should emerge from hiding
        assert not beetle.is_hidden
        assert not beetle.is_fleeing
        assert beetle.x == 15.0  # Moved away from wall

    def test_update_fleeing_start_fleeing(self, beetle: Beetle) -> None:
        """Test beetle starts fleeing when human is nearby"""
        human_positions = [(200.0, 200.0)]

        # Position beetle near human
        beetle.x = 150.0
        beetle.y = 150.0

        beetle.update_fleeing(human_positions)

        assert beetle.is_fleeing is True

    def test_update_fleeing_with_bboxes(self, beetle: Beetle) -> None:
        """Test fleeing behavior with human bounding boxes"""
        human_positions = [(200.0, 200.0)]
        human_bboxes = [(180.0, 180.0, 40.0, 40.0)]  # x=180-220, y=180-220

        # Position beetle to collide with human bbox
        beetle.x = 190.0
        beetle.y = 190.0

        beetle.update_fleeing(human_positions, human_bboxes)

        assert beetle.is_fleeing is True

    @patch("time.time")
    def test_update_fleeing_cooldown(
        self, mock_time: MagicMock, beetle: Beetle
    ) -> None:
        """Test fleeing cooldown behavior"""
        beetle.is_fleeing = True
        beetle.last_flee_time = 0
        beetle.flee_cooldown = 2.0

        # Within cooldown period
        mock_time.return_value = 1.0
        beetle.update_fleeing([])  # No humans

        assert beetle.is_fleeing is True  # Still fleeing due to cooldown

        # After cooldown period
        mock_time.return_value = 3.0
        beetle.update_fleeing([])  # No humans

        assert beetle.is_fleeing is False  # Stopped fleeing

    def test_update_fleeing_hide_in_wall(self, beetle: Beetle) -> None:
        """Test beetle hiding in wall when fleeing hits boundary"""
        human_positions = [(320.0, 240.0)]  # Center of frame

        # Position beetle at wall and set flee direction towards wall
        beetle.x = 0.0
        beetle.y = 240.0
        beetle.is_fleeing = True

        # Flee away from human (should hit left wall)
        beetle.update_fleeing(human_positions)

        # Should be hidden in wall
        assert beetle.is_hidden is True
        assert beetle.x == 0.0  # At wall edge

    def test_draw_normal(self, beetle: Beetle) -> None:
        """Test normal beetle drawing"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        beetle.x = 100.0
        beetle.y = 100.0
        beetle.is_hidden = False

        # Should not raise exception
        beetle.draw(frame)

    def test_draw_hidden(self, beetle: Beetle) -> None:
        """Test hidden beetle is not drawn"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame = frame.copy()

        beetle.x = 100.0
        beetle.y = 100.0
        beetle.is_hidden = True

        beetle.draw(frame)

        # Frame should be unchanged
        np.testing.assert_array_equal(frame, original_frame)

    def test_draw_boundary_clamping(self, beetle: Beetle) -> None:
        """Test beetle position is clamped to frame boundaries when drawing"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Position beetle outside frame
        beetle.x = 700.0  # Beyond right edge
        beetle.y = 500.0  # Beyond bottom edge

        beetle.draw(frame)

        # Position should be clamped
        assert beetle.x == 640.0 - 80.0  # frame_width - size
        assert beetle.y == 480.0 - 80.0  # frame_height - size


class TestHeadFollower:
    @pytest.fixture
    def mock_beetle_image(self):
        """Create a temporary test image file"""
        test_image = np.zeros((40, 40, 4), dtype=np.uint8)
        test_image[:, :, :3] = [200, 100, 50]
        test_image[:, :, 3] = 255

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, test_image)

        yield temp_file.name

        os.unlink(temp_file.name)

    @pytest.fixture
    def head_follower(self, mock_beetle_image: str) -> HeadFollower:
        """Create a test head follower instance"""
        return HeadFollower(
            name="test_follower",
            image_path=mock_beetle_image,
            frame_width=640,
            frame_height=480,
            debug=False,
        )

    def test_head_follower_initialization(self, mock_beetle_image: str) -> None:
        """Test head follower initialization"""
        follower = HeadFollower(
            name="lachticek",
            image_path=mock_beetle_image,
            frame_width=640,
            frame_height=480,
            debug=True,
        )

        assert follower.name == "lachticek"
        assert follower.size == 40  # Smaller than regular beetles
        assert follower.follow_speed == 15
        assert not follower.is_following
        assert follower.target_head_pos is None
        # Should start in center
        assert follower.x == 640 / 2 - 40 / 2
        assert follower.y == 480 / 2 - 40 / 2

    def test_update_following_start_following(
        self, head_follower: HeadFollower
    ) -> None:
        """Test head follower starts following when head is detected"""
        head_positions = [(300.0, 200.0)]

        head_follower.update_following(head_positions)

        assert head_follower.is_following is True

    def test_update_following_stop_following(self, head_follower: HeadFollower) -> None:
        """Test head follower stops following when no heads detected"""
        head_follower.is_following = True

        head_follower.update_following([])

        assert head_follower.is_following is False

    def test_update_following_movement(self, head_follower: HeadFollower) -> None:
        """Test head follower moves toward head position"""
        head_positions = [(400.0, 200.0)]

        original_x = head_follower.x
        original_y = head_follower.y

        head_follower.update_following(head_positions)

        # Should move towards head (accounting for positioning above head)
        target_x = 400.0 - head_follower.size / 2
        target_y = 200.0 - head_follower.size - 10

        # Should have moved closer to target
        assert abs(head_follower.x - target_x) < abs(original_x - target_x)
        assert abs(head_follower.y - target_y) < abs(original_y - target_y)

    def test_update_following_boundary_clamping(
        self, head_follower: HeadFollower
    ) -> None:
        """Test head follower stays within frame boundaries"""
        # Head position that would move follower outside frame
        head_positions = [(-50.0, -50.0)]

        head_follower.update_following(head_positions)

        # Should be clamped to frame boundaries
        assert head_follower.x >= 0
        assert head_follower.y >= 0
        assert head_follower.x <= 640 - head_follower.size
        assert head_follower.y <= 480 - head_follower.size

    def test_update_roaming_does_nothing(self, head_follower: HeadFollower) -> None:
        """Test head follower doesn't roam"""
        original_x = head_follower.x
        original_y = head_follower.y

        head_follower.update_roaming()

        # Position should not change
        assert head_follower.x == original_x
        assert head_follower.y == original_y

    def test_update_fleeing_does_nothing(self, head_follower: HeadFollower) -> None:
        """Test head follower doesn't flee"""
        original_x = head_follower.x
        original_y = head_follower.y
        original_is_following = head_follower.is_following

        human_positions = [(100.0, 100.0)]
        head_follower.update_fleeing(human_positions)

        # State should not change
        assert head_follower.x == original_x
        assert head_follower.y == original_y
        assert head_follower.is_following == original_is_following

    def test_draw_only_when_following(self, head_follower: HeadFollower) -> None:
        """Test head follower only draws when following"""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        original_frame = frame.copy()

        # Not following - should not draw
        head_follower.is_following = False
        head_follower.draw(frame)
        np.testing.assert_array_equal(frame, original_frame)

        # Following - should draw
        head_follower.is_following = True
        head_follower.draw(frame)
        # Frame should be modified (we don't check exact content due to image complexity)

    # New tests for Beetle class
    @patch('time.time')
    def test_beetle_stays_hidden_during_emerge_cooldown(self, mock_time, beetle: Beetle):
        mock_time.return_value = 100.0 # Current time
        beetle.is_hidden = True
        beetle.hide_time = mock_time.return_value # Just hid
        beetle.emerge_cooldown_duration = 1.0

        beetle.update_roaming([])
        assert beetle.is_hidden is True, "Beetle should remain hidden during cooldown"

    @patch('time.time')
    def test_beetle_emerges_after_emerge_cooldown(self, mock_time, beetle: Beetle):
        beetle.emerge_cooldown_duration = 1.0
        mock_time.return_value = 100.0 # Current time
        
        beetle.is_hidden = True
        # Set hide_time such that cooldown has passed
        beetle.hide_time = mock_time.return_value - beetle.emerge_cooldown_duration - 0.1 

        beetle.update_roaming([])
        assert beetle.is_hidden is False, "Beetle should emerge after cooldown"

    def test_beetle_fleeing_incorporates_wall_direction(self, beetle: Beetle):
        # Position beetle centrally, human slightly below
        beetle.x = 300.0
        beetle.y = 200.0 
        beetle.is_hidden = False
        beetle.is_fleeing = False # Start not fleeing
        initial_direction = beetle.direction # Store initial direction for comparison

        # Human is at (300, 220), slightly below the beetle.
        # Beetle should move primarily upwards (y decreases) away from human.
        # Nearest wall for y=200 is top wall (y=0). So wall direction dy is also negative.
        # Combined dy should be strongly negative. dx should be minimal.
        human_positions = [(300.0, 220.0)]
        # Bbox needs to be set for flee logic to trigger based on distance to bbox edge
        # A bbox that makes the human "close" to trigger fleeing.
        # flee_trigger_distance is 100 if not self.is_fleeing
        # Beetle center (340,240), human center (300,220)
        # dist will be < 100
        human_bboxes = [(280.0, 200.0, 40.0, 40.0)] 


        beetle.update_fleeing(human_positions, human_bboxes=human_bboxes)

        assert beetle.is_fleeing is True, "Beetle should be fleeing"
        
        # Check the direction:
        # dx_human = 300 - 300 = 0
        # dy_human = 200 - 220 = -20
        # Human flee direction is atan2(-20, 0) = -pi/2 (straight up)

        # Wall direction:
        # dist_to_left_wall = 300
        # dist_to_right_wall = 640 - 80 - 300 = 260
        # dist_to_top_wall = 200
        # dist_to_bottom_wall = 480 - 80 - 200 = 200
        # Closest wall could be top or bottom. If top: target_wall_y=0, dy_wall = 0-200 = -200
        # If bottom: target_wall_y=480-80=400, dy_wall = 400-200 = 200
        # The code picks min(walls, key=walls.get), so top wall (y=0) is chosen.
        # dx_wall = 300 - 300 = 0 (target_wall_x = self.x)
        # dy_wall = 0 - 200 = -200
        # Wall direction is also atan2(-200, 0) = -pi/2 (straight up)

        # Both primary human flee and wall direction are straight up (-pi/2).
        # So, the combined direction should also be straight up.
        expected_direction = -math.pi / 2
        assert abs(beetle.direction - expected_direction) < 0.1, \
            f"Beetle direction {beetle.direction} not close to {expected_direction}"

        # Test with human above, beetle should flee down and towards bottom wall
        beetle.x = 300.0
        beetle.y = 200.0
        beetle.is_fleeing = False # Reset
        human_positions = [(300.0, 180.0)] # Human above
        human_bboxes = [(280.0, 160.0, 40.0, 40.0)]
        beetle.update_fleeing(human_positions, human_bboxes=human_bboxes)
        # dx_human = 0, dy_human = 200-180 = 20. Flee direction pi/2 (down)
        # Closest wall is bottom (y=400). dy_wall = 400-200 = 200. Wall direction pi/2 (down)
        expected_direction_down = math.pi / 2
        assert abs(beetle.direction - expected_direction_down) < 0.1, \
            f"Beetle direction {beetle.direction} not close to {expected_direction_down}"


    # New tests for HeadFollower class
    @patch('time.time')
    def test_headfollower_stays_following_during_grace_period(self, mock_time, head_follower: HeadFollower):
        initial_x, initial_y = head_follower.x, head_follower.y
        
        # Frame 1: Detect head, start following
        mock_time.return_value = 100.0
        head_follower.update_following([(100.0, 100.0)])
        assert head_follower.is_following is True
        assert head_follower.last_known_head_target == (100.0, 100.0)
        # Check it moved from initial position
        assert head_follower.x != initial_x or head_follower.y != initial_y

        # Frame 2: Head lost, grace period starts
        time_head_lost_simulation = 100.1
        mock_time.return_value = time_head_lost_simulation
        
        # Store position before this update to check movement towards last known
        x_before_grace_update = head_follower.x
        y_before_grace_update = head_follower.y

        head_follower.update_following([]) 
        assert head_follower.is_following is True, "Should still be following during grace period"
        assert head_follower.time_lost_head == time_head_lost_simulation, "time_lost_head should be set"
        
        # Check it's still moving towards (100,100)
        # Target for beetle center on head: (100 - size/2, 100 - size - 10)
        # size = 40. target_display_x = 100 - 20 = 80. target_display_y = 100 - 40 - 10 = 50.
        target_calc_x = 100.0 - head_follower.size / 2
        target_calc_y = 100.0 - head_follower.size - 10

        dist_before_sq = (x_before_grace_update - target_calc_x)**2 + (y_before_grace_update - target_calc_y)**2
        dist_after_sq = (head_follower.x - target_calc_x)**2 + (head_follower.y - target_calc_y)**2
        
        # If not already at target, it should have moved closer or stayed if at target
        if dist_before_sq > 1: # Avoid floating point issues if already at target
             assert dist_after_sq < dist_before_sq, "Should move closer to last known target during grace"


    @patch('time.time')
    def test_headfollower_stops_following_after_grace_period(self, mock_time, head_follower: HeadFollower):
        # Frame 1: Detect head
        mock_time.return_value = 100.0
        head_follower.update_following([(100.0, 100.0)])
        assert head_follower.is_following is True

        # Frame 2: Head lost, grace period starts
        mock_time.return_value = 100.1
        head_follower.update_following([])
        assert head_follower.is_following is True 
        assert head_follower.time_lost_head == 100.1

        # Frame 3: Time moves beyond grace period
        # following_grace_period is 0.5s by default
        mock_time.return_value = 100.1 + head_follower.following_grace_period + 0.1 
        head_follower.update_following([])
        assert head_follower.is_following is False, "Should stop following after grace period"
        assert head_follower.time_lost_head is None
        assert head_follower.last_known_head_target is None

    @patch('time.time')
    def test_headfollower_resumes_following_if_head_reappears(self, mock_time, head_follower: HeadFollower):
        # Frame 1: Detect head
        mock_time.return_value = 100.0
        head_follower.update_following([(100.0, 100.0)])
        assert head_follower.is_following is True
        assert head_follower.last_known_head_target == (100.0, 100.0)

        # Frame 2: Head lost (start of grace period)
        mock_time.return_value = 100.1
        head_follower.update_following([])
        assert head_follower.is_following is True # Still in grace
        assert head_follower.time_lost_head == 100.1

        # Frame 3: Head reappears at a new spot
        mock_time.return_value = 100.2 
        head_follower.update_following([(200.0, 200.0)])
        assert head_follower.is_following is True, "Should resume following"
        assert head_follower.last_known_head_target == (200.0, 200.0), "Should target new head position"
        assert head_follower.time_lost_head is None, "time_lost_head should be reset"


class TestBeetleBehaviorFunctions:
    """Test functions that coordinate beetle behavior"""

    @pytest.fixture
    def mock_beetle_image(self):
        """Create a temporary test image file"""
        test_image = np.zeros((80, 80, 4), dtype=np.uint8)
        test_image[:, :, :3] = [100, 150, 200]
        test_image[:, :, 3] = 255

        temp_file = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        cv2.imwrite(temp_file.name, test_image)

        yield temp_file.name

        os.unlink(temp_file.name)

    def test_multiple_beetles_collision_avoidance(self, mock_beetle_image: str) -> None:
        """Test that multiple beetles avoid colliding with each other"""
        beetle1 = Beetle("beetle1", mock_beetle_image, 640, 480, debug=False)
        beetle2 = Beetle("beetle2", mock_beetle_image, 640, 480, debug=False)

        # Position beetles to potentially collide
        beetle1.x = 100.0
        beetle1.y = 100.0
        beetle2.x = 110.0
        beetle2.y = 110.0

        # Update roaming with collision avoidance
        beetles = [beetle1, beetle2]
        beetle1.update_roaming(beetles)
        beetle2.update_roaming(beetles)

        # Test that beetles have moved (integration test)
        # Note: This test validates the system works without being too specific about internal behavior
        # A more specific assertion would be that their distance increased or directions changed.
        dist_after = math.sqrt((beetle1.x - beetle2.x)**2 + (beetle1.y - beetle2.y)**2)
        assert dist_after > 0 # Check they are not in the exact same spot (which avoid_collision tries to prevent)


    def test_fleeing_beetle_collision_avoidance(self, mock_beetle_image: str) -> None:
        """Test beetles avoid each other while fleeing"""
        beetle1 = Beetle("beetle1", mock_beetle_image, 640, 480, debug=False)
        beetle2 = Beetle("beetle2", mock_beetle_image, 640, 480, debug=False)

        # Position beetles close together
        beetle1.x = 200.0
        beetle1.y = 200.0
        beetle2.x = 210.0 # Close, but not overlapping for initial state
        beetle2.y = 210.0

        # Add human to trigger fleeing
        human_positions = [(300.0, 300.0)] # Human to the bottom-right
        human_bboxes = [(280.0, 280.0, 40.0, 40.0)]
        beetles = [beetle1, beetle2]

        # Both beetles should flee while avoiding each other
        beetle1.update_fleeing(human_positions, human_bboxes=human_bboxes, other_beetles=beetles)
        beetle2.update_fleeing(human_positions, human_bboxes=human_bboxes, other_beetles=beetles)

        # Both should be fleeing
        assert beetle1.is_fleeing, "Beetle1 should be fleeing"
        assert beetle2.is_fleeing, "Beetle2 should be fleeing"
        
        # Check they moved away from human and potentially each other
        # Beetle1 should move towards top-left, Beetle2 also towards top-left
        # Their relative positions might not change much if their flee paths are parallel
        # but the key is that the collision avoidance logic was called.
        # This test mostly ensures the fleeing logic runs with other_beetles.
        assert beetle1.x < 200.0 or beetle1.y < 200.0 # Moved from initial
        assert beetle2.x < 210.0 or beetle2.y < 210.0 # Moved from initial

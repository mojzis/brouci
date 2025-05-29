import math
import random
import time

import cv2
import numpy as np
from numpy.typing import NDArray


class Beetle:
    def __init__(
        self,
        name: str,
        image_path: str,
        frame_width: int,
        frame_height: int,
        debug: bool = False,
    ) -> None:
        self.name: str = name
        self.image: NDArray[np.uint8] = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if self.image is None:
            raise ValueError(f"Could not load beetle image: {image_path}")

        # Resize beetle to reasonable size
        self.size: int = 80
        self.image = cv2.resize(self.image, (self.size, self.size))

        # Position and movement
        self.x: float = random.randint(0, frame_width - self.size)
        self.y: float = random.randint(0, frame_height - self.size)
        self.speed: int = 5 if name == "kudlanka" else 8  # blecha is faster
        self.direction: float = random.uniform(0, 2 * math.pi)

        # State
        self.is_fleeing: bool = False
        self.is_hidden: bool = False  # Hidden in wall when fleeing
        self.flee_target_x: float = 0
        self.flee_target_y: float = 0
        self.last_direction_change: float = time.time()
        self.last_flee_time: float = 0  # Track when beetle last fled
        self.flee_cooldown: float = (
            2.0  # Seconds to continue fleeing after losing sight
        )
        self.hide_time: float = 0  # When beetle went into hiding

        # Frame boundaries
        self.frame_width: int = frame_width
        self.frame_height: int = frame_height

        # Debug flag
        self.debug: bool = debug

    def get_bounding_box(self) -> tuple[float, float, float, float]:
        """Get beetle's bounding box (x, y, width, height)"""
        return (self.x, self.y, self.size, self.size)

    def is_at_wall(self) -> bool:
        """Check if beetle is touching any wall"""
        wall_threshold = 5  # Pixels from edge to consider "at wall"
        return (
            self.x <= wall_threshold
            or self.x >= self.frame_width - self.size - wall_threshold
            or self.y <= wall_threshold
            or self.y >= self.frame_height - self.size - wall_threshold
        )

    @staticmethod
    def rectangles_intersect(
        rect1: tuple[float, float, float, float],
        rect2: tuple[float, float, float, float],
    ) -> bool:
        """Check if two rectangles intersect
        Rectangles are (x, y, width, height) where (x,y) is top-left corner"""
        x1, y1, w1, h1 = rect1
        x2, y2, w2, h2 = rect2

        # Check if rectangles don't overlap
        if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
            return False
        return True

    def check_collision(self, other_beetle: "Beetle") -> bool:
        """Check if this beetle collides with another beetle using bounding boxes"""
        # Hidden beetles don't collide
        if self.is_hidden or other_beetle.is_hidden:
            return False

        return self.rectangles_intersect(
            self.get_bounding_box(), other_beetle.get_bounding_box()
        )

    def avoid_collision(self, other_beetle: "Beetle") -> None:
        """Adjust direction to avoid collision with another beetle"""
        # Calculate direction away from other beetle
        dx = self.x - other_beetle.x
        dy = self.y - other_beetle.y

        # If beetles are exactly on top of each other, pick random direction
        if abs(dx) < 1 and abs(dy) < 1:
            dx = random.uniform(-10, 10)
            dy = random.uniform(-10, 10)

        # Set new direction away from other beetle
        if dx != 0 or dy != 0:
            self.direction = math.atan2(dy, dx)
            # Add some randomness to make movement more natural
            self.direction += random.uniform(-math.pi / 6, math.pi / 6)

    def update_roaming(self, other_beetles: list["Beetle"] | None = None) -> None:
        """Update beetle position when roaming (no humans detected)"""
        # Check if beetle should emerge from hiding (no humans around)
        if self.is_hidden:
            self.is_hidden = False
            if self.debug:
                print(f"🪲🏠 Beetle {self.name} EMERGING from wall (no humans)")
            # Move slightly away from wall when emerging
            if self.x <= 10:
                self.x = 15
            elif self.x >= self.frame_width - self.size - 10:
                self.x = self.frame_width - self.size - 15
            if self.y <= 10:
                self.y = 15
            elif self.y >= self.frame_height - self.size - 10:
                self.y = self.frame_height - self.size - 15
            # Reset fleeing state when emerging
            self.is_fleeing = False
            return

        current_time = time.time()
        old_x, old_y = self.x, self.y

        # Check for collisions with other beetles
        if other_beetles:
            for other in other_beetles:
                if other != self and self.check_collision(other):
                    self.avoid_collision(other)
                    if self.debug:
                        print(
                            f"🪲↔️ Beetle {self.name} avoiding collision with {other.name}"
                        )

        # Change direction randomly every 2-4 seconds
        if current_time - self.last_direction_change > 3:  # Fixed interval for testing
            old_direction = self.direction
            self.direction += random.uniform(-math.pi / 3, math.pi / 3)
            self.last_direction_change = current_time
            if self.debug:
                print(
                    f"Beetle {self.name} changed direction from {old_direction:.2f} to {self.direction:.2f}"
                )

        # Move in current direction
        self.x += self.speed * math.cos(self.direction)
        self.y += self.speed * math.sin(self.direction)

        # Bounce off walls
        if self.x <= 0 or self.x >= self.frame_width - self.size:
            self.direction = math.pi - self.direction
            self.x = max(0, min(self.frame_width - self.size, self.x))
            if self.debug:
                print(f"Beetle {self.name} bounced off horizontal wall")

        if self.y <= 0 or self.y >= self.frame_height - self.size:
            self.direction = -self.direction
            self.y = max(0, min(self.frame_height - self.size, self.y))
            if self.debug:
                print(f"Beetle {self.name} bounced off vertical wall")

        # Only print movement every 30 frames to reduce spam
        if self.debug and int(current_time * 10) % 30 == 0:
            print(
                f"Beetle {self.name} roaming: ({old_x:.1f},{old_y:.1f}) -> ({self.x:.1f},{self.y:.1f})"
            )

    def update_fleeing(
        self,
        human_positions: list[tuple[float, float]],
        human_bboxes: list[tuple[float, float, float, float]] | None = None,
        other_beetles: list["Beetle"] | None = None,
    ) -> None:
        """Update beetle position when fleeing from humans"""
        current_time = time.time()

        if not human_positions:
            # Check if we should continue fleeing due to cooldown
            if (
                self.is_fleeing
                and (current_time - self.last_flee_time) < self.flee_cooldown
            ):
                # Continue moving in the same direction during cooldown (if not hidden)
                if not self.is_hidden:
                    self.x += self.speed * 3 * math.cos(self.direction)
                    self.y += self.speed * 3 * math.sin(self.direction)
                    self.x = max(0, min(self.frame_width - self.size, self.x))
                    self.y = max(0, min(self.frame_height - self.size, self.y))
                return
            else:
                if self.is_fleeing:
                    if self.debug:
                        print(
                            f"🪲✋ Beetle {self.name} STOPPED FLEEING (cooldown expired)"
                        )
                self.is_fleeing = False
                return

        # Check for collisions with human bounding boxes
        should_flee = False
        nearest_human = None
        min_distance = float("inf")

        if human_bboxes:
            # Use bounding box collision detection
            beetle_bbox = self.get_bounding_box()

            for i, human_bbox in enumerate(human_bboxes):
                if self.rectangles_intersect(beetle_bbox, human_bbox):
                    # We're colliding with a human!
                    should_flee = True
                    nearest_human = human_positions[i]
                    min_distance = 0
                    break
                else:
                    # Check distance to edge of human bounding box
                    hx, hy, hw, hh = human_bbox
                    human_center_x = hx + hw / 2
                    human_center_y = hy + hh / 2

                    # Calculate distance from beetle center to human bbox edge
                    beetle_center_x = self.x + self.size / 2
                    beetle_center_y = self.y + self.size / 2

                    # Distance to human bbox edge (approximate)
                    dx = max(
                        0,
                        abs(beetle_center_x - human_center_x) - hw / 2 - self.size / 2,
                    )
                    dy = max(
                        0,
                        abs(beetle_center_y - human_center_y) - hh / 2 - self.size / 2,
                    )
                    edge_distance = math.sqrt(dx * dx + dy * dy)

                    if edge_distance < min_distance:
                        min_distance = edge_distance
                        nearest_human = human_positions[i]

            # Check if we should flee based on distance to bbox edge
            flee_trigger_distance = (
                100 if not self.is_fleeing else 150
            )  # Reduced distances for bbox
            should_flee = should_flee or (min_distance < flee_trigger_distance)
        else:
            # Fallback to center point detection if no bboxes available
            for pos in human_positions:
                distance = math.sqrt((self.x - pos[0]) ** 2 + (self.y - pos[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    nearest_human = pos

            flee_trigger_distance = 400 if not self.is_fleeing else 500
            should_flee = nearest_human and min_distance < flee_trigger_distance

        if should_flee and nearest_human:
            # If hidden, stay hidden while humans are around
            if self.is_hidden:
                return

            was_fleeing = self.is_fleeing
            self.is_fleeing = True
            self.last_flee_time = current_time

            if not was_fleeing:  # Only log when starting to flee
                if self.debug:
                    print(
                        f"🪲➡️ Beetle {self.name} STARTS FLEEING from human at ({nearest_human[0]:.0f},{nearest_human[1]:.0f}), distance: {min_distance:.0f}"
                    )

            # Calculate flee direction (away from human)
            dx = self.x - nearest_human[0]
            dy = self.y - nearest_human[1]

            # If very close to human, use panic mode with random strong movement
            if min_distance < 50:  # Panic distance threshold
                # Generate strong random movement away from human
                panic_angle = (
                    math.atan2(dy, dx)
                    if (abs(dx) > 0.1 or abs(dy) > 0.1)
                    else random.uniform(0, 2 * math.pi)
                )
                panic_angle += random.uniform(
                    -math.pi / 4, math.pi / 4
                )  # Add randomness
                dx = math.cos(panic_angle) * 100
                dy = math.sin(panic_angle) * 100
                print(
                    f"🪲😱 Beetle {self.name} PANICKING! Distance: {min_distance:.0f}"
                )

            # Normalize and apply flee speed
            length = math.sqrt(dx * dx + dy * dy)
            if length > 0.1:  # Lower threshold to prevent division issues
                # Check for collisions with other beetles while fleeing
                flee_direction = math.atan2(dy, dx)
                if other_beetles:
                    for other in other_beetles:
                        if other != self and self.check_collision(other):
                            # Adjust flee direction to avoid collision
                            avoid_dx = self.x - other.x
                            avoid_dy = self.y - other.y
                            if abs(avoid_dx) > 1 or abs(avoid_dy) > 1:
                                avoid_angle = math.atan2(avoid_dy, avoid_dx)
                                # Blend flee and avoidance directions
                                flee_direction = (flee_direction + avoid_angle) / 2
                                print(
                                    f"🪲↔️💨 Beetle {self.name} avoiding {other.name} while fleeing"
                                )

                # Adjust flee speed based on distance (slower when far, faster when close)
                if min_distance < 50:  # Panic mode - maximum speed
                    speed_multiplier = 15
                else:
                    speed_multiplier = max(3, min(10, 400 / max(min_distance, 50)))

                flee_speed = self.speed * speed_multiplier
                move_x = flee_speed * math.cos(flee_direction)
                move_y = flee_speed * math.sin(flee_direction)

                self.x += move_x
                self.y += move_y

                # Update direction for cooldown movement
                self.direction = flee_direction

                # Only log movement periodically to reduce spam
                if self.debug and int(current_time * 10) % 10 == 0:
                    print(
                        f"🪲💨 Beetle {self.name} fleeing: moved ({move_x:.1f},{move_y:.1f}), distance: {min_distance:.0f}"
                    )
            else:
                # If we somehow have zero length, just move in a random direction
                self.direction = random.uniform(0, 2 * math.pi)
                self.x += self.speed * 10 * math.cos(self.direction)
                self.y += self.speed * 10 * math.sin(self.direction)
                print(f"🪲🔄 Beetle {self.name} emergency random movement!")

            # Keep within bounds and check for wall hiding
            old_x, old_y = self.x, self.y
            self.x = max(0, min(self.frame_width - self.size, self.x))
            self.y = max(0, min(self.frame_height - self.size, self.y))

            # If we hit a boundary while fleeing, hide in the wall
            if (self.x != old_x or self.y != old_y) and self.is_fleeing:
                if not self.is_hidden:
                    self.is_hidden = True
                    self.hide_time = current_time
                    print(f"🪲🏠 Beetle {self.name} HIDING in wall!")

                    # Position beetle exactly at the wall edge
                    if self.x <= 0:
                        self.x = 0
                    elif self.x >= self.frame_width - self.size:
                        self.x = self.frame_width - self.size
                    if self.y <= 0:
                        self.y = 0
                    elif self.y >= self.frame_height - self.size:
                        self.y = self.frame_height - self.size
            elif self.x == old_x and self.y == old_y and not self.is_hidden:
                # Normal boundary bouncing when not hitting walls
                if self.x <= 0 or self.x >= self.frame_width - self.size:
                    self.direction = math.pi - self.direction
                if self.y <= 0 or self.y >= self.frame_height - self.size:
                    self.direction = -self.direction
        else:
            # Only stop fleeing after cooldown period
            if (
                self.is_fleeing
                and (current_time - self.last_flee_time) >= self.flee_cooldown
            ):
                print(
                    f"🪲✋ Beetle {self.name} STOPPED FLEEING (distance: {min_distance:.0f})"
                )
                self.is_fleeing = False
            elif self.is_fleeing:
                # Continue moving during cooldown
                old_x, old_y = self.x, self.y
                self.x += self.speed * 3 * math.cos(self.direction)
                self.y += self.speed * 3 * math.sin(self.direction)
                self.x = max(0, min(self.frame_width - self.size, self.x))
                self.y = max(0, min(self.frame_height - self.size, self.y))

                # Bounce off walls during cooldown
                if self.x != old_x + self.speed * 3 * math.cos(self.direction):
                    self.direction = math.pi - self.direction
                if self.y != old_y + self.speed * 3 * math.sin(self.direction):
                    self.direction = -self.direction

    def draw(self, frame: NDArray[np.uint8]) -> None:
        """Draw beetle on frame"""
        # Don't draw if beetle is hidden in wall
        if self.is_hidden:
            return

        x, y = int(self.x), int(self.y)

        # Clamp to frame boundaries
        x = max(0, min(x, frame.shape[1] - self.size))
        y = max(0, min(y, frame.shape[0] - self.size))

        # Update beetle position if it was clamped
        if x != int(self.x) or y != int(self.y):
            self.x, self.y = float(x), float(y)

        try:
            # Handle transparency if the image has an alpha channel
            if len(self.image.shape) == 3 and self.image.shape[2] == 4:
                # Extract alpha channel
                alpha = self.image[:, :, 3] / 255.0

                # Ensure alpha has the right shape
                alpha = alpha[:, :, np.newaxis]

                # Get the region of interest
                beetle_rgb = self.image[:, :, :3].astype(np.float32)
                frame_region = frame[y : y + self.size, x : x + self.size].astype(
                    np.float32
                )

                # Blend with alpha
                blended = alpha * beetle_rgb + (1 - alpha) * frame_region
                frame[y : y + self.size, x : x + self.size] = blended.astype(np.uint8)
            else:
                # No transparency, just overlay
                frame[y : y + self.size, x : x + self.size] = self.image[:, :, :3]
        except Exception as e:
            print(f"Error drawing beetle {self.name}: {e}")
            # Draw a simple colored rectangle as fallback
            color = (255, 0, 0) if self.name == "kudlanka" else (0, 0, 255)
            frame[y : y + self.size, x : x + self.size] = color


class HeadFollower(Beetle):
    """Special beetle that follows human heads instead of fleeing"""

    def __init__(
        self,
        name: str,
        image_path: str,
        frame_width: int,
        frame_height: int,
        debug: bool = False,
    ) -> None:
        super().__init__(name, image_path, frame_width, frame_height, debug)
        self.target_head_pos: tuple[float, float] | None = None
        self.follow_speed: int = 15  # Faster than regular beetles
        self.is_following: bool = False

        # Smaller size for head follower
        self.size = 40
        self.image = cv2.resize(self.image, (self.size, self.size))

        # Start in center of frame
        self.x = frame_width / 2 - self.size / 2
        self.y = frame_height / 2 - self.size / 2

    def update_following(self, head_positions: list[tuple[float, float]]) -> None:
        """Update position to follow the nearest human head."""
        # Debug: Conditionally print when called
        if self.debug:
            print(f"🔍 {self.name} update_following called with {len(head_positions)} heads")
            if head_positions:
                print(f"   Head positions: {head_positions}")

        if not head_positions:
            # No heads detected - stop following and disappear
            if self.is_following:
                self.is_following = False
                if self.debug:
                    print(f"❌ {self.name} lost track of human head")
            elif self.debug:
                print(f"⏳ {self.name} still waiting for heads (not following)")
            return

        # Find nearest head (or just use first one for simplicity)
        target_head = head_positions[0]

        if not self.is_following:
            self.is_following = True
            if self.debug:
                print(f"✅ {self.name} STARTING to follow human head at {target_head}!")
        elif self.debug:
            print(f"⬆️ {self.name} continuing to follow head")

        # Calculate position on top of head
        head_x, head_y = target_head
        target_x = head_x - self.size / 2  # Center beetle on head
        target_y = head_y - self.size - 10  # Position above head with small offset

        # Smooth movement towards target
        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.sqrt(dx * dx + dy * dy)

        if distance > 5:  # Only move if not close enough
            # Move towards target with smooth interpolation
            move_factor = min(1.0, self.follow_speed / distance)
            self.x += dx * move_factor
            self.y += dy * move_factor

            # Keep within frame bounds
            self.x = max(0, min(self.frame_width - self.size, self.x))
            self.y = max(0, min(self.frame_height - self.size, self.y))

    def update_roaming(self, other_beetles: list["Beetle"] | None = None) -> None:
        """Head followers don't roam - they only follow or stay hidden"""
        pass

    def update_fleeing(
        self,
        human_positions: list[tuple[float, float]],
        human_bboxes: list[tuple[float, float, float, float]] | None = None,
        other_beetles: list["Beetle"] | None = None,
    ) -> None:
        """Head followers don't flee - they're friendly"""
        pass

    def draw(self, frame: NDArray[np.uint8]) -> None:
        """Draw head follower only when following a head"""
        if self.debug:
            print(f"🎨 {self.name} draw() called, is_following={self.is_following}")
        if not self.is_following:
            if self.debug:
                print(f"🚫 {self.name} draw() returning early - not following")
            return

        if self.debug:
            print(f"✏️ {self.name} drawing at position ({self.x:.1f}, {self.y:.1f})")
        # Use parent draw method
        super().draw(frame)

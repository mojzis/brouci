import math
import random
import time

import cv2
import numpy as np
from numpy.typing import NDArray


class Beetle:
    def __init__(self, name: str, image_path: str, frame_width: int, frame_height: int) -> None:
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
        self.flee_target_x: float = 0
        self.flee_target_y: float = 0
        self.last_direction_change: float = time.time()
        self.last_flee_time: float = 0  # Track when beetle last fled
        self.flee_cooldown: float = 2.0  # Seconds to continue fleeing after losing sight

        # Frame boundaries
        self.frame_width: int = frame_width
        self.frame_height: int = frame_height

    def check_collision(self, other_beetle: 'Beetle') -> bool:
        """Check if this beetle collides with another beetle"""
        distance = math.sqrt(
            (self.x - other_beetle.x) ** 2 + (self.y - other_beetle.y) ** 2
        )
        return distance < (self.size * 0.8)  # Use 80% of size for smoother avoidance

    def avoid_collision(self, other_beetle: 'Beetle') -> None:
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

    def update_roaming(self, other_beetles: list['Beetle'] | None = None) -> None:
        """Update beetle position when roaming (no humans detected)"""
        current_time = time.time()
        old_x, old_y = self.x, self.y

        # Check for collisions with other beetles
        if other_beetles:
            for other in other_beetles:
                if other != self and self.check_collision(other):
                    self.avoid_collision(other)
                    print(
                        f"🪲↔️ Beetle {self.name} avoiding collision with {other.name}"
                    )

        # Change direction randomly every 2-4 seconds
        if current_time - self.last_direction_change > 3:  # Fixed interval for testing
            old_direction = self.direction
            self.direction += random.uniform(-math.pi / 3, math.pi / 3)
            self.last_direction_change = current_time
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
            print(f"Beetle {self.name} bounced off horizontal wall")

        if self.y <= 0 or self.y >= self.frame_height - self.size:
            self.direction = -self.direction
            self.y = max(0, min(self.frame_height - self.size, self.y))
            print(f"Beetle {self.name} bounced off vertical wall")

        # Only print movement every 30 frames to reduce spam
        if int(current_time * 10) % 30 == 0:
            print(
                f"Beetle {self.name} roaming: ({old_x:.1f},{old_y:.1f}) -> ({self.x:.1f},{self.y:.1f})"
            )

    def update_fleeing(self, human_positions: list[tuple[float, float]], other_beetles: list['Beetle'] | None = None) -> None:
        """Update beetle position when fleeing from humans"""
        current_time = time.time()

        if not human_positions:
            # Check if we should continue fleeing due to cooldown
            if (
                self.is_fleeing
                and (current_time - self.last_flee_time) < self.flee_cooldown
            ):
                # Continue moving in the same direction during cooldown
                self.x += self.speed * 3 * math.cos(self.direction)
                self.y += self.speed * 3 * math.sin(self.direction)
                self.x = max(0, min(self.frame_width - self.size, self.x))
                self.y = max(0, min(self.frame_height - self.size, self.y))
                return
            else:
                if self.is_fleeing:
                    print(f"🪲✋ Beetle {self.name} STOPPED FLEEING (cooldown expired)")
                self.is_fleeing = False
                return

        # Find nearest human
        min_distance = float("inf")
        nearest_human = None

        for pos in human_positions:
            distance = math.sqrt((self.x - pos[0]) ** 2 + (self.y - pos[1]) ** 2)
            if distance < min_distance:
                min_distance = distance
                nearest_human = pos

        # Use dynamic flee distance based on current state
        flee_trigger_distance = 400 if not self.is_fleeing else 500  # Hysteresis

        if nearest_human and min_distance < flee_trigger_distance:
            was_fleeing = self.is_fleeing
            self.is_fleeing = True
            self.last_flee_time = current_time

            if not was_fleeing:  # Only log when starting to flee
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
                if int(current_time * 10) % 10 == 0:
                    print(
                        f"🪲💨 Beetle {self.name} fleeing: moved ({move_x:.1f},{move_y:.1f}), distance: {min_distance:.0f}"
                    )
            else:
                # If we somehow have zero length, just move in a random direction
                self.direction = random.uniform(0, 2 * math.pi)
                self.x += self.speed * 10 * math.cos(self.direction)
                self.y += self.speed * 10 * math.sin(self.direction)
                print(f"🪲🔄 Beetle {self.name} emergency random movement!")

            # Keep within bounds and bounce if hitting edge
            old_x, old_y = self.x, self.y
            self.x = max(0, min(self.frame_width - self.size, self.x))
            self.y = max(0, min(self.frame_height - self.size, self.y))

            # If we hit a boundary while fleeing, adjust direction
            if self.x != old_x or self.y != old_y:
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
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Basic rules
- when asked for a bigger task, stop first, think, provide a detailed plan and request that it be reviewed. store the plan into a new file within the plans directory
- try to keep things in simple, very organized functions

## Project Overview

Hemzeni is a real-time pose detection application that provides live webcam feeds with pose landmark visualization. The project offers different pose detection backends and interactive features:

1. **AIpose with YoloV7** (`hemzeni/app.py`) - Uses the aipose library with YoloV7 model for pose detection
2. **MediaPipe** (`hemzeni/app_mediapipe.py`) - Uses Google's MediaPipe for pose landmark detection  
3. **Beetles Interactive** (`hemzeni/app_beetles.py`) - AIpose-based app with interactive beetle creatures that roam when no humans are detected and flee when humans appear
4. **Frame Capture** (`hemzeni/app_frame_capture.py`) - Flask app for capturing video frames to disk for debugging
5. **Beetle Replay** (`hemzeni/app_beetles_replay.py`) - Flask app for replaying saved frames with beetle behavior

All applications are built with Flask for the web interface and OpenCV for camera handling.

## Development Commands

### Environment Setup
```bash
# Install dependencies using Poetry
poetry install

# Activate virtual environment
poetry shell
```

### Running Applications
```bash
# Run the AIpose-based pose detection app
python hemzeni/app.py

# Run the MediaPipe-based pose detection app  
python hemzeni/app_mediapipe.py

# Run the interactive beetles app
python hemzeni/app_beetles.py

# Run the frame capture tool (port 5001)
python hemzeni/app_frame_capture.py

# Run the beetle replay debugger (port 5002)
python hemzeni/app_beetles_replay.py
```

### Code Quality
```bash
# Lint code with ruff
ruff check .

# Format code with ruff
ruff format .
```

## Architecture Notes

### Core Components
- **Camera Input**: All apps use OpenCV's VideoCapture for webcam access
- **Pose Detection**: Two different backends provide pose landmark detection
- **Visualization**: Real-time rendering of pose landmarks on video frames
- **Web Framework**: Flask handles web routing and serving HTML templates

### Key Differences Between Apps
- `app.py` uses aipose YoloV7Pose model for pose detection
- `app_mediapipe.py` uses MediaPipe's pose landmarker with a pre-trained model file (`pose_landmarker.task`)
- `app_beetles.py` extends the aipose app with interactive beetle creatures (kudlanka and blecha) that exhibit different behaviors based on human presence
- `app_frame_capture.py` captures video frames to disk for debugging purposes
- `app_beetles_replay.py` replays saved frames with beetle behavior for debugging
- MediaPipe version includes segmentation mask output capability

### Beetle Behavior System
- **Kudlanka** and **Blecha** beetles roam randomly when no humans are detected
- Beetles flee away from humans when they appear within 200 pixels
- Each beetle has different movement speeds and characteristics
- Beetle images are loaded from the `img/` directory

### Dependencies
- **Core**: flask, opencv-python, torch, pandas
- **Pose Detection**: aipose (YoloV7), mediapipe (commented out in pyproject.toml)
- **Development**: ruff for linting and formatting

## Coding Standards

### Type Annotations
All new code should use modern Python type annotations:

1. **Use lowercase built-in types** (Python 3.9+):
   ```python
   # Good
   def process(items: list[str]) -> dict[str, int]:
       pass
   
   # Bad - don't import from typing for basic types
   from typing import List, Dict
   def process(items: List[str]) -> Dict[str, int]:
       pass
   ```

2. **Use `|` for union types** instead of `Optional`:
   ```python
   # Good
   def find_item(name: str) -> Item | None:
       pass
   
   # Bad - Optional is old style
   from typing import Optional
   def find_item(name: str) -> Optional[Item]:
       pass
   ```

3. **Annotate all function parameters and return types**:
   ```python
   # Good
   def calculate_distance(x1: float, y1: float, x2: float, y2: float) -> float:
       return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
   
   # Bad - missing annotations
   def calculate_distance(x1, y1, x2, y2):
       return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
   ```

4. **Annotate class attributes**:
   ```python
   class Beetle:
       def __init__(self, name: str) -> None:
           self.name: str = name
           self.position: tuple[float, float] = (0.0, 0.0)
           self.is_fleeing: bool = False
   ```

5. **Use `NDArray` from numpy.typing for numpy arrays**:
   ```python
   from numpy.typing import NDArray
   import numpy as np
   
   def process_image(image: NDArray[np.uint8]) -> NDArray[np.float32]:
       return image.astype(np.float32) / 255.0
   ```

### Code Organization
- Keep related functionality in separate modules (e.g., `beetle.py`, `pose_detector.py`, `controller.py`)
- Use clear, descriptive names for classes, functions, and variables
- Follow single responsibility principle - each class/function should do one thing well

### Error Handling
- Use proper exception handling where appropriate
- Log errors appropriately without spamming the console
- Provide meaningful error messages to users via Flask's error handling

### Tests
- every function apart from the app file should have a test
- file app_beetles is not tested
- file with tests are within the dir tests
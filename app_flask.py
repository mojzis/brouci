"""Flask app for real-time beetle detection and head following."""

import time

import cv2
from flask import Flask, Response, render_template

from hemzeni.controller import BeetleController

app = Flask(__name__)

# Global controller instance
controller = None


def initialize_controller():
    """Initialize the beetle controller."""
    global controller
    if controller is None:
        controller = BeetleController(debug=False)
        controller.initialize_beetles()
        print("🪲 Controller initialized!")
        print(f"Beetles: {[beetle.name for beetle in controller.beetles]}")
        print(
            f"Head follower: {controller.head_follower.name if controller.head_follower else 'None'}"
        )


def generate_frames():
    """Generate video frames with beetle overlay."""
    global controller

    if controller is None:
        initialize_controller()

    frame_count = 0
    start_time = time.time()

    while True:
        try:
            # Process frame
            result = controller.process_frame()

            if result is None:
                # Frame rate limiting - skip this iteration
                time.sleep(0.001)
                continue

            frame, human_positions, human_bboxes = result
            frame_count += 1

            # Add frame counter and FPS info
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Draw debug info
            cv2.putText(
                frame,
                f"Frame: {frame_count}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Humans: {len(human_positions)}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

            # Head follower debug info
            if controller.head_follower:
                status = (
                    "Following" if controller.head_follower.is_following else "Waiting"
                )
                cv2.putText(
                    frame,
                    f"HeadFollower: {status}",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Position: ({int(controller.head_follower.x)}, {int(controller.head_follower.y)})",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2,
                )

            # Convert frame to JPEG
            _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            frame_bytes = buffer.tobytes()

            # Yield frame in proper format for streaming
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

        except Exception as e:
            print(f"Error in frame generation: {e}")
            # Generate a simple error frame
            error_frame = cv2.putText(
                cv2.ones((480, 640, 3), dtype=cv2.uint8) * 50,
                f"Error: {str(e)}",
                (50, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            _, buffer = cv2.imencode(".jpg", error_frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )


@app.route("/")
def index():
    """Main page with video stream."""
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/status")
def status():
    """Status endpoint for debugging."""
    global controller

    if controller is None:
        return {"status": "Controller not initialized"}

    status_info = {
        "controller_initialized": controller.initialized,
        "beetles_count": len(controller.beetles),
        "beetles_names": [beetle.name for beetle in controller.beetles],
        "head_follower_exists": controller.head_follower is not None,
        "head_follower_name": controller.head_follower.name
        if controller.head_follower
        else None,
        "head_follower_following": controller.head_follower.is_following
        if controller.head_follower
        else False,
        "head_follower_position": (
            controller.head_follower.x,
            controller.head_follower.y,
        )
        if controller.head_follower
        else None,
    }

    return status_info


@app.route("/test_head_detection")
def test_head_detection():
    """Test endpoint to manually trigger head following."""
    global controller

    if controller is None or controller.head_follower is None:
        return {"error": "Controller or head follower not initialized"}

    # Simulate head detection at center of screen
    test_head_positions = [(320.0, 240.0)]
    controller.head_follower.update_following(test_head_positions)

    return {
        "message": "Test head detection triggered",
        "head_positions": test_head_positions,
        "is_following": controller.head_follower.is_following,
        "position": (controller.head_follower.x, controller.head_follower.y),
    }


if __name__ == "__main__":
    print("🚀 Starting Flask Beetle App...")
    print("📱 Open http://localhost:8083 in your browser")
    print("📊 Status info at http://localhost:8083/status")
    print("🧪 Test head detection at http://localhost:8083/test_head_detection")

    # Initialize controller before starting
    initialize_controller()

    # Run Flask app with logging disabled
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    app.run(debug=False, host="127.0.0.1", port=8083, threaded=True)

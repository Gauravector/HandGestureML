"""
inference.py
Real-time cursor control using the right-hand index fingertip (landmark 8),
detected with MediaPipe Tasks API HandLandmarker (mediapipe==0.10.35).

Pipeline per frame:
    webcam frame -> HandLandmarker (IMAGE mode) -> normalized landmark (x, y)
    -> pixel coords in frame -> mapped + smoothed screen coords -> pyautogui.moveTo()

A placeholder `classify_gesture()` is included as the wiring point for a
teammate's trained gesture classifier (click / scroll / etc.).

Requirements:
    pip install mediapipe==0.10.35 opencv-python==4.13.0.92 numpy==2.2.6 pyautogui

You also need the HandLandmarker model file. Download it once:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
Save it next to this script as "hand_landmarker.task" (or change MODEL_PATH below).
"""

import time
import collections

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

MODEL_PATH = "hand_landmarker.task"

CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

# Index fingertip landmark id in MediaPipe's 21-point hand model.
INDEX_FINGERTIP = 8

# --- Coordinate-mapping tuning ---
# "Frame reduction" carves out a smaller active rectangle inside the webcam
# frame so you don't have to stretch your hand to the physical edges of the
# camera view to reach the edges of the screen. Increase this if reaching
# screen corners feels uncomfortable; decrease it if the cursor feels
# insensitive near the frame edges.
FRAME_REDUCTION_X = 100  # pixels trimmed from left+right of the frame
FRAME_REDUCTION_Y = 100  # pixels trimmed from top+bottom of the frame

# --- Smoothing tuning ---
# Exponential Moving Average factor. Higher = more responsive/jittery,
# lower = smoother/laggier. Try values between 0.2 and 0.6.
SMOOTHING_ALPHA = 0.35

# Disable pyautogui's built-in per-call delay so our own smoothing loop
# controls the pacing (we already run at camera frame rate).
pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True  # keep this on: slam mouse to a screen corner to abort


# --------------------------------------------------------------------------
# Placeholder for teammate's gesture classifier
# --------------------------------------------------------------------------

def is_thumbs_up(hand_landmarks):
    """
    Rule-based check for a "thumbs up" hand shape.

    Landmark ids used (see MediaPipe's 21-point hand model):
        4  = thumb tip        2  = thumb MCP (base knuckle)
        8  = index tip        6  = index PIP (middle knuckle)
        12 = middle tip       10 = middle PIP
        16 = ring tip         14 = ring PIP
        20 = pinky tip        18 = pinky PIP

    Logic:
        - Thumb is "extended upward": thumb tip is meaningfully above
          (smaller y, since image y grows downward) its own base knuckle.
        - Other four fingers are "curled": each fingertip's y is below
          (numerically greater than) its own PIP knuckle's y, i.e. folded
          into the palm rather than pointing outward.

    This is a simple, fast heuristic — good enough to test the pipeline
    before a trained classifier replaces it. It can misfire on hands held
    at extreme angles to the camera; a trained model will be more robust.
    """
    thumb_tip = hand_landmarks[4]
    thumb_base = hand_landmarks[2]

    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    thumb_extended = thumb_tip.y < (thumb_base.y - 0.05)

    fingers_curled = all(
        hand_landmarks[tip].y > hand_landmarks[pip].y
        for tip, pip in zip(finger_tips, finger_pips)
    )

    return thumb_extended and fingers_curled


def classify_gesture(hand_landmarks, handedness_label):
    """
    Rule-based placeholder for gesture detection — swap this out for a
    trained classifier later without changing how it's called.

    Args:
        hand_landmarks: list of 21 mediapipe NormalizedLandmark objects
                         (each has .x, .y, .z in [0, 1] normalized coords)
                         for the currently tracked hand.
        handedness_label: "Left" or "Right" (string) as reported by MediaPipe.

    Returns:
        A gesture label (str), or None if no gesture is recognized this frame.

    Currently implemented:
        - Right hand thumbs up -> "left_click"
        - Left hand thumbs up  -> "right_click"

    TODO (once trained model is ready): replace body with
        features = extract_features(hand_landmarks)
        gesture = trained_model.predict(features)
        return gesture
    """
    if is_thumbs_up(hand_landmarks):
        if handedness_label == "Right":
            return "left_click"
        elif handedness_label == "Left":
            return "right_click"
    return None


# --------------------------------------------------------------------------
# Smoothing helper
# --------------------------------------------------------------------------

class EMASmoother:
    """Exponential moving average smoother for a 2D point.

    smoothed_new = alpha * raw_new + (1 - alpha) * smoothed_old

    This is a simple low-pass filter: it blends the fresh (possibly jittery)
    reading with the previous smoothed value, so sudden one-frame jitters
    only nudge the output a little instead of snapping the cursor around.
    """

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._prev = None

    def update(self, point: np.ndarray) -> np.ndarray:
        if self._prev is None:
            self._prev = point
            return point
        smoothed = self.alpha * point + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed


# --------------------------------------------------------------------------
# Coordinate mapping: webcam frame space -> screen space
# --------------------------------------------------------------------------

def map_to_screen(pixel_x, pixel_y, frame_w, frame_h, screen_w, screen_h):
    """
    Maps a fingertip position in webcam pixel coordinates to OS screen
    coordinates.

    Step 1 - MediaPipe gives normalized coordinates in [0, 1] relative to
             the frame. We convert those to pixel coordinates elsewhere
             (pixel_x = landmark.x * frame_w) before calling this function.

    Step 2 - We define an "active rectangle" inside the frame, inset by
             FRAME_REDUCTION_X / FRAME_REDUCTION_Y on each side. Only
             movement inside this smaller rectangle is considered — this
             is what lets you reach every screen edge without stretching
             your hand to the camera's physical edge.

    Step 3 - Linear interpolation (the same math as np.interp) rescales the
             pixel position from the active-rectangle range to the full
             screen resolution range:

                 screen_x = (pixel_x - x_min) / (x_max - x_min) * screen_w

             np.interp does this ratio computation for us and additionally
             clamps automatically to the given output range.

    Step 4 - Clamp the result so the cursor can never be sent slightly
             off-screen (which would look like it "sticks" at the edge or,
             on some OSes, throw an error).
    """
    x_min, x_max = FRAME_REDUCTION_X, frame_w - FRAME_REDUCTION_X
    y_min, y_max = FRAME_REDUCTION_Y, frame_h - FRAME_REDUCTION_Y

    screen_x = np.interp(pixel_x, [x_min, x_max], [0, screen_w])
    screen_y = np.interp(pixel_y, [y_min, y_max], [0, screen_h])

    screen_x = float(np.clip(screen_x, 0, screen_w - 1))
    screen_y = float(np.clip(screen_y, 0, screen_h - 1))

    return screen_x, screen_y


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------

def main():
    screen_w, screen_h = pyautogui.size()
    print(f"Screen resolution detected: {screen_w}x{screen_h}")

    # --- Build the HandLandmarker (Tasks API), IMAGE running mode ---
    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,  # per-frame, synchronous
        num_hands=2,               # detect both hands so we can filter for "Right"
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_INDEX / permissions.")

    smoother = EMASmoother(SMOOTHING_ALPHA)

    prev_time = time.time()
    fps_history = collections.deque(maxlen=30)

    # Debounce state: tracks whether "left_click" was already active last
    # frame, so we fire pyautogui.click() once per thumbs-up gesture
    # (on the transition into the pose) instead of once per frame while
    # the thumb stays up (which would spam dozens of clicks/sec).
    gesture_was_active = {"left_click": False, "right_click": False}

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            # Mirror the frame horizontally so it feels like a mirror to the
            # user (move hand right -> cursor moves right on screen).
            # NOTE: MediaPipe's handedness labeling assumes a mirrored
            # (selfie-style) input image, so flipping here also keeps the
            # "Right"/"Left" labels matching the user's actual hands.
            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Synchronous per-frame detection (IMAGE mode).
            result = landmarker.detect(mp_image)

            right_hand_landmarks = None
            left_hand_landmarks = None

            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(
                    result.hand_landmarks, result.handedness
                ):
                    label = handedness[0].category_name  # "Left" or "Right"
                    if label == "Right" and right_hand_landmarks is None:
                        right_hand_landmarks = hand_landmarks
                    elif label == "Left" and left_hand_landmarks is None:
                        left_hand_landmarks = hand_landmarks

            if right_hand_landmarks is not None:
                tip = right_hand_landmarks[INDEX_FINGERTIP]

                # Normalized [0,1] -> pixel coordinates in the frame.
                pixel_x = tip.x * frame_w
                pixel_y = tip.y * frame_h

                screen_x, screen_y = map_to_screen(
                    pixel_x, pixel_y, frame_w, frame_h, screen_w, screen_h
                )

                smoothed_x, smoothed_y = smoother.update(
                    np.array([screen_x, screen_y])
                )

                pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)

                # --- Gesture classifier hook point (right hand) ---
                gesture = classify_gesture(right_hand_landmarks, "Right")

                # Left click fires once per gesture (on the transition into
                # thumbs-up), not once per frame the thumb stays up.
                is_left_click_now = (gesture == "left_click")
                if is_left_click_now and not gesture_was_active["left_click"]:
                    pyautogui.click(button="left")
                gesture_was_active["left_click"] = is_left_click_now

                if gesture == "scroll_up":
                    pyautogui.scroll(20)
                elif gesture == "scroll_down":
                    pyautogui.scroll(-20)
                # Add more gesture -> action mappings here once the model
                # is wired in.

                # Debug visualization
                cv2.circle(
                    frame, (int(pixel_x), int(pixel_y)), 8, (0, 255, 0), -1
                )

            # --- Gesture classifier hook point (left hand) ---
            # Checked independently of the right-hand block above since the
            # cursor-moving hand (right) and the click hand (left) are not
            # the same hand — left hand may be present even if right isn't.
            if left_hand_landmarks is not None:
                left_gesture = classify_gesture(left_hand_landmarks, "Left")

                is_right_click_now = (left_gesture == "right_click")
                if is_right_click_now and not gesture_was_active["right_click"]:
                    pyautogui.click(button="right")
                gesture_was_active["right_click"] = is_right_click_now
            else:
                gesture_was_active["right_click"] = False

            # FPS overlay (helps you judge if smoothing feels laggy vs. the
            # camera itself being slow)
            now = time.time()
            fps_history.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = sum(fps_history) / len(fps_history)
            cv2.putText(
                frame, f"FPS: {fps:.1f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

            cv2.imshow("Hand Cursor Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
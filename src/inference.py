"""
inference.py
Full real-time gesture control loop.

RIGHT hand:
    - 1 finger extended (pointing)  -> moves the cursor (continuous tracking)
    - 5 fingers (open palm) swiped horizontally -> undo / redo
    - 5 fingers (open palm) swiped vertically   -> scroll up / down
    - 3 fingers swiped downward                 -> screenshot

LEFT hand:
    - Static hand shape -> your trained Random Forest classifier
      (thumbs_up, thumbs_down, fist, peace, clock_minimize, clock_maximize,
       clock_close, idle) -> actions.py's perform_static_action()

Requirements:
    pip install mediapipe==0.10.35 opencv-python==4.13.0.92 numpy==2.2.6 pyautogui
Needs hand_landmarker.task in the project root (same as collect_data.py).
"""

import time
import collections

import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# Pulls in your REAL trained model + both action maps.
from actions import predict_gesture, perform_static_action, perform_motion_action


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

MODEL_PATH = "hand_landmarker.task"
CAM_INDEX = 0
CAM_WIDTH = 640
CAM_HEIGHT = 480

INDEX_FINGERTIP = 8   # landmark id for the fingertip used for cursor control

# --- HANDEDNESS FIX ---
# Which physical hand should control the cursor.
TARGET_CURSOR_HAND = "Right"

# IMPORTANT: run this script once, hold up your ACTUAL right hand, and check
# the on-screen "Hand: ..." label. If it says "Left" while you're holding up
# your right hand, your webcam's mirroring doesn't match MediaPipe's
# assumption — set this to True to correct it. This is a one-time,
# hardware-specific check; there's no way to know the right value without
# testing on your actual camera.
INVERT_HANDEDNESS = True

def resolve_handedness(raw_label):
    """Applies the INVERT_HANDEDNESS correction consistently everywhere."""
    if not INVERT_HANDEDNESS:
        return raw_label
    return "Right" if raw_label == "Left" else "Left"


# --- Coordinate-mapping tuning (cursor) ---
FRAME_REDUCTION_X = 130
FRAME_REDUCTION_Y = 130
SMOOTHING_ALPHA = 0.5

# --- Motion gesture tuning ---
MOTION_HISTORY_LEN = 8          # frames of palm-position history to look at
SWIPE_DISPLACEMENT_THRESHOLD = 0.18   # normalized (0-1) distance to count as a swipe
SWIPE_AXIS_DOMINANCE = 1.5      # how much bigger the dominant axis must be
SWIPE_COOLDOWN_SECONDS = 0.6    # minimum gap between two swipe triggers

# --- Static gesture debounce ---
# Prevents firing the same static action every single frame while a
# gesture is held — fires once on the transition INTO the gesture only.
STATIC_GESTURE_COOLDOWN_SECONDS = 0.5

pyautogui.PAUSE = 0.0
pyautogui.FAILSAFE = True


# =============================================================================
# SECTION 2: LANDMARK NORMALISATION
# MUST exactly match extract_and_normalise() in collect_data.py — the model
# was trained on data produced by that exact transformation, so inference
# has to reproduce it identically or predictions will be meaningless.
# =============================================================================

def extract_and_normalise(hand_landmarks_list):
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_list])
    wrist = raw[0]
    normalised = raw - wrist
    max_val = np.max(np.abs(normalised))
    if max_val > 0:
        normalised = normalised / max_val
    return normalised.flatten().tolist()


# =============================================================================
# SECTION 3: FINGER-COUNTING (for right-hand pose: pointing vs palm vs 3-finger)
# =============================================================================

def count_extended_fingers(hand_landmarks_list, hand_label):
    """
    Returns how many of the 5 fingers are extended, using simple tip-vs-PIP
    y-comparisons (extended = tip is above its own middle knuckle, i.e.
    smaller y, since image y grows downward).

    Thumb is handled separately since it extends sideways, not upward —
    compared on x instead, and direction depends on which hand it is.
    """
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    count = sum(
        1 for tip, pip in zip(finger_tips, finger_pips)
        if hand_landmarks_list[tip].y < hand_landmarks_list[pip].y
    )

    thumb_tip = hand_landmarks_list[4]
    thumb_ip  = hand_landmarks_list[3]
    if hand_label == "Right":
        thumb_extended = thumb_tip.x < thumb_ip.x
    else:
        thumb_extended = thumb_tip.x > thumb_ip.x

    return count + (1 if thumb_extended else 0)


def get_palm_center(hand_landmarks_list):
    """
    Average position of wrist + 4 finger base knuckles — a stable point
    to track for whole-hand swipe motion (steadier than a single fingertip,
    since it doesn't jump around as individual fingers move).
    """
    ids = [0, 5, 9, 13, 17]
    xs = [hand_landmarks_list[i].x for i in ids]
    ys = [hand_landmarks_list[i].y for i in ids]
    return np.array([sum(xs) / len(xs), sum(ys) / len(ys)])


# =============================================================================
# SECTION 4: SMOOTHING (unchanged from teammate's version)
# =============================================================================

class EMASmoother:
    def __init__(self, alpha):
        self.alpha = alpha
        self._prev = None

    def update(self, point):
        if self._prev is None:
            self._prev = point
            return point
        smoothed = self.alpha * point + (1 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed


def map_to_screen(pixel_x, pixel_y, frame_w, frame_h, screen_w, screen_h):
    x_min, x_max = FRAME_REDUCTION_X, frame_w - FRAME_REDUCTION_X
    y_min, y_max = FRAME_REDUCTION_Y, frame_h - FRAME_REDUCTION_Y
    screen_x = np.interp(pixel_x, [x_min, x_max], [0, screen_w])
    screen_y = np.interp(pixel_y, [y_min, y_max], [0, screen_h])
    return float(np.clip(screen_x, 0, screen_w - 1)), float(np.clip(screen_y, 0, screen_h - 1))


# =============================================================================
# SECTION 5: MOTION GESTURE DETECTOR (right hand — undo/redo/scroll/screenshot)
# =============================================================================

class MotionGestureDetector:
    """
    Tracks the right hand's palm-center position over recent frames and
    fires a motion action when a clear swipe is detected.

    Key design choice: only accumulates history while the hand is in a
    "swipe-relevant" pose (3 or 5 fingers extended) — NOT while pointing
    (1 finger). This is what stops ordinary cursor movement from being
    misread as a swipe, since pointing and swiping use different hand shapes.
    """

    def __init__(self):
        self.history = collections.deque(maxlen=MOTION_HISTORY_LEN)
        self.last_trigger_time = 0.0

    def reset(self):
        self.history.clear()

    def update(self, hand_landmarks_list, finger_count):
        now = time.time()

        # Pointing pose or ambiguous pose -> not a swipe attempt, clear history
        if finger_count not in (3, 4, 5):
            self.reset()
            return None

        palm = get_palm_center(hand_landmarks_list)
        self.history.append(palm)

        if len(self.history) < MOTION_HISTORY_LEN:
            return None   # not enough history yet to judge a swipe

        if now - self.last_trigger_time < SWIPE_COOLDOWN_SECONDS:
            return None   # still in cooldown from the last swipe

        start = self.history[0]
        end = self.history[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]

        motion_label = None

        if abs(dx) > SWIPE_DISPLACEMENT_THRESHOLD and abs(dx) > abs(dy) * SWIPE_AXIS_DOMINANCE:
            # Horizontal swipe -> only valid with an open palm (4-5 fingers)
            if finger_count >= 4:
                motion_label = 'swipe_left' if dx < 0 else 'swipe_right'

        elif abs(dy) > SWIPE_DISPLACEMENT_THRESHOLD and abs(dy) > abs(dx) * SWIPE_AXIS_DOMINANCE:
            if finger_count == 3 and dy > 0:
                motion_label = 'swipe_down_3finger'      # screenshot, downward only
            elif finger_count >= 4:
                motion_label = 'swipe_down' if dy > 0 else 'swipe_up'

        if motion_label is not None:
            self.last_trigger_time = now
            self.reset()   # clear history so the swipe doesn't re-trigger immediately

        return motion_label


# =============================================================================
# SECTION 6: STATIC GESTURE DEBOUNCER (left hand)
# =============================================================================

class StaticGestureDebouncer:
    """
    Fires perform_static_action() once when a gesture is first recognised,
    not repeatedly every frame it's held. 'idle' and low-confidence frames
    reset the state so the next real gesture can fire fresh.
    """

    def __init__(self):
        self.active_label = None
        self.last_trigger_time = 0.0

    def update(self, label):
        now = time.time()

        if label is None or label == 'idle':
            self.active_label = None
            return

        # Same gesture still being held -> don't refire
        if label == self.active_label:
            return

        if now - self.last_trigger_time < STATIC_GESTURE_COOLDOWN_SECONDS:
            return

        perform_static_action(label)
        self.active_label = label
        self.last_trigger_time = now


# =============================================================================
# SECTION 7: MAIN LOOP
# =============================================================================

def main():
    screen_w, screen_h = pyautogui.size()
    print(f"Screen resolution detected: {screen_w}x{screen_h}")

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.4,
        min_hand_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check CAM_INDEX / permissions.")

    smoother = EMASmoother(SMOOTHING_ALPHA)
    motion_detector = MotionGestureDetector()
    static_debouncer = StaticGestureDebouncer()

    prev_time = time.time()
    fps_history = collections.deque(maxlen=30)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam.")
                break

            frame = cv2.flip(frame, 1)
            frame_h, frame_w = frame.shape[:2]

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            result = landmarker.detect(mp_image)

            right_hand_landmarks = None
            left_hand_landmarks = None
            right_label_seen = None       # raw MediaPipe label, for on-screen debugging
            left_label_seen = None
            left_hand_resolved_label = None   # corrected label, fed to the classifier

            if result.hand_landmarks:
                for hand_landmarks, handedness in zip(result.hand_landmarks, result.handedness):
                    raw_label = handedness[0].category_name
                    label = resolve_handedness(raw_label)   # corrected physical hand
                    if label == TARGET_CURSOR_HAND and right_hand_landmarks is None:
                        right_hand_landmarks = hand_landmarks
                        right_label_seen = raw_label
                    elif label != TARGET_CURSOR_HAND and left_hand_landmarks is None:
                        left_hand_landmarks = hand_landmarks
                        left_label_seen = raw_label
                        left_hand_resolved_label = label   # 'Left', after correction

            # ---------------- RIGHT HAND: cursor + motion gestures ----------------
            if right_hand_landmarks is not None:
                finger_count = count_extended_fingers(right_hand_landmarks, TARGET_CURSOR_HAND)

                if finger_count == 1:
                    # Pointing pose -> move the cursor
                    tip = right_hand_landmarks[INDEX_FINGERTIP]
                    pixel_x = tip.x * frame_w
                    pixel_y = tip.y * frame_h
                    screen_x, screen_y = map_to_screen(pixel_x, pixel_y, frame_w, frame_h, screen_w, screen_h)
                    smoothed_x, smoothed_y = smoother.update(np.array([screen_x, screen_y]))
                    pyautogui.moveTo(smoothed_x, smoothed_y, duration=0)
                    cv2.circle(frame, (int(pixel_x), int(pixel_y)), 8, (0, 255, 0), -1)

                # Motion detector runs regardless of pose — it internally
                # ignores pointing pose and only tracks 3/5-finger poses.
                motion_label = motion_detector.update(right_hand_landmarks, finger_count)
                if motion_label is not None:
                    perform_motion_action(motion_label)
                    cv2.putText(frame, f"MOTION: {motion_label}", (10, frame_h - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                motion_detector.reset()

            # ---------------- LEFT HAND: static gesture classifier ----------------
            if left_hand_landmarks is not None:
                landmark_data = extract_and_normalise(left_hand_landmarks)
                try:
                    label, confidence = predict_gesture(landmark_data, hand_label=left_hand_resolved_label)
                except Exception as e:
                    # A bad prediction call should never crash the whole app —
                    # log it once per occurrence and just skip this frame's
                    # gesture classification instead of taking down the demo.
                    label, confidence = None, 0.0
                    cv2.putText(frame, f"Model error: {e}", (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                static_debouncer.update(label)

                if label is not None:
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            else:
                static_debouncer.update(None)

            # ---------------- Debug overlay ----------------
            hand_debug = f"R:{right_label_seen or '-'}  L:{left_label_seen or '-'}"
            cv2.putText(frame, hand_debug, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            now = time.time()
            fps_history.append(1.0 / max(now - prev_time, 1e-6))
            prev_time = now
            fps = sum(fps_history) / len(fps_history)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Hand Gesture Controller", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()


if __name__ == "__main__":
    main()
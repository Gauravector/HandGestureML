# =============================================================================
# control.py
# PURPOSE: Live webcam loop — detects your hand, predicts which of the 14
#          gestures you're holding, and triggers the mapped OS action.
# HOW TO RUN: python src/control.py
# CONTROLS:  Press Q to quit
#
# REQUIRES: data/gesture_model.joblib (produced by train_model.py)
#           hand_landmarker.task (auto-downloaded by collect_data.py earlier)
# =============================================================================

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import numpy as np
import joblib
import time
import os

from gesture_actions import GESTURE_ACTIONS, no_action

MODEL_FILE = os.path.join('data', 'gesture_model.joblib')
HAND_MODEL_PATH = 'hand_landmarker.task'

# ---------------------------------------------------------------------------
# TUNING KNOBS
# ---------------------------------------------------------------------------
CONFIDENCE_THRESHOLD = 0.75   # ignore predictions the model isn't confident about
STABLE_FRAMES_NEEDED = 8      # must predict the SAME gesture this many frames running
                              # before it counts as a real hold (filters out flicker
                              # between poses while your hand is transitioning)
ACTION_COOLDOWN = 1.0         # seconds before the SAME gesture can re-trigger its action
                              # (stops "volume up" from firing every single frame
                              # while you keep holding thumbs_up)


def load_model():
    if not os.path.isfile(MODEL_FILE):
        raise FileNotFoundError(
            f"{MODEL_FILE} not found — run `python src/train_model.py` first."
        )
    data = joblib.load(MODEL_FILE)
    return data['model']


def extract_and_normalise(hand_landmarks):
    """Same normalisation used in collect_data.py — must match exactly,
    otherwise the model sees different-shaped input than it was trained on."""
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks])
    wrist = raw[0]
    normalised = raw - wrist
    max_val = np.max(np.abs(normalised))
    if max_val > 0:
        normalised = normalised / max_val
    return normalised.flatten().tolist()


def main():
    clf = load_model()

    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    last_prediction = None
    stable_count = 0
    last_trigger_time = 0.0
    last_triggered_label = None

    print("[READY] Gesture control running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(image)

        display_text = "No hand detected"
        display_color = (0, 0, 255)

        if result.hand_landmarks:
            hand_lm = result.hand_landmarks[0]
            features = extract_and_normalise(hand_lm)

            probs = clf.predict_proba([features])[0]
            best_idx = int(np.argmax(probs))
            pred_label = clf.classes_[best_idx]
            confidence = probs[best_idx]

            if confidence >= CONFIDENCE_THRESHOLD:
                display_text = f"{pred_label} ({confidence:.0%})"
                display_color = (0, 255, 0)

                if pred_label == last_prediction:
                    stable_count += 1
                else:
                    stable_count = 1
                    last_prediction = pred_label

                now = time.time()
                is_stable = stable_count >= STABLE_FRAMES_NEEDED
                can_retrigger = (
                    pred_label != last_triggered_label
                    or (now - last_trigger_time) > ACTION_COOLDOWN
                )

                if is_stable and can_retrigger:
                    action = GESTURE_ACTIONS.get(pred_label, no_action)
                    action()
                    print(f"[ACTION] Triggered: {pred_label}")
                    last_trigger_time = now
                    last_triggered_label = pred_label
            else:
                # low confidence — reset the stability counter so a
                # half-formed gesture doesn't accidentally count as a hold
                stable_count = 0
                last_prediction = None
                display_text = f"Unsure ({confidence:.0%})"
                display_color = (0, 165, 255)

        cv2.putText(frame, display_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, display_color, 2)
        cv2.imshow("Gesture Control", frame)

        if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
            print("\n[QUIT] Exiting gesture control.")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
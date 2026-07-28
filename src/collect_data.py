# =============================================================================
# collect_data.py  —  compatible with mediapipe 0.10.x (Tasks API)
# PURPOSE: Open webcam, detect hand landmarks + which hand it is, and save
#          labelled STATIC gesture data to a CSV for ML training.
#
# NOTE: This script is only for the 7 STATIC left-hand gestures + idle.
#       The 4 MOTION gestures (swipes/scroll/screenshot) are NOT trained
#       here — they're detected with movement-tracking logic directly in
#       inference.py later, since a single frame can't capture motion.
#
# HOW TO RUN: python src/collect_data.py   (run from project root folder)
# CONTROLS:   Press 1-8 to record the gesture you're currently holding
#             Press Q to quit
# =============================================================================

import cv2
import mediapipe as mp
import csv
import os
import urllib.request
import numpy as np

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================

# Your 7 static left-hand gestures + an "idle" class.
# IDLE IS CRITICAL: without it, the model must guess one of the 7 real
# gestures every frame, even when your hand is just relaxed/doing nothing —
# this causes constant false-trigger clicks, minimizes, etc. once deployed.
GESTURE_LABELS = {
    '1': 'thumbs_up',        # left click (when right hand is pointing)
    '2': 'thumbs_down',      # right click
    '3': 'fist',              # play/pause video
    '4': 'peace',              # sleep
    '5': 'clock_minimize',     # index + middle + thumb pinched together
    '6': 'clock_maximize',     # index + middle + ring + thumb pinched together
    '7': 'clock_close',        # ring finger + thumb pinched together
    '8': 'idle',                # relaxed hand, no active gesture — IMPORTANT
}

OUTPUT_FILE = os.path.join('data', 'gesture_data.csv')
SAMPLES_PER_PRESS = 30

MODEL_PATH = 'hand_landmarker.task'
MODEL_URL  = (
    'https://storage.googleapis.com/mediapipe-models/'
    'hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
)


# =============================================================================
# SECTION 2: MODEL DOWNLOAD (same as before, one-time)
# =============================================================================

def download_model_if_needed():
    if os.path.isfile(MODEL_PATH):
        print(f"[INFO] Model file found: {MODEL_PATH}")
        return
    print(f"[INFO] Downloading hand landmark model (~28MB) — one-time setup...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[INFO] Model saved to: {MODEL_PATH}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        raise


# =============================================================================
# SECTION 3: MEDIAPIPE SETUP
# num_hands=1 because during data collection you're only recording gestures
# for ONE hand at a time (your left hand doing static gestures). Right-hand
# pointing doesn't need any training data at all — it's tracked live later.
# =============================================================================

def create_hand_detector():
    options = HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


# =============================================================================
# SECTION 4: CSV SETUP
# Schema: x0,y0,z0 ... x20,y20,z20 (63 landmark columns) + hand + label = 65
# The "hand" column is now AUTO-DETECTED from MediaPipe's handedness output,
# not hardcoded — so this script stays correct even if you later also
# collect right-hand static gestures.
# =============================================================================

def get_csv_header():
    header = []
    for i in range(21):
        header.extend([f'x{i}', f'y{i}', f'z{i}'])
    header.append('hand')     # 'Left' or 'Right', from MediaPipe itself
    header.append('label')    # gesture name
    return header

def setup_csv():
    os.makedirs('data', exist_ok=True)
    file_exists = os.path.isfile(OUTPUT_FILE)
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(get_csv_header())
            print(f"[INFO] Created new CSV: {OUTPUT_FILE}")
        else:
            print(f"[INFO] Appending to existing CSV: {OUTPUT_FILE}")


# =============================================================================
# SECTION 5: LANDMARK EXTRACTION + NORMALISATION (unchanged logic)
# =============================================================================

def extract_and_normalise(hand_landmarks_list):
    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks_list])
    wrist = raw[0]
    normalised = raw - wrist
    max_val = np.max(np.abs(normalised))
    if max_val > 0:
        normalised = normalised / max_val
    return normalised.flatten().tolist()


def get_handedness_label(result):
    """
    MediaPipe's HandLandmarker also classifies WHICH hand it sees (Left/Right)
    with its own confidence score. result.handedness is a list (one entry
    per detected hand); each entry is a list of Category objects — we take
    the top prediction's .category_name, which is 'Left' or 'Right'.

    IMPORTANT MIRROR NOTE: we flip the frame horizontally for a natural
    mirror view (see main loop). This means MediaPipe sees a mirrored image,
    so its 'Left'/'Right' label is already flipped to match what YOU see in
    the mirror — i.e. if MediaPipe says 'Left', that's the hand appearing
    on the left side of your screen, which is actually your right hand.
    We correct for this below so the saved label matches your ANATOMICAL hand.
    """
    if not result.handedness:
        return 'unknown'

    top_category = result.handedness[0][0]   # first hand, top-confidence guess
    mirrored_label = top_category.category_name   # 'Left' or 'Right'

    # Correct for the horizontal flip applied to the frame for mirror view
    actual_hand = 'Right' if mirrored_label == 'Left' else 'Left'
    return actual_hand


# =============================================================================
# SECTION 6: SAVING DATA
# =============================================================================

def save_samples(landmark_data, hand, label, num_samples):
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        for _ in range(num_samples):
            row = landmark_data + [hand, label]
            writer.writerow(row)
    print(f"[SAVED] {num_samples} rows -> hand='{hand}', label='{label}'  |  Total: {count_existing_samples()}")


# =============================================================================
# SECTION 7: DRAWING HELPERS
# =============================================================================

def draw_landmarks_manual(frame, hand_landmarks_list):
    h, w = frame.shape[:2]
    points = []
    for lm in hand_landmarks_list:
        cx, cy = int(lm.x * w), int(lm.y * h)
        points.append((cx, cy))
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
    connections = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]
    for start, end in connections:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 1)

def draw_instructions(frame):
    y = 30
    cv2.putText(frame, "Hold gesture + press key:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    y += 22
    for key, label in GESTURE_LABELS.items():
        cv2.putText(frame, f"  [{key}] {label}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
    cv2.putText(frame, "  [Q] Quit", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

def draw_feedback(frame, message, color=(200, 200, 200)):
    h = frame.shape[0]
    cv2.putText(frame, message, (10, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

def draw_detected_hand(frame, hand_label):
    """Shows which hand MediaPipe currently detects, top-right area."""
    w = frame.shape[1]
    text = f"Hand: {hand_label}" if hand_label else "Hand: none detected"
    cv2.putText(frame, text, (w - 220, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

def draw_sample_count(frame, count):
    w = frame.shape[1]
    cv2.putText(frame, f"Samples: {count}", (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

def count_existing_samples():
    if not os.path.isfile(OUTPUT_FILE):
        return 0
    with open(OUTPUT_FILE, 'r') as f:
        return max(0, sum(1 for _ in f) - 1)


# =============================================================================
# SECTION 8: MAIN LOOP
# =============================================================================

def main():
    download_model_if_needed()
    setup_csv()
    detector = create_hand_detector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    print("\n[READY] Webcam open!")
    print("        Hold your LEFT hand up for these static gestures.")
    print("        Aim for 300-400 samples per gesture (press key ~10-14 times each).")
    print("        DON'T FORGET the 'idle' class (key 8) — hold a relaxed hand and")
    print("        record it just as much as the real gestures. It matters a lot.\n")

    feedback_msg   = "Show your hand, then press 1-8"
    feedback_color = (200, 200, 200)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Lost webcam feed.")
            break

        frame = cv2.flip(frame, 1)   # mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = detector.detect(mp_image)

        landmark_data = None
        detected_hand = None

        if result.hand_landmarks:
            hand_lm_list = result.hand_landmarks[0]
            draw_landmarks_manual(frame, hand_lm_list)
            landmark_data = extract_and_normalise(hand_lm_list)
            detected_hand = get_handedness_label(result)

        draw_instructions(frame)
        draw_feedback(frame, feedback_msg, feedback_color)
        draw_sample_count(frame, count_existing_samples())
        draw_detected_hand(frame, detected_hand)

        cv2.imshow("Gesture Data Collector — press 1-8 to record, Q to quit", frame)

        key = cv2.waitKey(1) & 0xFF
        key_char = chr(key) if key != 255 else ''

        if key_char.lower() == 'q':
            print(f"\n[QUIT] Total samples: {count_existing_samples()}")
            print(f"       Saved at: {OUTPUT_FILE}")
            break

        elif key_char in GESTURE_LABELS:
            if landmark_data is not None:
                label = GESTURE_LABELS[key_char]
                save_samples(landmark_data, detected_hand, label, SAMPLES_PER_PRESS)
                feedback_msg   = f"Saved {SAMPLES_PER_PRESS}x '{label}' ({detected_hand})"
                feedback_color = (0, 255, 0)
            else:
                feedback_msg   = "No hand detected — move your hand into frame!"
                feedback_color = (0, 0, 255)

    cap.release()
    cv2.destroyAllWindows()
    detector.close()


if __name__ == '__main__':
    main()
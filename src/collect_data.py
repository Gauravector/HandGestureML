# =============================================================================
# collect_data.py
# PURPOSE: Open webcam, detect hand landmarks using MediaPipe, and save
#          labeled gesture data to a CSV file for ML training later.
# HOW TO RUN: python src/collect_data.py
# CONTROLS:  Press 1-5 to record the gesture you're currently holding
#            Press Q to quit
# =============================================================================

import cv2                          # OpenCV  — handles webcam feed and drawing on screen
import mediapipe as mp              # MediaPipe — Google's library that detects hand landmarks
import csv                          # csv — built-in Python module to write data into CSV files
import os                           # os — built-in module to handle file paths and folders
import numpy as np                  # NumPy — for fast math operations on arrays of numbers


# =============================================================================
# SECTION 1: CONFIGURATION
# Edit these to match your project's gestures and file paths
# =============================================================================

# The 5 gestures your model will learn to recognise.
# Key = the keyboard key you press, Value = the gesture name saved in CSV
GESTURE_LABELS = {
    '1': 'open_palm',      # all fingers spread open
    '2': 'fist',           # all fingers closed
    '3': 'peace',          # index + middle finger up (V sign)
    '4': 'thumbs_up',      # only thumb pointing up
    '5': 'pointing',       # only index finger pointing
}

# Where the collected data gets saved.
# os.path.join builds the correct path whether you're on Windows, Mac, or Linux
OUTPUT_FILE = os.path.join('data', 'gesture_data.csv')

# How many landmark samples to collect per keypress.
# Every time you hold a gesture and press a key, this many rows get saved.
# 1 row = one "snapshot" of your 63 landmark values at that moment.
SAMPLES_PER_PRESS = 30


# =============================================================================
# SECTION 2: MEDIAPIPE SETUP
# MediaPipe's Hands module does the heavy lifting — it finds your hand in the
# webcam frame and gives back 21 landmark points (each with x, y, z values).
# =============================================================================

# Replace Section 2 (MediaPipe setup) with this:
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import urllib.request
import os

HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17), (0, 5)
)

# Download the hand landmarker model file (only needed once)
MODEL_PATH = 'hand_landmarker.task'
if not os.path.exists(MODEL_PATH):
    print("[INFO] Downloading hand landmarker model...")
    urllib.request.urlretrieve(
        'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        MODEL_PATH
    )
    print("[INFO] Model downloaded.")

# Create the landmarker
options = HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
landmarker = HandLandmarker.create_from_options(options)

# =============================================================================
# SECTION 3: CSV FILE SETUP
# We need 63 column names (x0,y0,z0 ... x20,y20,z20) + a "label" column.
# MediaPipe gives 21 landmarks, each with 3 values (x, y, z) = 21×3 = 63 values.
# =============================================================================

def get_csv_header():
    """
    Builds the list of column names for the CSV file.
    Returns: ['x0','y0','z0', 'x1','y1','z1', ... 'x20','y20','z20', 'label']
    That's 63 coordinate columns + 1 label column = 64 columns total.
    """
    header = []
    for i in range(21):                      # landmarks are numbered 0 to 20
        header.extend([f'x{i}', f'y{i}', f'z{i}'])   # add x, y, z for each
    header.append('label')                   # last column is the gesture name
    return header

def setup_csv():
    """
    Creates the CSV file with a header row if it doesn't already exist.
    If the file exists (from a previous session), we just append to it — no data lost.
    """
    os.makedirs('data', exist_ok=True)       # create the data/ folder if it doesn't exist

    file_exists = os.path.isfile(OUTPUT_FILE)

    # 'a' mode = append mode (adds rows without overwriting existing data)
    # newline='' is required on Windows to prevent blank lines between rows
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(get_csv_header())    # only write header once
            print(f"[INFO] Created new file: {OUTPUT_FILE}")
        else:
            print(f"[INFO] Appending to existing file: {OUTPUT_FILE}")


# =============================================================================
# SECTION 4: LANDMARK EXTRACTION + NORMALISATION
# Raw landmark values from MediaPipe are in "relative" coordinates —
# x and y are fractions of frame width/height (0.0 to 1.0), z is depth.
# We normalise them so the gesture looks the same regardless of where your
# hand is on screen or how far it is from the camera.
# =============================================================================

def extract_and_normalise(hand_landmarks):
    """
    Takes the 21 raw landmark objects from MediaPipe and returns a flat list
    of 63 normalised numbers ready to be saved as one CSV row.

    Normalisation steps:
      1. Subtract the wrist (landmark 0) position from all other landmarks
         → makes the gesture position-independent (doesn't matter where on
           screen your hand is)
      2. Divide by the max absolute value in the list
         → makes the gesture scale-independent (doesn't matter how close
           your hand is to the camera)
    """
    # Support both the older MediaPipe solutions API (object with .landmark)
    # and the newer HandLandmarker Tasks API (plain list of landmarks).
    landmark_list = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks

    # Step 1: Extract raw x, y, z for all 21 landmarks into a NumPy array
    # Shape: (21, 3) — 21 rows, 3 columns (x, y, z)
    raw = np.array([[lm.x, lm.y, lm.z] for lm in landmark_list])

    # Step 2: Subtract wrist position (landmark index 0)
    # raw[0] is the wrist. After this, wrist becomes (0, 0, 0) and every
    # other landmark is expressed relative to it.
    wrist = raw[0]
    normalised = raw - wrist

    # Step 3: Scale so the largest value in the whole array becomes 1.0
    # This removes the effect of hand size / distance from camera
    max_val = np.max(np.abs(normalised))
    if max_val > 0:                          # avoid division by zero
        normalised = normalised / max_val

    # Step 4: Flatten from shape (21, 3) to a plain list of 63 numbers
    # [x0, y0, z0, x1, y1, z1, ..., x20, y20, z20]
    return normalised.flatten().tolist()


# =============================================================================
# SECTION 5: SAVING DATA TO CSV
# =============================================================================

def save_samples(landmark_data, label, num_samples):
    """
    Saves `num_samples` identical rows to the CSV, all with the given label.
    In a real pipeline you'd vary the data, but for a webcam stream this is fine
    because tiny natural hand movements give you slightly different values each time.

    landmark_data : list of 63 floats (the normalised coordinates)
    label         : string like 'fist' or 'peace'
    num_samples   : how many rows to write (default: SAMPLES_PER_PRESS)
    """
    with open(OUTPUT_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        for _ in range(num_samples):
            row = landmark_data + [label]    # 63 numbers + the gesture name
            writer.writerow(row)
    print(f"[SAVED] {num_samples} samples saved for gesture: '{label}'")


# =============================================================================
# SECTION 6: DISPLAY HELPERS
# Functions that draw text and instructions onto the webcam frame
# so whoever is running the script knows what to do.
# =============================================================================

def draw_instructions(frame):
    """Draws the key legend in the top-left corner of the frame."""
    y = 30
    cv2.putText(frame, "Press key to record gesture:", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    y += 25
    for key, label in GESTURE_LABELS.items():
        cv2.putText(frame, f"  [{key}] {label}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += 22
    cv2.putText(frame, "  [Q] Quit", (10, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 100, 255), 1)

def draw_feedback(frame, message, color=(0, 255, 0)):
    """Draws a feedback message at the bottom of the frame."""
    h, w = frame.shape[:2]                  # frame.shape = (height, width, channels)
    cv2.putText(frame, message, (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def draw_landmarks_on_frame(frame, hand_landmarks):
    """Draws hand landmarks and their connections directly with OpenCV."""
    h, w = frame.shape[:2]
    landmark_list = hand_landmarks.landmark if hasattr(hand_landmarks, 'landmark') else hand_landmarks

    for start, end in HAND_CONNECTIONS:
        start_pt = landmark_list[start]
        end_pt = landmark_list[end]
        cv2.line(
            frame,
            (int(start_pt.x * w), int(start_pt.y * h)),
            (int(end_pt.x * w), int(end_pt.y * h)),
            (0, 255, 0),
            2
        )

    for landmark in landmark_list:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)


def draw_landmark_count(frame, count):
    """Shows how many samples are already in the CSV (top-right corner)."""
    h, w = frame.shape[:2]
    cv2.putText(frame, f"Total samples: {count}", (w - 220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

def count_existing_samples():
    """Counts how many data rows already exist in the CSV (excluding header)."""
    if not os.path.isfile(OUTPUT_FILE):
        return 0
    with open(OUTPUT_FILE, 'r') as f:
        return max(0, sum(1 for _ in f) - 1)   # subtract 1 for the header row


# =============================================================================
# SECTION 7: MAIN LOOP
# This is where everything connects. We open the webcam, read frames in a loop,
# run MediaPipe on each frame, draw landmarks, and listen for keypresses.
# =============================================================================

def main():
    setup_csv()                          # make sure CSV + data/ folder exist

    cap = cv2.VideoCapture(0)            # 0 = default webcam. Try 1 if you have multiple cameras.

    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Check if another app is using it.")
        return

    print("\n[READY] Webcam open. Hold a gesture and press the matching key.")
    print("        Collect at least 200–300 samples per gesture for good accuracy.\n")

    feedback_message = "Hold a gesture, then press 1–5"
    feedback_color   = (200, 200, 200)

    while True:
        # --- Read one frame from webcam ---
        ret, frame = cap.read()          # ret = True if frame was read successfully
        if not ret:
            print("[ERROR] Failed to read from webcam.")
            break

        # --- Flip horizontally so it acts like a mirror ---
        # Without this, moving your right hand goes left on screen — confusing!
        frame = cv2.flip(frame, 1)

        # --- Convert colour space for MediaPipe ---
        # OpenCV reads frames in BGR (Blue-Green-Red) but MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Run MediaPipe hand detection ---
        # In the latest MediaPipe Tasks API, detection is done with the
        # HandLandmarker object instead of the older hands_detector.process() flow.
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = landmarker.detect(image)

        landmark_data = None             # will hold the 63 numbers if a hand is found

        if result.hand_landmarks:
            # We set num_hands=1 in the landmarker options, so there is at most one item.
            hand_lm = result.hand_landmarks[0]

            # Draw the hand landmarks directly with OpenCV so the code stays
            # compatible with the newer MediaPipe Tasks API.
            draw_landmarks_on_frame(frame, hand_lm)

            # Extract and normalise the 63 coordinate values
            landmark_data = extract_and_normalise(hand_lm)

        # --- Draw UI elements onto the frame ---
        draw_instructions(frame)
        draw_feedback(frame, feedback_message, feedback_color)
        draw_landmark_count(frame, count_existing_samples())

        # --- Show the frame in a window ---
        cv2.imshow("Gesture Data Collector — Hand Gesture Controller", frame)

        # --- Listen for keypresses (waitKey waits 1ms between frames) ---
        # cv2.waitKey returns the ASCII code of the key pressed, or -1 if none
        key = cv2.waitKey(1) & 0xFF      # & 0xFF masks to 8 bits (handles some OS quirks)
        key_char = chr(key) if key != 255 else ''   # convert ASCII code to character

        if key_char == 'q' or key_char == 'Q':
            print("\n[QUIT] Exiting. Your data is saved in:", OUTPUT_FILE)
            break

        elif key_char in GESTURE_LABELS:
            if landmark_data is not None:
                label = GESTURE_LABELS[key_char]
                save_samples(landmark_data, label, SAMPLES_PER_PRESS)
                feedback_message = f"Recorded {SAMPLES_PER_PRESS}x '{label}' ✓"
                feedback_color   = (0, 255, 0)       # green = success
            else:
                feedback_message = "No hand detected! Show your hand clearly."
                feedback_color   = (0, 0, 255)       # red = error

    # --- Cleanup ---
    cap.release()                        # release the webcam so other apps can use it
    cv2.destroyAllWindows()              # close all OpenCV windows
    print(f"\n[DONE] Total samples collected: {count_existing_samples()}")


# =============================================================================
# ENTRY POINT
# This block runs only when you execute this file directly (python collect_data.py)
# and NOT when another file imports it. Standard Python convention.
# =============================================================================

if __name__ == '__main__':
    main()
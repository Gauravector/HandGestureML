# =============================================================================
# actions.py
# PURPOSE: Load the trained gesture classifier, predict gestures from live
#          landmark data, and map both STATIC gestures (classifier-based) and
#          MOTION gestures (movement-based, detected elsewhere) to real OS
#          actions. Windows-only (uses Win-key shortcuts).
#
# HOW TO USE:
#   from actions import predict_gesture, perform_static_action, perform_motion_action
#
#   label, confidence = predict_gesture(landmark_data)   # from classifier
#   if label:
#       perform_static_action(label)
#
#   perform_motion_action('swipe_left')                  # from motion tracker
# =============================================================================

import joblib               # loads the saved .pkl model and label encoder
import numpy as np          # needed to reshape landmark data for prediction
import pyautogui             # simulates clicks, scrolling, screenshots, key presses
from pynput.keyboard import Key, Controller as KeyboardController
import os
from datetime import datetime


# =============================================================================
# SECTION 1: LOAD THE TRAINED MODEL
# These load once when this file is first imported, not on every prediction —
# loading a .pkl file from disk is relatively slow, so we do it once and reuse
# the loaded objects for every frame during inference.
# =============================================================================

MODEL_PATH   = os.path.join('models', 'gesture_model_rf.pkl')
ENCODER_PATH = os.path.join('models', 'label_encoder.pkl')

# The trained Random Forest classifier — takes 63 landmark numbers, predicts
# a gesture class (as a number, since sklearn works with numeric labels).
model = joblib.load(MODEL_PATH)

# The label encoder converts between text labels ('fist', 'thumbs_up', ...)
# and the numbers the model actually predicts internally. We need this to
# turn the model's numeric output back into a gesture name we can use.
label_encoder = joblib.load(ENCODER_PATH)

print(f"[INFO] Loaded model from {MODEL_PATH}")
print(f"[INFO] Known gesture classes: {list(label_encoder.classes_)}")


# =============================================================================
# SECTION 2: CONFIDENCE THRESHOLD
# A prediction is only trusted if the model is at least this confident.
# Random Forest gives a probability for each class via predict_proba() —
# if the top probability is below this threshold, we treat it as "no gesture"
# rather than risk acting on a shaky guess. This is what prevents random
# hand movements from misfiring as clicks, minimizes, etc.
# =============================================================================

CONFIDENCE_THRESHOLD = 0.85


def predict_gesture(landmark_data, hand_label='Left'):
    """
    Takes a list of 63 normalised landmark floats (same format your
    collect_data.py saves) plus which hand this is, and returns
    (label, confidence).

    hand_label : 'Left' or 'Right' — the ACTUAL physical hand (after any
                 handedness correction has already been applied by the
                 caller). The model was trained with a 64th feature,
                 'hand_is_right' (1.0 if right hand, 0.0 if left), so we
                 have to reproduce that exact feature here or every
                 prediction will error out or be wrong.

    If the model isn't confident enough, returns (None, confidence) so the
    calling code can treat it as "no gesture detected" instead of acting
    on an unreliable guess.

    Usage in inference.py:
        label, confidence = predict_gesture(landmark_data, hand_label='Left')
        if label is not None:
            perform_static_action(label)
    """
    hand_is_right = 1.0 if hand_label == 'Right' else 0.0

    # Build the full 64-value feature row: 63 landmark values + hand_is_right,
    # in EXACTLY the order the model was trained on (confirmed via
    # model.feature_names_in_: x0,y0,z0 ... x20,y20,z20, hand_is_right).
    full_features = landmark_data + [hand_is_right]

    # sklearn expects a 2D array: (n_samples, n_features).
    # We have 1 sample with 64 features, so reshape from (64,) to (1, 64).
    X = np.array(full_features).reshape(1, -1)

    # predict_proba returns probabilities for every class, e.g.
    # [0.02, 0.01, 0.90, 0.01, 0.02, 0.01, 0.01, 0.02] for 8 classes.
    # We only care about the highest one and which class it belongs to.
    probabilities = model.predict_proba(X)[0]

    best_index = np.argmax(probabilities)        # index of the highest probability
    confidence = probabilities[best_index]         # the probability value itself

    if confidence < CONFIDENCE_THRESHOLD:
        return None, confidence     # not confident enough — treat as no gesture

    # Convert the numeric class index back to a text label like 'fist'.
    # inverse_transform expects a list/array, so we wrap best_index in [ ].
    label = label_encoder.inverse_transform([best_index])[0]

    return label, confidence


# =============================================================================
# SECTION 3: SETUP FOR OS ACTIONS
# =============================================================================

keyboard = KeyboardController()

SCREENSHOT_FOLDER = os.path.join('screenshots')
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

# Safety: move mouse to top-left corner of screen to force-stop pyautogui
# if a gesture misfires and starts doing something unwanted.
pyautogui.FAILSAFE = True


# =============================================================================
# SECTION 4: STATIC GESTURE ACTIONS
# One function per static gesture your classifier recognises. Each does
# exactly one thing so you can test them individually.
# =============================================================================

def left_click():
    """Left mouse click at the current cursor position (set by index-finger tracking)."""
    pyautogui.click(button='left')
    print("[ACTION] Left click")


def right_click():
    """Right mouse click at the current cursor position."""
    pyautogui.click(button='right')
    print("[ACTION] Right click")


def play_pause_media():
    """
    Sends the dedicated media play/pause key, which works with most video
    players and browsers (YouTube, Spotify, VLC) without needing to know
    which app is focused.
    """
    keyboard.press(Key.media_play_pause)
    keyboard.release(Key.media_play_pause)
    print("[ACTION] Play/Pause media")


def sleep_pc():
    """
    Puts the PC to sleep via the standard Windows shell command.
    0,1,0 = don't force close apps, sleep (not hibernate), don't prompt.
    """
    print("[ACTION] Putting PC to sleep...")
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")


def minimize_window():
    """Minimizes the currently active window using Win+Down."""
    with keyboard.pressed(Key.cmd):
        keyboard.press(Key.down)
        keyboard.release(Key.down)
    print("[ACTION] Minimize window")


def maximize_window():
    """Maximizes the currently active window using Win+Up."""
    with keyboard.pressed(Key.cmd):
        keyboard.press(Key.up)
        keyboard.release(Key.up)
    print("[ACTION] Maximize window")


def close_window():
    """Closes the currently active window using Alt+F4."""
    with keyboard.pressed(Key.alt):
        keyboard.press(Key.f4)
        keyboard.release(Key.f4)
    print("[ACTION] Close window")


def do_nothing():
    """
    The action for the 'idle' gesture — deliberately does nothing.
    Having idle explicitly mapped (instead of just missing from the dict)
    makes it clear this is intentional, not a bug or an unmapped gesture.
    """
    pass   # no-op, on purpose


# Maps each STATIC gesture label (from your classifier) to its action.
# These keys must exactly match the class names inside label_encoder.classes_
# printed when this file loads — check that output matches this dict exactly.
STATIC_GESTURE_ACTION_MAP = {
    'thumbs_up':       left_click,
    'thumbs_down':      right_click,
    'fist':              play_pause_media,
    'peace':              sleep_pc,
    'clock_minimize':      minimize_window,
    'clock_maximize':       maximize_window,
    'clock_close':           close_window,
    'idle':                   do_nothing,
}


def perform_static_action(gesture_label):
    """
    Looks up a static gesture label in STATIC_GESTURE_ACTION_MAP and runs it.
    Returns True if an action was found and run, False otherwise — so
    inference.py can log/handle unrecognised labels without crashing.
    """
    action_fn = STATIC_GESTURE_ACTION_MAP.get(gesture_label)

    if action_fn is None:
        print(f"[WARNING] No action mapped for static gesture: '{gesture_label}'")
        return False

    action_fn()
    return True


# =============================================================================
# SECTION 5: MOTION GESTURE ACTIONS
# These are NOT predicted by the classifier — they're detected by tracking
# hand position/pose over several frames (built separately in inference.py's
# motion-tracking module). This dict just defines what happens once a motion
# gesture IS detected, keeping that logic separate from how it's detected.
# =============================================================================

def take_screenshot():
    """Captures the full screen and saves it with a timestamped filename."""
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(SCREENSHOT_FOLDER, f'screenshot_{timestamp}.png')
    screenshot = pyautogui.screenshot()
    screenshot.save(filename)
    print(f"[ACTION] Screenshot saved: {filename}")


def undo():
    """Sends Ctrl+Z."""
    with keyboard.pressed(Key.ctrl):
        keyboard.press('z')
        keyboard.release('z')
    print("[ACTION] Undo (Ctrl+Z)")


def redo():
    """Sends Ctrl+Y (standard redo on Windows)."""
    with keyboard.pressed(Key.ctrl):
        keyboard.press('y')
        keyboard.release('y')
    print("[ACTION] Redo (Ctrl+Y)")


def scroll_up():
    """Scrolls up. Positive value = scroll up direction."""
    pyautogui.scroll(100)
    print("[ACTION] Scroll up")


def scroll_down():
    """Scrolls down. Negative value = scroll down direction."""
    pyautogui.scroll(-100)
    print("[ACTION] Scroll down")


# Maps motion gesture NAMES (strings your motion-tracking code in inference.py
# will produce, e.g. after detecting a leftward swipe) to their actions.
# Update the keys here to exactly match whatever strings your motion
# detector outputs once that module is written.
MOTION_GESTURE_ACTION_MAP = {
    'swipe_left':          undo,
    'swipe_right':          redo,
    'swipe_down_3finger':    take_screenshot,
    'swipe_up':               scroll_up,
    'swipe_down':              scroll_down,
}


def perform_motion_action(motion_label):
    """
    Looks up a motion gesture label in MOTION_GESTURE_ACTION_MAP and runs it.
    Called by inference.py once its movement-tracking logic decides a swipe
    happened — separate from perform_static_action() since motion gestures
    never go through the classifier.
    """
    action_fn = MOTION_GESTURE_ACTION_MAP.get(motion_label)

    if action_fn is None:
        print(f"[WARNING] No action mapped for motion gesture: '{motion_label}'")
        return False

    action_fn()
    return True


# =============================================================================
# MANUAL TEST BLOCK
# Tests the model loads correctly and every action fires, without needing
# the webcam or real landmark data. sleep_pc is commented out by default —
# uncomment carefully only if you're ready for your PC to actually sleep.
# =============================================================================

if __name__ == '__main__':
    import time

    print("\n[TEST] Verifying model + label encoder loaded correctly...")
    print(f"       Model type: {type(model).__name__}")
    print(f"       Classes: {list(label_encoder.classes_)}\n")

    print("[TEST] Running through each static action with a 2 second gap...")
    print("[WARNING] sleep_pc is commented out by default.\n")

    static_test_actions = [
        left_click,
        right_click,
        play_pause_media,
        minimize_window,
        maximize_window,
        close_window,
        do_nothing,
        # sleep_pc,   # uncomment ONLY if you're ready for your PC to sleep
    ]

    for action in static_test_actions:
        action()
        time.sleep(2)

    print("\n[TEST] Running through each motion action with a 2 second gap...\n")

    motion_test_actions = [
        undo,
        redo,
        take_screenshot,
        scroll_up,
        scroll_down,
    ]

    for action in motion_test_actions:
        action()
        time.sleep(2)

    print("\n[TEST COMPLETE]")
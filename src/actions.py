# =============================================================================
# actions.py
# PURPOSE: Map gesture label strings (output by your trained classifier) to
#          real OS actions — clicks, scrolling, screenshots, window controls,
#          sleep, and shutdown. Windows-only (uses Win-key shortcuts).
# HOW TO USE: from actions import perform_action
#             perform_action("screenshot")
# =============================================================================

import pyautogui          # simulates mouse clicks, scrolling, screenshots, key presses
import pynput              # lower-level keyboard control, used for special key combos
from pynput.keyboard import Key, Controller as KeyboardController
import time                 # built-in — used for the shutdown confirmation delay
import os                   # built-in — file paths for saving screenshots
from datetime import datetime   # built-in — timestamps for screenshot filenames


# =============================================================================
# SECTION 1: SETUP
# =============================================================================

# pynput's keyboard controller lets us simulate key combos like Win+D
# that pyautogui can't always handle reliably on Windows.
keyboard = KeyboardController()

# Folder where gesture-triggered screenshots get saved.
SCREENSHOT_FOLDER = os.path.join('screenshots')
os.makedirs(SCREENSHOT_FOLDER, exist_ok=True)

# Safety: pyautogui has a built-in "fail-safe" — if you move the mouse to the
# top-left corner of the screen, it raises an exception and stops.
# This is a lifesaver if a gesture misfires and starts doing something unwanted.
pyautogui.FAILSAFE = True

# How many seconds a shutdown gesture must be held/confirmed before it fires.
# This exists so a single misdetected frame doesn't shut down your PC.
SHUTDOWN_CONFIRM_DELAY = 3.0


# =============================================================================
# SECTION 2: INDIVIDUAL ACTION FUNCTIONS
# Each function does exactly one thing. Keeping them separate means you can
# test each action individually, e.g. call take_screenshot() on its own.
# =============================================================================

def left_click():
    """
    Performs a left mouse click at the CURRENT cursor position.
    Note: cursor position itself is set separately by your teammate's
    inference.py, which tracks the index fingertip. This function just clicks
    wherever the cursor already is.
    """
    pyautogui.click(button='left')
    print("[ACTION] Left click")


def right_click():
    """Performs a right mouse click at the current cursor position."""
    pyautogui.click(button='right')
    print("[ACTION] Right click")


def scroll_up():
    """
    Scrolls up. pyautogui.scroll() takes a positive number to scroll up,
    negative to scroll down. The number is "scroll clicks" not pixels —
    100 is a reasonable amount per gesture trigger.
    """
    pyautogui.scroll(100)
    print("[ACTION] Scroll up")


def scroll_down():
    """Scrolls down. Negative value = scroll down direction."""
    pyautogui.scroll(-100)
    print("[ACTION] Scroll down")


def take_screenshot():
    """
    Captures the full screen and saves it with a timestamp filename so
    repeated screenshots never overwrite each other.
    e.g. screenshots/screenshot_2026-07-28_14-32-05.png
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = os.path.join(SCREENSHOT_FOLDER, f'screenshot_{timestamp}.png')

    screenshot = pyautogui.screenshot()   # captures the entire screen as an image
    screenshot.save(filename)

    print(f"[ACTION] Screenshot saved: {filename}")


def minimize_window():
    """
    Minimizes the currently active window using the Win+Down shortcut.
    pynput lets us hold multiple keys together using a 'with' block —
    both keys are pressed, then both released, simulating a real key combo.
    """
    with keyboard.pressed(Key.cmd):     # Key.cmd = the Windows key
        keyboard.press(Key.down)
        keyboard.release(Key.down)
    print("[ACTION] Minimize window")


def maximize_window():
    """Maximizes the currently active window using Win+Up."""
    with keyboard.pressed(Key.cmd):
        keyboard.press(Key.up)
        keyboard.release(Key.up)
    print("[ACTION] Maximize window")


def switch_window():
    """
    Triggers Alt+Tab to switch between open windows.
    Note: holding Alt+Tab open (like a real user tabbing through a window
    switcher) is more complex — this does a single quick switch to the
    previous window, which is the simplest reliable version for a gesture.
    """
    with keyboard.pressed(Key.alt):
        keyboard.press(Key.tab)
        keyboard.release(Key.tab)
    print("[ACTION] Switch window (Alt+Tab)")


def sleep_pc():
    """
    Puts the PC to sleep using the Windows shell command.
    rundll32.exe with these exact flags is the standard Windows sleep command
    — 0,1,0 means: don't force apps closed, go to sleep (not hibernate), don't prompt.
    """
    print("[ACTION] Putting PC to sleep...")
    os.system("rundll32.exe powrprof.dll,SetSuspendState 0,1,0")


def shutdown_pc(confirmed=False):
    """
    Shuts down the PC — but ONLY if `confirmed=True` is explicitly passed.

    Why the confirmed flag? Shutdown is the most dangerous action in this
    whole system. A single misclassified frame should NEVER be able to shut
    down someone's PC. The calling code (inference.py) must hold/repeat the
    shutdown gesture for SHUTDOWN_CONFIRM_DELAY seconds before calling this
    with confirmed=True. See confirm_and_shutdown() below for that logic.
    """
    if not confirmed:
        print("[BLOCKED] Shutdown requires explicit confirmation. Ignoring.")
        return

    print("[ACTION] Shutting down PC in 5 seconds... (Ctrl+C in terminal to cancel)")
    # /s = shutdown, /t 5 = wait 5 seconds first (gives a last chance to cancel)
    os.system("shutdown /s /t 5")


def confirm_and_shutdown(gesture_held_since):
    """
    Helper for inference.py to call every frame the shutdown gesture is detected.

    gesture_held_since : the timestamp (from time.time()) when this gesture
                          was FIRST detected continuously.

    Returns True and triggers shutdown once the gesture has been held for
    SHUTDOWN_CONFIRM_DELAY seconds straight. Returns False otherwise.

    Usage pattern in inference.py:
        if detected_gesture == 'shutdown_sign':
            if shutdown_start_time is None:
                shutdown_start_time = time.time()
            confirm_and_shutdown(shutdown_start_time)
        else:
            shutdown_start_time = None   # reset if gesture is released early
    """
    held_duration = time.time() - gesture_held_since
    if held_duration >= SHUTDOWN_CONFIRM_DELAY:
        shutdown_pc(confirmed=True)
        return True
    else:
        remaining = SHUTDOWN_CONFIRM_DELAY - held_duration
        print(f"[HOLD] Shutdown gesture held — {remaining:.1f}s more to confirm...")
        return False


# =============================================================================
# SECTION 3: GESTURE-TO-ACTION MAPPING
# This dictionary is the single source of truth connecting gesture label
# strings (whatever your trained model outputs) to the functions above.
#
# IMPORTANT: these keys must EXACTLY match the label strings used in your
# CSV during data collection (collect_data.py). Update this dict once your
# team finalises the 17 gesture names.
# =============================================================================

GESTURE_ACTION_MAP = {
    # --- Right hand gestures ---
    'left_click_pinch':      left_click,
    'right_click_pinch':     right_click,
    'scroll_up_gesture':     scroll_up,
    'scroll_down_gesture':   scroll_down,

    # --- Left hand gestures ---
    'thumbs_up':              left_click,       # per your spec: left thumbs up = left click
    'thumbs_down':             right_click,
    'open_palm_left':          take_screenshot,
    'fist_left':                minimize_window,
    'peace_left':               minimize_window,   # placeholder — update once finalised
    'three_fingers_left':       maximize_window,
    'four_fingers_left':        switch_window,
    'pinky_only_left':          sleep_pc,
    # 'shutdown_sign' is intentionally NOT mapped directly here —
    # it goes through confirm_and_shutdown() in inference.py instead,
    # since it needs the hold-to-confirm safety logic.

    # NOTE: 'index_pointer_right' (cursor movement) is NOT in this map.
    # Cursor movement is continuous tracking, not a discrete action —
    # it's handled directly in inference.py, not through this dictionary.
}


def perform_action(gesture_label):
    """
    The main function other scripts call: given a gesture label string,
    looks it up in GESTURE_ACTION_MAP and runs the matching function.

    Returns True if the action was found and run, False if the label
    doesn't match anything (so calling code can handle unknown gestures
    gracefully instead of crashing).
    """
    action_fn = GESTURE_ACTION_MAP.get(gesture_label)

    if action_fn is None:
        print(f"[WARNING] No action mapped for gesture: '{gesture_label}'")
        return False

    action_fn()   # call the matched function, e.g. left_click()
    return True


# =============================================================================
# MANUAL TEST BLOCK
# Run this file directly to test each action works, without needing the
# webcam or trained model. Comment out any action you don't want to
# accidentally trigger (especially sleep_pc and shutdown_pc!).
# =============================================================================

if __name__ == '__main__':
    print("[TEST] Running through each action with a 2 second gap...\n")
    print("[WARNING] sleep_pc and shutdown_pc are commented out by default.")
    print("          Uncomment carefully if you want to test them.\n")

    test_actions = [
        left_click,
        right_click,
        scroll_up,
        scroll_down,
        take_screenshot,
        minimize_window,
        maximize_window,
        switch_window,
        # sleep_pc,       # uncomment ONLY if you're ready for your PC to sleep
        # shutdown_pc,    # dangerous — better tested via confirm_and_shutdown()
    ]

    for action in test_actions:
        action()
        time.sleep(2)

    print("\n[TEST COMPLETE]")
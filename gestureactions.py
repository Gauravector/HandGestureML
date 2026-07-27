# =============================================================================
# gesture_actions.py
# PURPOSE: Maps each gesture label (the exact strings you used as CSV labels
#          when collecting data) to a real OS media/system action.
#
# SETUP:
#   pip install pyautogui screen-brightness-control
#
# TO SEE YOUR EXACT LABELS, run:
#   python -c "import pandas as pd; print(pd.read_csv('data/gesture_data.csv')['label'].unique())"
#
# Then edit GESTURE_ACTIONS at the bottom of this file to match all 14.
# =============================================================================

import platform
import subprocess
import pyautogui

try:
    import screen_brightness_control as sbc
    HAS_BRIGHTNESS = True
except ImportError:
    HAS_BRIGHTNESS = False

SYSTEM = platform.system()  # 'Windows', 'Darwin' (Mac), or 'Linux'


# -----------------------------------------------------------------------
# ACTION FUNCTIONS
# pyautogui's media keys (volumeup/down/mute, playpause, nexttrack,
# prevtrack) work natively on Windows and macOS. On Linux they're
# unreliable, so volume falls back to `amixer` shell commands there —
# adjust 'Master'/'pulse' below if your system uses a different mixer.
# -----------------------------------------------------------------------

def _linux_volume(amixer_arg):
    subprocess.run(['amixer', '-D', 'pulse', 'sset', 'Master', amixer_arg], check=False)

def volume_up():
    _linux_volume('5%+') if SYSTEM == 'Linux' else pyautogui.press('volumeup')

def volume_down():
    _linux_volume('5%-') if SYSTEM == 'Linux' else pyautogui.press('volumedown')

def volume_mute():
    _linux_volume('toggle') if SYSTEM == 'Linux' else pyautogui.press('volumemute')

def play_pause():
    pyautogui.press('playpause')

def next_track():
    pyautogui.press('nexttrack')

def prev_track():
    pyautogui.press('prevtrack')

def brightness_up():
    if HAS_BRIGHTNESS:
        current = sbc.get_brightness(display=0)[0]
        sbc.set_brightness(min(current + 10, 100))
    else:
        print("[WARN] pip install screen-brightness-control for brightness control")

def brightness_down():
    if HAS_BRIGHTNESS:
        current = sbc.get_brightness(display=0)[0]
        sbc.set_brightness(max(current - 10, 0))
    else:
        print("[WARN] pip install screen-brightness-control for brightness control")

def screenshot():
    pyautogui.screenshot('screenshot.png')
    print("[ACTION] Screenshot saved to screenshot.png")

def lock_screen():
    if SYSTEM == 'Windows':
        subprocess.run(['rundll32.exe', 'user32.dll,LockWorkStation'])
    elif SYSTEM == 'Darwin':
        subprocess.run(['pmset', 'displaysleepnow'])
    else:
        subprocess.run(['loginctl', 'lock-session'])

def open_app_launcher():
    # Simple cross-platform "search everything" style shortcut
    if SYSTEM == 'Darwin':
        pyautogui.hotkey('command', 'space')
    else:
        pyautogui.press('win' if SYSTEM == 'Windows' else 'super')

def minimize_window():
    if SYSTEM == 'Darwin':
        pyautogui.hotkey('command', 'm')
    else:
        pyautogui.hotkey('win', 'down')

def no_action():
    pass  # gesture recognized but intentionally does nothing (e.g. a "neutral/rest" pose)


# -----------------------------------------------------------------------
# EDIT THIS DICTIONARY
# Left side  = exact label string from your CSV (case-sensitive)
# Right side = one of the functions above
#
# Below are placeholders for 5 gestures matching collect_data.py's default
# GESTURE_LABELS, plus commented examples for the other 9. Replace with
# your actual 14 label names and desired actions.
# -----------------------------------------------------------------------
GESTURE_ACTIONS = {
    'open_palm':   play_pause,
    'fist':        volume_mute,
    'peace':       volume_down,
    'thumbs_up':   volume_up,
    'pointing':    next_track,

    # 'thumbs_down':  prev_track,
    # 'ok_sign':      screenshot,
    # 'rock_sign':    brightness_up,
    # 'l_sign':       brightness_down,
    # 'call_me':      lock_screen,
    # 'flat_hand':    no_action,        # good for a "rest" pose you don't want to trigger anything
    # 'crossed_fingers': open_app_launcher,
    # 'pinch':        minimize_window,
    # 'wave':         no_action,
}
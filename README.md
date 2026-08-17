# HandGestureML
ML project for ITSP by Eximius

The problem

Most computer interaction depends on a keyboard and mouse. That's a real barrier for people with motor impairments, and it's inconvenient in hands-free settings like presentations, sterile environments, or AR/VR. Existing gesture-control systems usually need extra hardware (gloves, depth cameras) or are too heavy to run in real time on a normal laptop.

The solution

A gesture-control system that runs on a standard laptop webcam, CPU-only, in real time — no extra hardware. It reads your hand through the camera, figures out what you're doing with each hand, and turns that into real OS actions (moving the cursor, clicking, scrolling, minimizing windows, etc.).

The key design idea — three approaches, not one

Early on we considered training one big classifier for every gesture. We deliberately moved away from that, because different gestures are fundamentally different kinds of information:

Type	Example	How it's handled
Continuous position	Moving the cursor by pointing	Direct coordinate tracking — no ML needed
Static shape	Thumbs up, fist, peace sign	A trained ML classifier (Random Forest)
Motion over time	Swiping left to undo	Rule-based movement tracking across several frames

Trying to force motion gestures (like a swipe) through a single-frame classifier doesn't work — a classifier sees one snapshot at a time and can't represent "movement." That's why the system is split this way instead of using one model for everything.

Final gesture set (12 total)

Right hand — pointing + motion:

Index finger extended → move cursor (continuous tracking)
Open palm, swipe left / right → undo / redo
Open palm, swipe up / down → scroll up / down
3 fingers, swipe down → screenshot

Left hand — static classifier (7 gestures + idle):

Thumbs up → left click
Thumbs down → right click
Fist → play/pause media
Peace sign → sleep
"Clock" pinch (index+middle+thumb) → minimize window
"Clock" pinch (index+middle+ring+thumb) → maximize window
"Clock" pinch (ring+thumb) → close window
Idle (relaxed hand) → does nothing — exists specifically to stop the model guessing a real gesture every single frame

2. Repository / Folder Structure
hand-gesture-controller/
│
├── data/
│   └── gesture_data.csv        ← collected training data (63 landmarks + hand + label per row)
│
├── models/
│   ├── gesture_model_rf.pkl    ← trained Random Forest classifier
│   ├── label_encoder.pkl       ← converts between gesture names and numeric labels
│   └── best_model.txt          ← notes on which model was selected and why
│
├── src/
│   ├── collect_data.py         ← records labelled gesture data from webcam
│   ├── train_model.py          ← trains and evaluates the classifier
│   ├── actions.py              ← loads the model, maps gestures to real OS actions
│   ├── inference.py            ← the final real-time app — ties everything together
│   └── diagnose_model.py       ← small utility to inspect what a saved model expects
│
├── hand_landmarker.task         ← MediaPipe's pretrained hand landmark detection model (downloaded automatically, not written by us)
├── requirements.txt              ← exact package versions everyone installs
└── README.md

3. What Each File Does — Team-Wide Overview

collect_data.py — Data Collection & Preprocessing (Misha)

Opens the webcam, uses MediaPipe to detect a hand and its 21 landmark points (63 x/y/z numbers), and lets you press number keys to save labelled samples to a CSV. Applies normalisation — subtracting the wrist position and scaling — so the same gesture looks the same in the data whether your hand is near or far from the camera, or positioned differently on screen. Also auto-detects which hand (left/right) is shown, using MediaPipe's own handedness classification.

train_model.py — Model Training & Evaluation (Rishi)

Loads the collected CSV, splits it into training and test sets, trains a Random Forest classifier (63 landmark values + a hand_is_right flag = 64 features → 8 gesture classes), evaluates accuracy, and saves the trained model plus a label encoder to models/.

actions.py — Action Layer (Gaurav)

Loads the trained model and label encoder once at startup. Provides predict_gesture(), which takes live landmark data and returns a predicted gesture + confidence score (rejecting low-confidence guesses rather than acting on them). Also holds two dictionaries mapping gesture names to real OS actions — one for static gestures (from the classifier) and one for motion gestures (from the swipe detector) — using pyautogui and pynput to actually click, scroll, minimize windows, etc.

inference.py — Real-Time Integration (Siya, with later fixes)

The actual running application. Opens the webcam, detects both hands every frame, and routes each hand to the right job: right hand → cursor movement (if pointing) or swipe detection (if in an open-palm/3-finger pose); left hand → normalise landmarks → predict_gesture() → perform_static_action(). Includes debouncing (so a held gesture doesn't fire repeatedly) and smoothing (so the cursor doesn't jitter).

diagnose_model.py — Debug Utility

A small script that loads a saved model and prints exactly what features it expects, in what order. Used to fix a real bug where the model expected 64 features but was initially only given 63.

data/gesture_data.csv

The actual training data — one row per recorded sample: 63 landmark coordinates + which hand + the gesture label.

models/gesture_model_rf.pkl and label_encoder.pkl

The trained model itself (saved with joblib) and the encoder that translates between text labels like "fist" and the numbers the model actually works with internally.

4. Key Technical Concepts (Glossary)


Landmark — A single tracked point on the hand (like a knuckle or fingertip). MediaPipe gives 21 per hand, each with an (x, y, z) position — 63 numbers total per hand, per frame.

Normalisation — Adjusting raw coordinates so they represent the shape of a gesture, not its position or size on screen. We subtract the wrist position from every landmark (so it doesn't matter where your hand is on screen) and scale by the largest value (so it doesn't matter how close your hand is to the camera).

Handedness — MediaPipe's own classification of whether a detected hand is "Left" or "Right." This can be affected by whether your webcam feed is mirrored, which varies by hardware — something we had to test and correct for empirically.

Classifier / Random Forest — An ML model made of many decision trees that vote on the answer. Given 64 numbers (landmarks + hand flag), it outputs a probability for each of the 8 possible gestures; we take the highest one.

Confidence threshold — We only trust the model's prediction if its top probability is above 0.85. Below that, we treat it as "no gesture" rather than risk acting on a shaky guess — this is what stops random hand movements from misfiring as clicks.

Debounce — Logic that makes an action fire once per gesture, not once per frame the gesture is held. Without it, holding a fist for 2 seconds at 30 FPS would try to trigger "play/pause" 60 times.

Motion / swipe detection — Since a single frame can't show movement, we track the hand's position across the last several frames and check if it moved far enough in one direction to count as a swipe — separate from the classifier entirely.

.pkl file / joblib — A saved, ready-to-use copy of a trained model, so you don't need to retrain it every time you run the program. joblib.load() reads it back into memory.

5. End-to-End Data Flow
   COLLECTION (once)                    TRAINING (once)                  LIVE USE (every frame)
   ─────────────────                    ───────────────                  ──────────────────────
   collect_data.py                      train_model.py                   inference.py
        │                                     │                                │
   webcam + MediaPipe                   loads gesture_data.csv           webcam + MediaPipe
        │                                     │                                │
   21 landmarks → normalise             trains Random Forest             21 landmarks (both hands)
        │                                     │                                │
   save to gesture_data.csv             saves .pkl model + encoder       ┌─────┴─────┐
                                                    │                     RIGHT hand    LEFT hand
                                                    │                     (position/    (normalise →
                                                    │                      motion)       actions.py
                                                    │                          │        predict_gesture)
                                                    └──────────────────────────┴────────────┘
                                                                    │
                                                              actions.py
                                                                    │
                                                         real OS action fires
                                                    (click, scroll, minimize, etc.)
6. Ownership Reference
   
Rishi Pokar	train_model.py	How the CSV becomes a trained model; what accuracy/confusion matrix mean
Ahir Misha Keyurkumar	collect_data.py	Why normalisation and the idle class matter
Siya Rathore	inference.py	How both hands are tracked simultaneously and routed differently
Gaurav	actions.py	How predictions become real OS actions; the 64-feature fix

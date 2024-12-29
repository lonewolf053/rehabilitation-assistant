import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance
import json
import os
import time

# -------------------- Global Config -------------------- #
MAX_NUM_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.7

# Distance scaling factors for gesture matching
GESTURE_SCALING_FACTOR = 200
FINGER_SCALING_FACTOR = 200

# Number of samples to record for each gesture
NUM_SAMPLES_PER_GESTURE = 3

# Whether to show "ghost" overlay in use_gesture mode
SHOW_GHOST_OVERLAY = True

# -------------------- MediaPipe Setup -------------------- #
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# -------------------- Utility Functions -------------------- #
def load_gestures(filepath='gestures.json'):
    """Load gestures data from a JSON file."""
    if not os.path.exists(filepath):
        return {}
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        st.error(f"[Error] Failed to load gestures from {filepath}: {e}")
        return {}


def save_gestures(gesture_dict, filepath='gestures.json'):
    """Save gestures data to a JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(gesture_dict, f, indent=4)
    except IOError as e:
        st.error(f"[Error] Failed to save gestures to {filepath}: {e}")


def calibrate_landmarks(landmarks):
    """
    Normalize landmarks by the distance between wrist (index=0)
    and middle finger tip (index=12).
    """
    if len(landmarks) < 21:
        return landmarks  # Not a full hand

    wrist = np.array(landmarks[0])
    middle_tip = np.array(landmarks[12])

    ref_distance = np.linalg.norm(middle_tip - wrist)
    if ref_distance < 1e-6:
        ref_distance = 1.0

    calibrated = [
        (
            (lm[0] - wrist[0]) / ref_distance,
            (lm[1] - wrist[1]) / ref_distance,
            (lm[2] - wrist[2]) / ref_distance
        )
        for lm in landmarks
    ]
    return calibrated


def average_landmarks(samples):
    """Compute the average position of each landmark from multiple samples."""
    if not samples:
        return None
    arrays = [np.array(sample) for sample in samples]
    mean_array = np.mean(arrays, axis=0)
    return mean_array.tolist()


def compare_gesture(current_landmarks, reference_landmarks):
    """
    Compare current calibrated landmarks with reference calibrated landmarks.
    Returns:
      - overall_gesture_completion (float)
      - finger_completions (dict)
    """
    finger_indices = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20]
    }

    # Finger-level completion
    finger_completions = {}
    for finger, indices in finger_indices.items():
        distances_finger = []
        for idx in indices:
            dist = distance.euclidean(current_landmarks[idx], reference_landmarks[idx])
            distances_finger.append(dist)
        avg_dist_finger = np.mean(distances_finger)
        finger_completion = max(0, 100 - (avg_dist_finger * FINGER_SCALING_FACTOR))
        finger_completions[finger] = round(finger_completion, 2)

    # Overall gesture completion
    all_distances = []
    for c_lm, r_lm in zip(current_landmarks, reference_landmarks):
        dist = distance.euclidean(c_lm, r_lm)
        all_distances.append(dist)
    avg_dist = np.mean(all_distances)
    gesture_completion = max(0, 100 - (avg_dist * GESTURE_SCALING_FACTOR))
    return round(gesture_completion, 2), finger_completions


def draw_ghost(frame, ref_landmarks, color=(255, 0, 0)):
    """
    Draw a "ghost" outline of the reference (target) gesture on the frame.
    We'll assume the reference landmarks are normalized, so we place them
    relative to the top-left corner for simple visualization.
    """
    h, w, _ = frame.shape
    x_shift, y_shift = 50, 100
    scale = 250

    connections = mp_hands.HAND_CONNECTIONS

    points_2d = []
    for (x, y, z) in ref_landmarks:
        px = int(x_shift + x * scale)
        py = int(y_shift + y * scale)
        points_2d.append((px, py))

    for conn in connections:
        start_idx, end_idx = conn
        if start_idx < len(points_2d) and end_idx < len(points_2d):
            cv2.line(frame, points_2d[start_idx], points_2d[end_idx], color, 2)

    for (px, py) in points_2d:
        cv2.circle(frame, (px, py), 4, color, -1)

# -------------------- Streamlit State Initialization -------------------- #
def init_streamlit_state():
    """Initialize Streamlit session state variables if not present."""
    if "gesture_data" not in st.session_state:
        st.session_state["gesture_data"] = load_gestures()
    if "mode" not in st.session_state:
        st.session_state["mode"] = "Idle"
    if "current_gesture_name" not in st.session_state:
        st.session_state["current_gesture_name"] = ""
    if "recorded_samples" not in st.session_state:
        st.session_state["recorded_samples"] = []
    if "recorded_count" not in st.session_state:
        st.session_state["recorded_count"] = 0
    if "use_attempts" not in st.session_state:
        st.session_state["use_attempts"] = 0
    if "best_score" not in st.session_state:
        st.session_state["best_score"] = 0.0
    if "hands" not in st.session_state:
        st.session_state["hands"] = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_NUM_HANDS,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )
    if "cap" not in st.session_state:
        # Attempt to open the camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("[Error] Cannot open webcam.")
            st.stop()  # Stop the app if no camera found
        st.session_state["cap"] = cap

# -------------------- Gesture Recording Logic -------------------- #
def start_recording():
    """Switch to recording mode and reset counters."""
    st.session_state["mode"] = "record"
    st.session_state["recorded_samples"] = []
    st.session_state["recorded_count"] = 0


def record_sample():
    """Record one sample (if a hand is detected)."""
    if "frame_result" not in st.session_state or st.session_state["frame_result"] is None:
        st.warning("No hand detection result available yet.")
        return

    result = st.session_state["frame_result"]
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  # first hand only
        current_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        current_landmarks = calibrate_landmarks(current_landmarks)
        st.session_state["recorded_samples"].append(current_landmarks)
        st.session_state["recorded_count"] += 1
        st.success(f"Sample {st.session_state['recorded_count']} recorded.")
    else:
        st.warning("No hand detected. Please make sure your hand is in view.")


def finalize_gesture_recording():
    """Finalize recording by averaging samples and saving JSON."""
    if st.session_state["recorded_count"] < NUM_SAMPLES_PER_GESTURE:
        st.warning("You haven't recorded enough samples yet.")
        return

    gesture_name = st.session_state["current_gesture_name"].strip()
    if not gesture_name:
        st.error("Gesture name is empty.")
        return

    avg_lm = average_landmarks(st.session_state["recorded_samples"])
    st.session_state["gesture_data"][gesture_name] = {
        "samples": st.session_state["recorded_samples"],
        "average": avg_lm
    }
    save_gestures(st.session_state["gesture_data"])
    st.success(f"Gesture '{gesture_name}' recorded and saved successfully.")

    # Reset recording mode
    st.session_state["mode"] = "Idle"
    st.session_state["recorded_samples"] = []
    st.session_state["recorded_count"] = 0


def cancel_recording():
    """Cancel gesture recording."""
    st.session_state["mode"] = "Idle"
    st.session_state["recorded_samples"] = []
    st.session_state["recorded_count"] = 0
    st.info("Recording cancelled.")

# -------------------- Gesture Use Logic -------------------- #
def start_using_gesture():
    """Switch to use mode and reset usage counters."""
    st.session_state["mode"] = "use"
    st.session_state["use_attempts"] = 0
    st.session_state["best_score"] = 0.0


def stop_using_gesture():
    """Stop using mode."""
    st.session_state["mode"] = "Idle"
    st.session_state["use_attempts"] = 0
    st.session_state["best_score"] = 0.0
    st.info("Stopped using gesture.")


def color_for_score(score):
    """
    Return a BGR color tuple for the text overlay based on the score:
    - below 50%: red
    - 50%-80%: yellow
    - above 80%: green
    """
    if score < 50:
        return (0, 0, 255)   # red
    elif score < 80:
        return (0, 255, 255) # yellow
    else:
        return (0, 255, 0)   # green


def process_usage_frame(frame, result, gesture_name):
    """Display usage details on the frame and compute match score."""
    gesture_info = st.session_state["gesture_data"].get(gesture_name, None)
    if not gesture_info:
        st.warning(f"Gesture '{gesture_name}' not found.")
        return frame

    ref_landmarks = gesture_info["average"]

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        current_landmarks = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        current_landmarks = calibrate_landmarks(current_landmarks)

        # Compare the gesture
        overall_score, finger_scores = compare_gesture(current_landmarks, ref_landmarks)

        # Update attempts
        st.session_state["use_attempts"] += 1
        if overall_score > st.session_state["best_score"]:
            st.session_state["best_score"] = overall_score

        # Draw overall score
        overall_color = color_for_score(overall_score)
        cv2.putText(frame, f"Overall: {overall_score}%", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, overall_color, 2, cv2.LINE_AA)

        # Provide textual feedback in the Streamlit UI
        with st.container():
            if overall_score < 50:
                st.warning("Your overall gesture match is below 50%. Try adjusting your hand positioning further.")
            elif overall_score < 80:
                st.info("You're getting closer! Keep adjusting your fingers.")
            else:
                st.success("Great job! Your gesture match looks good.")

        # Draw finger scores
        y_offset = 70
        for finger, comp in finger_scores.items():
            c = color_for_score(comp)
            cv2.putText(frame, f"{finger.capitalize()}: {comp}%", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, c, 2, cv2.LINE_AA)
            y_offset += 30

        # Provide feedback about each finger in Streamlit
        feedback_messages = []
        for finger, comp in finger_scores.items():
            if comp < 50:
                feedback_messages.append(f"{finger.capitalize()}: needs significant adjustment.")
            elif comp < 80:
                feedback_messages.append(f"{finger.capitalize()}: moderate alignment. Keep fine-tuning.")
            else:
                feedback_messages.append(f"{finger.capitalize()}: good alignment!")
        # Display them in one block to avoid repeated messages flickering
        st.write("**Finger Feedback:**")
        for msg in feedback_messages:
            st.write("- ", msg)

        # Additional info
        cv2.putText(frame, f"Attempts: {st.session_state['use_attempts']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        y_offset += 30
        cv2.putText(frame, f"Best Score: {st.session_state['best_score']}%", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)

        # Optional ghost
        if SHOW_GHOST_OVERLAY:
            draw_ghost(frame, ref_landmarks, color=(255, 0, 0))

    return frame

# -------------------- Main Streamlit App -------------------- #
def main():
   
    # Initialize session state
    init_streamlit_state()
    gesture_data = st.session_state["gesture_data"]

    # UI for choosing mode
    col1, col2 = st.columns(2)

    with col1:
        new_gesture_name = st.text_input("Enter a new gesture name:")
        record_btn = st.button("Start Recording", on_click=start_recording)

    with col2:
        if gesture_data:
            gesture_list = list(gesture_data.keys())
            chosen_gesture = st.selectbox("Choose an existing gesture to use:", gesture_list)
            use_btn = st.button("Use Selected Gesture", on_click=start_using_gesture)
        else:
            st.info("No gestures recorded yet.")
            chosen_gesture = ""

    # Update current gesture name if user typed something
    if st.session_state["mode"] == "record" and new_gesture_name:
        st.session_state["current_gesture_name"] = new_gesture_name

    if st.session_state["mode"] == "use" and chosen_gesture:
        st.session_state["current_gesture_name"] = chosen_gesture

    # Show recording / usage controls based on the mode
    if st.session_state["mode"] == "record":
        st.subheader(f"Recording Gesture: {st.session_state['current_gesture_name']}")
        st.write(f"Samples Recorded: {st.session_state['recorded_count']} / {NUM_SAMPLES_PER_GESTURE}")
        if st.button("Record Sample"):
            record_sample()
        if st.button("Finalize Gesture"):
            finalize_gesture_recording()
        if st.button("Cancel Recording"):
            cancel_recording()

    elif st.session_state["mode"] == "use":
        st.subheader(f"Using Gesture: {st.session_state['current_gesture_name']}")
        st.write(f"Attempts so far: {st.session_state['use_attempts']}")
        st.write(f"Current Best Score: {st.session_state['best_score']}%")
        if st.button("Stop Using Gesture"):
            stop_using_gesture()

    # Display the webcam frames with detection/usage logic
    frame_placeholder = st.empty()

    # We’ll do a small loop to simulate a near-real-time feed.
    # This is a workaround in Streamlit’s environment.
    if st.session_state["mode"] in ("record", "use"):
        for _ in range(10000):
            cap = st.session_state["cap"]
            hands = st.session_state["hands"]
            ret, frame = cap.read()

            if not ret:
                st.error("Camera frame not received.")
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb_frame)
            st.session_state["frame_result"] = result  # store for record_sample()

            # If in usage mode, process usage overlay
            if st.session_state["mode"] == "use":
                frame = process_usage_frame(frame, result, st.session_state["current_gesture_name"])
            else:
                # If in record mode, just draw the landmarks
                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Convert BGR -> RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            time.sleep(0.03)  # small delay to avoid hogging CPU
            # If the user changed mode in the meantime, break
            if st.session_state["mode"] not in ("record", "use"):
                break

    else:
        st.info("Choose a mode: Record a new gesture or use an existing one.")


if __name__ == "__main__":
    main()

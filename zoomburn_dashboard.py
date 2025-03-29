import cv2
import mediapipe as mp
import math
from deepface import DeepFace
import numpy as np
import streamlit as st
import time
import pandas as pd
from datetime import datetime

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_eye_ratio(face_landmarks, h, w):
    def get_landmark(idx):
        pt = face_landmarks.landmark[idx]
        return int(pt.x * w), int(pt.y * h)

    top = get_landmark(159)
    bottom = get_landmark(145)
    left = get_landmark(33)
    right = get_landmark(133)

    eye_openness = euclidean(top, bottom)
    eye_width = euclidean(left, right)
    return eye_openness / eye_width

# --- Streamlit UI Setup ---
st.set_page_config(page_title="ZoomBurn AI", layout="wide")
st.title("🧠 ZoomBurn AI – Real-Time Burnout Detection")

if 'run' not in st.session_state:
    st.session_state.run = False

if 'data' not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Time", "Eye Ratio", "Emotion", "Burnout Level"])

start_stop = st.button("▶ Start Detection" if not st.session_state.run else "⏹ Stop Detection")
if start_stop:
    st.session_state.run = not st.session_state.run

frame_slot = st.empty()
status_slot = st.empty()

# --- Sidebar Dashboard ---
with st.sidebar:
    st.subheader("📈 Burnout Trend (last 10)")
    if not st.session_state.data.empty:
        chart_data = st.session_state.data.copy()
        chart_data["Time"] = pd.to_datetime(chart_data["Time"])
        chart_data.set_index("Time", inplace=True)

        st.line_chart(chart_data[["Eye Ratio"]].tail(20))
        st.dataframe(chart_data.tail(10), use_container_width=True)

        if st.button("💾 Export CSV"):
            chart_data.to_csv("burnout_log.csv")
            st.success("Saved to burnout_log.csv")

# --- Detection Loop ---
if st.session_state.run:
    cap = cv2.VideoCapture(0)
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh()

    while st.session_state.run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Webcam failed.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        burnout_score = 0
        eye_ratio = 0.0
        emotion = "unknown"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                eye_ratio = get_eye_ratio(face_landmarks, h, w)

                if eye_ratio < 0.20:
                    burnout_score += 1

                try:
                    analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    if emotion in ['sad', 'angry', 'tired', 'disgust', 'fear']:
                        burnout_score += 1
                except:
                    emotion = "unknown"

        # Burnout classification
        burnout_level = "Low"
        if burnout_score == 1:
            burnout_level = "Medium"
        elif burnout_score >= 2:
            burnout_level = "High 🚨"

        # Update UI
        status_slot.markdown(f"""
        - **Eye Ratio**: `{eye_ratio:.2f}`
        - **Emotion**: `{emotion}`
        - **Burnout Level**: `{burnout_level}`
        """)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_slot.image(frame_bgr, channels="BGR")

        # Track data
        current_time = datetime.now()
        st.session_state.data = pd.concat([
            st.session_state.data,
            pd.DataFrame([{
                "Time": current_time,
                "Eye Ratio": round(eye_ratio, 2),
                "Emotion": emotion,
                "Burnout Level": burnout_level
            }])
        ], ignore_index=True)

        time.sleep(1)

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

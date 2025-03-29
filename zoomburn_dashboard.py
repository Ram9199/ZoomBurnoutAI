import cv2
import mediapipe as mp
import math
from deepface import DeepFace
import numpy as np
import streamlit as st
import time
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# --- Utility Functions ---
def euclidean(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_eye_ratio(face_landmarks, h, w):
    def get_landmark(idx):
        pt = face_landmarks.landmark[idx]
        return int(pt.x * w), int(pt.y * h)

    # Left eye landmarks
    left_top = get_landmark(159)
    left_bottom = get_landmark(145)
    left_left = get_landmark(33)
    left_right = get_landmark(133)

    # Right eye landmarks
    right_top = get_landmark(386)
    right_bottom = get_landmark(374)
    right_left = get_landmark(362)
    right_right = get_landmark(263)

    # Calculate ratios for both eyes
    left_eye_openness = euclidean(left_top, left_bottom)
    left_eye_width = euclidean(left_left, left_right)
    left_ratio = left_eye_openness / left_eye_width

    right_eye_openness = euclidean(right_top, right_bottom)
    right_eye_width = euclidean(right_left, right_right)
    right_ratio = right_eye_openness / right_eye_width

    # Return average of both eyes
    return (left_ratio + right_ratio) / 2

def get_mouth_ratio(face_landmarks, h, w):
    def get_landmark(idx):
        pt = face_landmarks.landmark[idx]
        return int(pt.x * w), int(pt.y * h)

    # Mouth landmarks
    top = get_landmark(13)
    bottom = get_landmark(14)
    left = get_landmark(61)
    right = get_landmark(291)

    mouth_height = euclidean(top, bottom)
    mouth_width = euclidean(left, right)
    return mouth_height / mouth_width

# --- Streamlit UI Setup ---
st.set_page_config(page_title="ZoomBurn AI", layout="wide")
st.title("🧠 ZoomBurn AI – Real-Time Burnout Detection")

# Initialize session state variables
if 'run' not in st.session_state:
    st.session_state.run = False
    st.session_state.data = pd.DataFrame(columns=["Time", "Eye Ratio", "Mouth Ratio", "Emotion", "Burnout Level", "Score"])
    st.session_state.start_time = None
    st.session_state.chart_data = []

# Create two columns for layout
col1, col2 = st.columns([2, 1])

with col1:
    start_stop = st.button("▶ Start Detection" if not st.session_state.run else "⏹ Stop Detection", key="start_stop_button")
    if start_stop:
        st.session_state.run = not st.session_state.run
        if st.session_state.run:
            st.session_state.start_time = datetime.now()
            # Clear previous data when starting new session
            st.session_state.data = pd.DataFrame(columns=["Time", "Eye Ratio", "Mouth Ratio", "Emotion", "Burnout Level", "Score"])
            st.session_state.chart_data = []

    frame_slot = st.empty()
    status_slot = st.empty()

with col2:
    st.markdown("### 📈 Burnout Trend Analysis")
    
    # Create containers for dynamic updates
    chart_container = st.container()
    with chart_container:
        trend_chart = st.empty()
        st.markdown("### Recent Measurements")
        metrics_table = st.empty()
        if st.button("💾 Export Data", key="export_button"):
            if not st.session_state.data.empty:
                st.session_state.data.to_csv("burnout_log.csv", index=False)
                st.success("Saved to burnout_log.csv")
            else:
                st.warning("No data to export yet!")

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
        mouth_ratio = 0.0
        emotion = "unknown"

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                eye_ratio = get_eye_ratio(face_landmarks, h, w)
                mouth_ratio = get_mouth_ratio(face_landmarks, h, w)

                # Eye fatigue detection
                if eye_ratio < 0.25:  # Adjusted threshold
                    burnout_score += 1.5

                # Yawning detection
                if mouth_ratio > 0.5:
                    burnout_score += 1

                try:
                    analysis = DeepFace.analyze(frame_rgb, actions=['emotion'], enforce_detection=False)
                    emotion = analysis[0]['dominant_emotion']
                    
                    # Enhanced emotion scoring
                    emotion_scores = {
                        'sad': 1.5,
                        'angry': 1.2,
                        'tired': 1.3,
                        'disgust': 1.4,
                        'fear': 1.1,
                        'neutral': 0.5,
                        'happy': 0.2
                    }
                    burnout_score += emotion_scores.get(emotion, 0)
                except:
                    emotion = "unknown"

        # Normalize score to 0-100 range
        normalized_score = min(100, max(0, burnout_score * 20))

        # Enhanced burnout classification
        burnout_level = "Low"
        if normalized_score > 40:
            burnout_level = "Medium"
        elif normalized_score > 70:
            burnout_level = "High 🚨"

        # Update UI with more detailed information
        status_slot.markdown(f"""
        ### Current Status
        - **Eye Ratio**: `{eye_ratio:.2f}`
        - **Mouth Ratio**: `{mouth_ratio:.2f}`
        - **Emotion**: `{emotion}`
        - **Burnout Score**: `{normalized_score:.1f}/100`
        - **Burnout Level**: `{burnout_level}`
        """)

        # Draw face mesh on frame
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame_slot.image(frame_bgr, channels="BGR")

        # Track data with more metrics
        current_time = datetime.now()
        elapsed_time = current_time - st.session_state.start_time
        time_str = current_time.strftime("%H:%M:%S")
        
        # Store data in session state
        new_row = {
            "Time": time_str,
            "Elapsed": str(elapsed_time).split('.')[0],
            "Eye Ratio": round(eye_ratio, 2),
            "Mouth Ratio": round(mouth_ratio, 2),
            "Emotion": emotion,
            "Burnout Level": burnout_level,
            "Score": round(normalized_score, 1)
        }
        
        st.session_state.data = pd.concat([
            st.session_state.data,
            pd.DataFrame([new_row])
        ], ignore_index=True)
        
        st.session_state.chart_data.append(normalized_score)

        # Update trend visualization
        if len(st.session_state.chart_data) > 0:
            fig = go.Figure()
            
            # Add main burnout score line
            fig.add_trace(go.Scatter(
                y=st.session_state.chart_data,
                mode='lines+markers',
                name='Burnout Score',
                line=dict(color='#FF6B6B', width=2),
                marker=dict(size=6)
            ))
            
            # Add threshold lines
            fig.add_hline(y=40, line_dash="dash", line_color="yellow", annotation_text="Medium")
            fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="High")

            fig.update_layout(
                title="Real-time Burnout Score",
                xaxis_title="Time (seconds)",
                yaxis_title="Score",
                height=300,
                margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False,
                plot_bgcolor='rgba(255, 255, 255, 0.9)',
                yaxis=dict(
                    range=[0, 100],
                    gridcolor='rgba(0, 0, 0, 0.1)',
                    zerolinecolor='rgba(0, 0, 0, 0.2)'
                ),
                xaxis=dict(
                    gridcolor='rgba(0, 0, 0, 0.1)',
                    zerolinecolor='rgba(0, 0, 0, 0.2)'
                )
            )
            
            trend_chart.plotly_chart(fig, use_container_width=True)

            # Update metrics table
            metrics_table.dataframe(
                st.session_state.data.tail(5)[["Time", "Elapsed", "Emotion", "Score", "Burnout Level"]],
                hide_index=True,
                use_container_width=True
            )

        time.sleep(0.5)  # Reduced delay for smoother updates

    cap.release()
    face_mesh.close()
    cv2.destroyAllWindows()

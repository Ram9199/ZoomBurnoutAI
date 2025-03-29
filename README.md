# ZoomBurn AI - Real-Time Burnout Detection

A sophisticated real-time AI-powered application that monitors and detects signs of burnout during video calls and computer work using advanced facial analysis, emotion recognition, and physiological indicators.

## 🌟 Key Features

- **Real-time Face Analysis**
  - Advanced face mesh detection using MediaPipe
  - Dual eye fatigue monitoring system
  - Yawning detection for tiredness assessment
  - Emotion analysis using DeepFace

- **Burnout Metrics**
  - Eye openness ratio tracking
  - Mouth ratio analysis for yawning detection
  - Multi-factor emotion scoring
  - Normalized burnout score (0-100)
  - Three-level burnout classification (Low, Medium, High)

- **Interactive Dashboard**
  - Real-time video feed with face mesh overlay
  - Live burnout score trending
  - Dynamic threshold indicators
  - Recent measurements table
  - Data export functionality

## 🛠️ Technical Stack

- **Core Technologies**
  - Python 3.7+
  - OpenCV for video processing
  - MediaPipe for face mesh detection
  - DeepFace for emotion analysis
  - Streamlit for interactive UI
  - Plotly for real-time visualization

- **Key Dependencies**
  ```
  opencv-python>=4.8.0
  mediapipe>=0.10.0
  deepface>=0.0.79
  streamlit>=1.28.0
  pandas>=2.0.0
  numpy>=1.24.0
  plotly>=5.18.0
  ```

## 📦 Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/Ram9199/ZoomBurnoutAI.git
   cd ZoomBurnoutAI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**
   ```bash
   streamlit run zoomburn_dashboard.py
   ```

## 🎯 How It Works

1. **Face Detection & Analysis**
   - Captures video feed from webcam
   - Applies MediaPipe face mesh for precise facial landmark detection
   - Tracks 468 facial landmarks in real-time

2. **Burnout Detection Metrics**
   - **Eye Fatigue**: Monitors eye openness ratio using facial landmarks
   - **Yawning Detection**: Analyzes mouth aspect ratio
   - **Emotion Analysis**: Uses DeepFace to detect 7 emotional states
   - **Scoring System**: Combines multiple factors with weighted scoring

3. **Real-time Monitoring**
   - Continuous score calculation and normalization
   - Dynamic threshold monitoring
   - Trend analysis and visualization
   - Historical data tracking

## 📊 Data Analysis

The application tracks multiple metrics:
- Timestamp and session duration
- Eye openness ratios
- Mouth aspect ratios
- Detected emotions
- Burnout scores and levels

Data can be exported to CSV format for further analysis.

## ⚙️ Configuration

Default thresholds:
- Eye Ratio < 0.25: Indicates fatigue
- Mouth Ratio > 0.5: Indicates yawning
- Burnout Score:
  - 0-40: Low
  - 41-70: Medium
  - 71-100: High

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

## 📄 License

This project is licensed under the MIT License.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a sole method for diagnosing burnout or other mental health conditions. Always consult healthcare professionals for proper diagnosis and treatment.

## 🔗 Contact

- GitHub: [Ram9199](https://github.com/Ram9199)
- Project Link: [ZoomBurnoutAI](https://github.com/Ram9199/ZoomBurnoutAI) 
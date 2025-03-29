# ZoomBurn AI - Real-Time Burnout Detection

A real-time webcam-based application that monitors and detects signs of burnout using facial analysis and emotion recognition.

## 🎯 Features

- Real-time webcam feed with face detection
- Eye openness ratio monitoring
- Emotion analysis using DeepFace
- Burnout level classification (Low, Medium, High)
- Interactive Streamlit dashboard
- Data tracking and visualization
- CSV export functionality

## 🛠️ Requirements

- Python 3.7+
- OpenCV
- MediaPipe
- DeepFace
- Streamlit
- Pandas
- NumPy

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Ram9199/ZoomBurnoutAI.git
cd ZoomBurnoutAI
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## 🚀 Usage

1. Run the Streamlit application:
```bash
streamlit run zoomburn_dashboard.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Click the "Start Detection" button to begin monitoring

4. The dashboard will show:
   - Live webcam feed
   - Real-time eye ratio measurements
   - Current emotion detection
   - Burnout level classification
   - Historical data visualization

5. Use the sidebar to view trends and export data

## 📊 How It Works

The application uses multiple indicators to detect burnout:

1. **Eye Openness Ratio**: Monitors the ratio between eye height and width to detect fatigue
2. **Emotion Analysis**: Analyzes facial expressions for negative emotions
3. **Burnout Classification**: Combines multiple factors to determine burnout level

## 📝 Data Export

The application automatically tracks:
- Timestamp
- Eye ratio measurements
- Detected emotions
- Burnout level classification

Data can be exported to CSV format for further analysis.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as the sole method for diagnosing burnout or other mental health conditions. Always consult with healthcare professionals for proper diagnosis and treatment. 
























\# WeCare 💚

\### Privacy-Preserving Elderly Health \& Safety Monitoring Using Wi-Fi Sensing
## 🌐 Live Prototype

The deployed WeCare prototype is available here:

https://wecare-mvp-d8ezdak9ta3tawewu8hfdq.streamlit.app/

> The current MVP demonstrates Wi-Fi CSI-based activity recognition using a trained machine-learning model and prototype safety-monitoring interfaces.



WeCare is an intelligent ambient safety monitoring prototype that uses Wi-Fi Channel State Information (CSI) and machine learning to recognize human activities without requiring cameras or wearable devices.



The system is designed with elderly safety, privacy and non-intrusive monitoring in mind.



\---



\## 🎯 Problem



Traditional elderly monitoring systems often depend on:



\- Cameras, which may compromise privacy

\- Wearable devices, which users may forget to wear

\- Manual supervision

\- Expensive dedicated sensing infrastructure



WeCare explores a privacy-preserving alternative using changes in Wi-Fi signals caused by human movement.



\---



\## 💡 Our Solution



Wi-Fi signals interact with the human body and surrounding environment.



When a person moves, the characteristics of these signals change.



WeCare processes Wi-Fi CSI data and uses a trained machine-learning model to classify the person's activity.



\### Current activity classes



\- Walking

\- Standing

\- Sitting Down

\- Static



The detected activity is then presented through an interactive safety and health monitoring dashboard.



\---



\## ✨ Key Features



\### 📡 Live CSI Sensing

Processes CSI samples and displays the current detected activity with AI confidence.



\### 🧠 AI Activity Recognition

Machine-learning-based recognition of walking, standing, sitting down and static activity.



\### 👤 Digital Human Twin

Interactive body visualization with:



\- Front, back, left and right views

\- Sensor visualization

\- Risk layer

\- Skeleton layer

\- Reported pain/discomfort layer

\- Body-region selection



\### ⚠️ Mobility \& Fall-Risk Interface

Displays movement information, inactivity indicators and safety-risk information.



\### ❤️ Sleep, Vitals \& Biometrics

Prototype interface for future health-sensing capabilities.



> These health/vital values are simulated UI demonstrations in the current MVP and are not clinical measurements produced by the activity-recognition model.



\### 🚨 Safety Log

Provides a centralized view of safety events with acknowledgement and Digital Human Twin navigation.



\### 👥 Occupant Attribution

Prototype multi-occupant interface for selecting and monitoring residents.



\### 📊 Reports

Displays session activity history and supports CSV report export.



\### 📝 Simple Summary

Explains the important monitoring information in simple language for caregivers and non-technical users.



\---



\## 🏗️ System Architecture



```text

Wi-Fi CSI Data

&#x20;     ↓

Amplitude + Phase Measurements

&#x20;     ↓

Feature Extraction

(mean, std, min, max)

&#x20;     ↓

240 Model Features

&#x20;     ↓

Machine Learning Model

&#x20;     ↓

Activity Classification

&#x20;     ↓

Confidence + Safety Interpretation

&#x20;     ↓

WeCare Streamlit Dashboard

```



\---



\## 🤖 Machine Learning



The current prototype uses a trained activity-recognition model operating on CSI amplitude and phase information.



For each CSI subcarrier, statistical features are generated:



\- Mean

\- Standard deviation

\- Minimum

\- Maximum



The resulting feature vector contains \*\*240 model features\*\*.



The dashboard then displays the predicted activity and confidence score.



\---



\## 📂 Project Structure



```text

wecare-mvp/

│

├── app.py

├── requirements.txt

├── .gitignore

├── README.md

│

├── data/

│   └── test.csv

│

└── model/

&#x20;   ├── wecare\_activity\_model.pkl

&#x20;   └── wecare\_feature\_columns.pkl

```



\---



\## 🛠️ Technology Stack



\- Python

\- Streamlit

\- Pandas

\- NumPy

\- Scikit-learn

\- Joblib

\- Plotly

\- Wi-Fi CSI

\- Machine Learning



\---



\## 🚀 Running WeCare Locally



\### 1. Clone the repository



```bash

git clone <YOUR-REPOSITORY-URL>

cd wecare-mvp

```



\### 2. Create a virtual environment



```bash

python -m venv .venv

```



\### 3. Install dependencies



```bash

pip install -r requirements.txt

```



\### 4. Run the application



```bash

streamlit run app.py

```



Streamlit will provide a local URL, usually:



```text

http://localhost:8501

```



\---



\## 🔒 Privacy-First Approach



WeCare is designed around ambient sensing rather than continuous video surveillance.



The long-term concept uses Wi-Fi CSI to understand movement patterns while avoiding cameras in private living environments.



\---



\## 🔮 Future Scope



Future development can include:



\- ESP32-based real-time CSI acquisition

\- Real-time Wi-Fi sensing hardware

\- Fall-event recognition

\- Improved multi-person attribution

\- Sleep pattern analysis

\- Vital-sign sensing research

\- Caregiver notifications

\- Emergency escalation

\- Cloud/edge deployment

\- Historical health analytics



\---



\## ⚠️ Prototype Disclaimer



WeCare is currently a prototype/MVP.



The activity-recognition functionality uses the trained CSI machine-learning pipeline included in this project.



Health metrics, biometric values, some safety indicators and multi-person features shown in the interface are prototype demonstrations unless explicitly connected to sensing/model functionality.



WeCare is not a medical device and should not be used for clinical diagnosis or emergency medical decision-making.



\---



\## 👩‍💻 Team



\*\*Techno Tiaras\*\*



Project: \*\*WeCare\*\*



Built for \*\*She Solves 3.0\*\*


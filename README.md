# AI LearnTrack - Student Performance Prediction System

![AI LearnTrack Banner]

## 📝 Overview
AI LearnTrack is an intelligent student performance prediction system that leverages machine learning to analyze student data and provide actionable insights. The system predicts final exam scores, identifies at-risk students, and offers personalized learning recommendations.

## ✨ Features

- **Performance Prediction**: Predicts final exam scores based on various academic and behavioral factors
- **Risk Assessment**: Identifies students at risk of underperforming
- **Learner Profiling**: Categorizes students into different learner types for tailored recommendations
- **Interactive Dashboard**: Visual analytics for tracking student performance trends
- **PDF Reports**: Generates detailed performance reports
- **Responsive Design**: Works on desktop and mobile devices

##  Project Structure

```
Capstone/
│
├── app.py                  # Main Flask application
├── train_models.py         # Script for training ML models
├── requirements.txt        # Python dependencies
│
├── data/                   # Data storage
│   └── predictions_log.csv # Log of all predictions
│
├── models/                 # Trained ML models
│   ├── model_rf.joblib
│   ├── model_clf.joblib
│   ├── model_cluster.joblib
│   └── cluster_scaler.pkl
│
├── static/
│   ├── css/
│   │   └── styles.css     # Custom styles
│   └── js/
│       ├── script.js      # Main JavaScript
│       └── insights.js    # Dashboard interactivity
│
├── templates/              # HTML templates
│   ├── index.html         # Main dashboard
│   ├── insights.html      # Analytics dashboard
│   └── report.html        # Detailed report view
│
└── reports/               # Generated PDF reports
    └── student_report_*.pdf
```

## 🛠️ Installation


1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python app.py
   ```

4. **Access the application**
   Open your browser and visit: `http://localhost:5000`

## 🧠 Machine Learning Models

The system uses three main models:

1. **Regression Model**
   - Predicts final exam scores
   - Uses Random Forest Regressor

2. **Classification Model**
   - Predicts risk levels (Low, Medium, High, Critical)
   - Uses Random Forest Classifier

3. **Clustering Model**
   - Categorizes students into learner types
   - Uses K-Means Clustering with StandardScaler

## 📊 Data Flow

1. User inputs data through the web interface
2. Data is preprocessed and fed to the ML models
3. Models generate predictions and insights
4. Results are displayed to the user and logged
5. Users can view historical data and analytics

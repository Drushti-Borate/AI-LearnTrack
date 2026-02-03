# app.py

from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import joblib
import os
from datetime import datetime
import csv  
from pathlib import Path

# For PDF generation
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors

app = Flask(__name__)

# Load models
MODELS_DIR = "models"

regressor = joblib.load(os.path.join(MODELS_DIR, "performance_regressor.pkl"))
classifier = joblib.load(os.path.join(MODELS_DIR, "risk_classifier.pkl"))
cluster_model = joblib.load(os.path.join(MODELS_DIR, "learner_cluster.pkl"))
cluster_scaler = joblib.load(os.path.join(MODELS_DIR, "cluster_scaler.pkl"))
cluster_label_map = joblib.load(os.path.join(MODELS_DIR, "learner_cluster_labels.pkl"))

# feature order must match train_models.py
FEATURE_COLS = [
    "subject1_marks",
    "subject1_outof",
    "subject2_marks",
    "subject2_outof",
    "subject3_marks",
    "subject3_outof",
    "subject4_marks",
    "subject4_outof",
    "subject5_marks",
    "subject5_outof",
    "average_marks_percent",
    "previous_year_percentage",
    "attendance",
    "study_hours",
    "sleep_hours",
    "focus_time",
]


def calculate_average_percent(subjects):
    """
    subjects: list of dicts with keys: marks, outof
    """
    valid = [s for s in subjects if s["outof"] > 0]
    if not valid:
        return 0.0
    percents = [(s["marks"] / s["outof"]) * 100 for s in valid]
    return sum(percents) / len(percents)


def generate_suggestions(pred_score, risk_level, avg_percent, prev_percent,
                         attendance, study_hours, sleep_hours, focus_time):
    """
    Suggestion system based on:
    - Performance
    - Attendance
    - Study hours
    - Sleep hours
    - Focus time
    - Comparison with previous year
    """
    suggestions = []

    # ---------------- PERFORMANCE ----------------
    if pred_score < 40:
        suggestions.append(
            "Your predicted performance is very low. Focus on strengthening basic concepts and follow a strict daily study plan."
        )
    elif pred_score < 60:
        suggestions.append(
            "Your predicted performance is below average. Increase your study consistency and revise weak topics regularly."
        )
    elif pred_score < 75:
        suggestions.append(
            "Your predicted performance is moderate. With more revision and practice, you can improve your score."
        )
    elif pred_score < 90:
        suggestions.append(
            "Your performance is good. Maintain your current routine and try to solve more practice questions to reach an excellent level."
        )
    else:
        suggestions.append(
            "Excellent predicted performance! Keep up your habits and continue refining your understanding with higher-level problems."
        )

    # ---------------- ATTENDANCE ----------------
    if attendance < 50:
        suggestions.append(
            "Your attendance is critically low (< 50%). Try to attend classes regularly to avoid missing important concepts."
        )
    elif 50 <= attendance < 65:
        suggestions.append(
            "Your attendance needs improvement (between 50% and 65%). Aim for at least 75% to stay in touch with the syllabus."
        )
    elif 65 <= attendance <= 90:
        suggestions.append(
            "Your attendance is moderate (65%–90%). Slight improvement can further boost your understanding and performance."
        )
    else:  # > 90
        suggestions.append(
            "Great attendance! (> 90%). Staying consistent in classes will continue to help you a lot."
        )

    # ---------------- STUDY HOURS ----------------
    if study_hours < 1:
        suggestions.append(
            "Your study hours are very low (< 1 hour/day). Try to study at least 1.5–2 hours per day to see noticeable improvement."
        )
    elif 1 <= study_hours < 1.5:
        suggestions.append(
            "Your study time is on the lower side. Increasing it slightly will help you cover topics more comfortably."
        )
    elif 1.5 <= study_hours < 4:
        suggestions.append(
            "Your study hours are moderate (1.5–4 hours/day). Maintain consistency and avoid distractions during this time."
        )
    elif 4 <= study_hours <= 7:
        suggestions.append(
            "You have good study duration (4–7 hours/day). Keep this habit and ensure you include revision and practice."
        )
    else:  # > 7
        suggestions.append(
            "You are studying more than 7 hours a day. Make sure to take breaks and avoid burnout by balancing rest and study."
        )

    # ---------------- SLEEP HOURS ----------------
    if sleep_hours < 4.5:
        suggestions.append(
            "You are sleeping too little (< 4.5 hours). Aim for at least 6–8 hours of sleep to maintain focus and memory."
        )
    elif 4.5 <= sleep_hours < 6:
        suggestions.append(
            "Your sleep duration is slightly low. Increasing sleep to 6–8 hours can significantly improve your concentration."
        )
    elif 6 <= sleep_hours <= 8:
        suggestions.append(
            "Your sleep schedule (6–8 hours) is optimal. Good sleep supports better learning and retention."
        )
    elif sleep_hours > 9:
        suggestions.append(
            "You might be oversleeping (> 9 hours). Try to maintain a balanced sleep schedule to stay active and focused."
        )

    # ---------------- FOCUS TIME ----------------
    if focus_time < 1:
        suggestions.append(
            "Your focused study time is very low (< 1 hour). Try techniques like Pomodoro to build better concentration."
        )
    elif 1 <= focus_time < 2:
        suggestions.append(
            "Your focus level needs improvement. Reduce distractions like phone and social media during study time."
        )
    elif 2 <= focus_time <= 3:
        suggestions.append(
            "You have good focus (2–3 hours of deep work). Maintain this habit to keep learning efficiently."
        )
    else:  # > 3
        suggestions.append(
            "Excellent focus! (> 3 hours of deep work). You are managing your concentration very well—keep it up."
        )

    # ---------------- COMPARE WITH PREVIOUS YEAR ----------------
    if pred_score > prev_percent + 10:
        suggestions.append(
            "You are predicted to perform much better than last year. Keep following your current strategies."
        )
    elif pred_score < prev_percent - 10:
        suggestions.append(
            "Your predicted score is significantly lower than last year. Try to identify where you are struggling and focus more on those areas."
        )
    else:
        suggestions.append(
            "Your predicted performance is close to last year's level. With some extra effort, you can push it to the next level."
        )

    return suggestions


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/insights")
def insights():
    return render_template("insights.html")


@app.route("/report")
def report():
    """
    Show latest prediction summary and a button to download PDF.
    Data source: last row of data/predictions_log.csv
    """
    log_path = Path("data/predictions_log.csv")
    latest_data = None

    if log_path.exists() and log_path.stat().st_size > 0:
        try:
            df = pd.read_csv(log_path)
            if not df.empty:
                latest = df.tail(1).iloc[0]
                latest_data = {
                    "timestamp": latest.get("timestamp", ""),
                    "predicted_score": latest.get("predicted_score", ""),
                    "risk_level": latest.get("risk_level", ""),
                    "learner_type": latest.get("learner_type", ""),
                    "average_marks": latest.get("average_marks", ""),
                    "study_hours": latest.get("study_hours", ""),
                    "sleep_hours": latest.get("sleep_hours", ""),
                    "focus_time": latest.get("focus_time", ""),
                    "attendance": latest.get("attendance", ""),
                    "previous_year_percentage": latest.get("previous_year_percentage", "")
                }
        except Exception as e:
            print("Error reading latest prediction for report:", e)

    return render_template("report.html", latest=latest_data)


@app.route("/download-report")
def download_report():
    """
    Generate a PDF for the latest prediction and send as download.
    """
    log_path = Path("data/predictions_log.csv")
    if not log_path.exists() or log_path.stat().st_size == 0:
        return "No predictions found to generate report.", 400

    try:
        df = pd.read_csv(log_path)
        if df.empty:
            return "No predictions found to generate report.", 400

        latest = df.tail(1).iloc[0]

        # Prepare PDF path
        reports_dir = Path("reports")
        reports_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = reports_dir / f"student_report_{timestamp_str}.pdf"

        # Get suggestions
        suggestions = generate_suggestions(
            pred_score=float(latest.get("predicted_score", 0)),
            risk_level=str(latest.get("risk_level", "")),
            avg_percent=float(latest.get("average_marks", 0)),
            prev_percent=float(latest.get("previous_year_percentage", 0)),
            attendance=float(latest.get("attendance", 0)),
            study_hours=float(latest.get("study_hours", 0)),
            sleep_hours=float(latest.get("sleep_hours", 0)),
            focus_time=float(latest.get("focus_time", 0))
        )

        # Create PDF
        c = canvas.Canvas(str(pdf_path), pagesize=A4)
        width, height = A4

        # Blue header
        c.setFillColorRGB(0.09, 0.27, 0.52)
        c.rect(0, height - 80, width, 80, stroke=0, fill=1)

        # Header text
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 18)
        c.drawString(40, height - 50, "AI Student Performance Report")

        # Body text
        c.setFillColor(colors.black)
        c.setFont("Helvetica", 11)

        y = height - 110
        line_gap = 16

        def put_line(text):
            nonlocal y
            c.drawString(40, y, text)
            y -= line_gap

        put_line(f"Report Generated On : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        put_line(f"Prediction Time     : {latest.get('timestamp', '')}")
        put_line("")

        put_line("=== Performance Summary ===")
        put_line(f"Predicted Final Score      : {latest.get('predicted_score', '')}")
        put_line(f"Average Current Marks (%)  : {latest.get('average_marks', '')}")
        put_line(f"Previous Year Percentage   : {latest.get('previous_year_percentage', '')}")
        put_line(f"Risk Level                 : {latest.get('risk_level', '')}")
        put_line(f"Learner Type               : {latest.get('learner_type', '')}")
        put_line("")

        put_line("=== Behaviour Summary ===")
        put_line(f"Study Hours / Day          : {latest.get('study_hours', '')}")
        put_line(f"Sleep Hours / Day          : {latest.get('sleep_hours', '')}")
        put_line(f"Focused Study Time (hrs)   : {latest.get('focus_time', '')}")
        put_line(f"Attendance (%)             : {latest.get('attendance', '')}")
        put_line("")

        # Add suggestions section
        put_line("=== Personalized Suggestions ===")
        put_line("")

        c.setFont("Helvetica", 10)

        for s in suggestions:
            # Wrap long lines into 80-character chunks
            max_len = 80
            parts = [s[i:i + max_len] for i in range(0, len(s), max_len)]
            for part in parts:
                put_line(f"• {part}")

            # Extra spacing after each suggestion
            y -= 5

            if y < 80:  # If we're near the bottom of the page
                c.showPage()
                y = height - 80
                c.setFont("Helvetica", 10)

        c.showPage()
        c.save()

        return send_file(str(pdf_path), as_attachment=True)

    except Exception as e:
        print("Error generating PDF:", e)
        return "Error generating PDF report.", 500


@app.route("/insights-data")
def insights_data():
    log_path = Path("data/predictions_log.csv")

    if not log_path.exists() or log_path.stat().st_size == 0:
        return {
            "average_score": 0,
            "average_study_hours": 0,
            "average_marks": 0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "learner_distribution": {},
        }

    try:
        clean_csv(log_path)
        df = pd.read_csv(log_path)

        required_columns = ["predicted_score", "risk_level", "learner_type",
                            "average_marks", "study_hours"]

        if not all(col in df.columns for col in required_columns):
            raise ValueError("CSV file is missing required columns")

        avg_score = df["predicted_score"].mean()
        avg_study = df["study_hours"].mean()
        avg_marks = df["average_marks"].mean()

        risk_dist = df["risk_level"].value_counts().to_dict()
        learner_dist = df["learner_type"].value_counts().to_dict()

        return {
            "average_score": round(avg_score, 2),
            "average_study_hours": round(avg_study, 2),
            "average_marks": round(avg_marks, 2),
            "risk_distribution": risk_dist,
            "learner_distribution": learner_dist,
            "total_students": len(df)
        }

    except Exception as e:
        print(f"Error processing insights data: {e}")
        return {
            "average_score": 0,
            "average_study_hours": 0,
            "average_marks": 0,
            "risk_distribution": {"Low": 0, "Medium": 0, "High": 0, "Critical": 0},
            "learner_distribution": {},
        }


def clean_csv(file_path):
    """
    Clean the predictions log CSV file by:
    - Removing empty lines
    - Ensuring proper line endings
    - Validating the number of columns
    - Creating a backup if needed
    
    Args:
        file_path (Path): Path to the CSV file to clean
    """
    try:
        # Read all lines from the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        cleaned_lines = []
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue

            # Keep header as is
            if i == 0:
                cleaned_lines.append(line.strip() + '\n')
                continue

            # Validate data rows have the correct number of columns
            parts = line.strip().split(',')
            if len(parts) == 10:  # Ensure we have exactly 10 columns
                cleaned_lines.append(line.strip() + '\n')

        # Write cleaned content back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

    except Exception as e:
        print(f"Error cleaning CSV file: {e}")
        
        # Create a backup of the corrupted file
        backup_path = file_path.with_suffix('.bak' + file_path.suffix)
        try:
            import shutil
            shutil.copy2(file_path, backup_path)
            print(f"Created backup at {backup_path}")

            # Recreate the file with just the header
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("timestamp,predicted_score,risk_level,learner_type,average_marks,study_hours,sleep_hours,focus_time,attendance,previous_year_percentage\n")

        except Exception as backup_error:
            print(f"Error creating backup: {backup_error}")


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction requests from the frontend.
    
    Processes student data, makes predictions using the trained models,
    and returns the results including score, risk level, learner type, and suggestions.
    Also logs the prediction to a CSV file for analytics.
    
    Expected JSON payload:
    {
        "subjects": [{"marks": number, "outof": number}, ...],  # Up to 5 subjects
        "previous_year_percentage": number,
        "attendance": number,
        "study_hours": number,
        "sleep_hours": number,
        "focus_time": number
    }
    
    Returns:
        JSON response with prediction results and suggestions
    """
    try:
        # Get and validate input data
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Get user's subjects (up to 5) and ensure we have exactly 5 entries for the model
        subjects = data.get("subjects", [])[:5]  # Get up to 5 subjects
        # Pad with empty subjects if needed to ensure exactly 5 entries
        # Using outof=0 for padding so they're ignored in average calculation
        while len(subjects) < 5:
            subjects.append({"marks": 0, "outof": 0})

        # Extract and validate numerical inputs with defaults
        prev_percent = float(data.get("previous_year_percentage", 0) or 0)
        attendance = float(data.get("attendance", 0) or 0)
        study_hours = float(data.get("study_hours", 0) or 0)
        sleep_hours = float(data.get("sleep_hours", 0) or 0)
        focus_time = float(data.get("focus_time", 0) or 0)

        # Calculate average percentage across all subjects
        avg_percent = calculate_average_percent(subjects)

        # Prepare feature vector for model prediction
        feature_values = [
            # Subject 1-5 marks and max marks (10 features)
            subjects[0]["marks"], subjects[0]["outof"],
            subjects[1]["marks"], subjects[1]["outof"],
            subjects[2]["marks"], subjects[2]["outof"],
            subjects[3]["marks"], subjects[3]["outof"],
            subjects[4]["marks"], subjects[4]["outof"],
            # Additional features
            avg_percent,      # Average marks percentage
            prev_percent,     # Previous year percentage
            attendance,       # Attendance percentage
            study_hours,      # Daily study hours
            sleep_hours,      # Daily sleep hours
            focus_time,       # Focused study time
        ]

        # Create DataFrame with proper feature names for the model
        X_input = pd.DataFrame([feature_values], columns=FEATURE_COLS)

        # Make predictions using the trained models
        predicted_score = float(regressor.predict(X_input)[0])
        predicted_score_rounded = round(predicted_score, 2)
        risk_level = str(classifier.predict(X_input)[0])

        # Determine learner type using clustering with scaled features
        cluster_df = pd.DataFrame([{
            "average_marks_percent": avg_percent,
            "previous_year_percentage": prev_percent,
            "attendance": attendance,
            "study_hours": study_hours,
            "sleep_hours": sleep_hours,
            "focus_time": focus_time
        }])

        cluster_scaled = cluster_scaler.transform(cluster_df)
        cluster_idx = int(cluster_model.predict(cluster_scaled)[0])
        learner_type = cluster_label_map.get(cluster_idx, "Unknown learner type")

        # Generate personalized suggestions
        suggestions = generate_suggestions(
            predicted_score_rounded,  # Predicted final score
            risk_level,              # Risk level (Low/Medium/High)
            avg_percent,             # Current average marks
            prev_percent,            # Previous year percentage
            attendance,              # Attendance percentage
            study_hours,             # Daily study hours
            sleep_hours,             # Daily sleep hours
            focus_time               # Focused study time
        )

        # Prepare response
        response = {
            "predicted_score": predicted_score_rounded,
            "risk_level": risk_level,
            "learner_type": learner_type,
            "average_marks_percent": round(avg_percent, 2),
            "suggestions": suggestions,  # List of personalized suggestions
        }

        # Log the prediction to CSV for analytics
        log_dir = Path("data")
        log_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
        log_path = log_dir / "predictions_log.csv"
        file_exists = log_path.exists()

        try:
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                # Write header if file is new
                if not file_exists:
                    writer.writerow([
                        "timestamp", "predicted_score", "risk_level", "learner_type",
                        "average_marks", "study_hours", "sleep_hours", "focus_time",
                        "attendance", "previous_year_percentage"
                    ])

                # Write prediction data
                writer.writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # Timestamp
                    predicted_score_rounded,                        # Predicted score
                    risk_level,                                     # Risk level
                    learner_type,                                   # Learner type
                    round(avg_percent, 2),                          # Average marks
                    study_hours,                                    # Study hours
                    sleep_hours,                                    # Sleep hours
                    focus_time,                                     # Focus time
                    attendance,                                     # Attendance
                    prev_percent,                                   # Previous year %
                ])

        except Exception as e:
            print(f"Error writing to log file: {e}")
            # Continue even if logging fails - don't affect user experience

        return jsonify(response)

    except Exception as e:
        # Log the full error for debugging
        print(f"Error in predict endpoint: {e}")
        # Return a user-friendly error message
        return jsonify({"error": "An error occurred while processing your request"}), 500


if __name__ == "__main__":
    # Run the Flask development server
    # Set debug=True for development (auto-reloads on code changes)
    # In production, use a production WSGI server like Gunicorn or uWSGI
    app.run(debug=True)

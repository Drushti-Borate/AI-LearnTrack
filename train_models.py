# train_models.py 

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, classification_report

# Paths
DATA_PATH = os.path.join("data", "student_dataset_realistic.csv")
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

print("Loading dataset from:", DATA_PATH)
df = pd.read_csv(DATA_PATH)


feature_cols = [
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

X = df[feature_cols]
y_reg = df["final_score"]
y_clf = df["risk_level"]

# ---- Train-test split ----
X_train, X_test, y_reg_train, y_reg_test = train_test_split(
    X, y_reg, test_size=0.2, random_state=42
)

X_train_clf, X_test_clf, y_clf_train, y_clf_test = train_test_split(
    X, y_clf, test_size=0.2, random_state=42
)

# ---- Regression model: Final Score ----
reg_model = RandomForestRegressor(
    n_estimators=300,       # stronger learning
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

reg_model.fit(X_train, y_reg_train)
y_reg_pred = reg_model.predict(X_test)

mse = mean_squared_error(y_reg_test, y_reg_pred)
r2 = r2_score(y_reg_test, y_reg_pred)

print("=== Regression Model (Final Score) ===")
print("MSE:", round(mse, 3))
print("R2:", round(r2, 3))
print()

joblib.dump(reg_model, os.path.join(MODELS_DIR, "performance_regressor.pkl"))
print("Saved: performance_regressor.pkl")

# ---- Classification model: Risk Level ----
clf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=14,
    random_state=42,
    n_jobs=-1
)

clf_model.fit(X_train_clf, y_clf_train)
y_clf_pred = clf_model.predict(X_test_clf)

print("=== Classification Model (Risk Level) ===")
print(classification_report(y_clf_test, y_clf_pred))
print()

joblib.dump(clf_model, os.path.join(MODELS_DIR, "risk_classifier.pkl"))
print("Saved: risk_classifier.pkl")

# ---- Clustering model: Learner Type (KMeans) ----
from sklearn.preprocessing import StandardScaler

cluster_features_cols = [
    "average_marks_percent",
    "previous_year_percentage",
    "attendance",
    "study_hours",
    "sleep_hours",
    "focus_time",
]

X_cluster = df[cluster_features_cols]

# Scale data (VERY IMPORTANT)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Save scaler
joblib.dump(scaler, os.path.join(MODELS_DIR, "cluster_scaler.pkl"))

# Train KMeans
k = 4
cluster_model = KMeans(
    n_clusters=k,
    random_state=42,
    n_init=10
)
cluster_model.fit(X_scaled)

joblib.dump(cluster_model, os.path.join(MODELS_DIR, "learner_cluster.pkl"))
print("Saved: learner_cluster.pkl")

# Generate human-readable cluster labels
centers = cluster_model.cluster_centers_

# Higher center means stronger learning behaviour
learning_strength = centers[:, 3] + centers[:, 5] + centers[:, 0]

sorted_idx = np.argsort(learning_strength)

cluster_label_map = {}

cluster_label_map[int(sorted_idx[0])] = "Slow-paced learner"
cluster_label_map[int(sorted_idx[1])] = "Needs improvement learner"
cluster_label_map[int(sorted_idx[2])] = "Consistent learner"
cluster_label_map[int(sorted_idx[3])] = "Fast learner"

joblib.dump(cluster_label_map, os.path.join(MODELS_DIR, "learner_cluster_labels.pkl"))
print("Saved: learner_cluster_labels.pkl")

print("\nTraining complete with fixed dataset! 🚀")

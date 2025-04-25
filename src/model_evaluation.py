import os
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# Load features
test_data = pd.read_csv("data/features/test.csv")

# Split features and labels
x_test = test_data.drop(columns=["label"]).values
y_test = test_data["label"].values

# Load trained model
model = joblib.load("data/models/model.pkl")

# Predict on test data
y_pred = model.predict(x_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Store metrics
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'roc_auc': roc_auc
}

# Ensure 'reports/' directory exists
os.makedirs("data/reports", exist_ok=True)

# Save metrics to JSON
with open("data/reports/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
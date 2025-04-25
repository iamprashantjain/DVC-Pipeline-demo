import pickle
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Load features
train_data = pd.read_csv("data/features/train.csv")
test_data = pd.read_csv("data/features/test.csv")

# Split into features and labels
x_train = train_data.drop(columns=["label"]).values
y_train = train_data["label"].values

x_test = test_data.drop(columns=["label"]).values
y_test = test_data["label"].values

# Train XGBoost model
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgb_model.fit(x_train, y_train)

# Create model directory if it doesn't exist
model_path = 'data/models'
os.makedirs(model_path, exist_ok=True)

# Save model
with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
    pickle.dump(xgb_model, f)

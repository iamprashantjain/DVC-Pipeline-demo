# import os
# import json
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# # Load features
# test_data = pd.read_csv("data/features/test.csv")

# # Split features and labels
# x_test = test_data.drop(columns=["label"]).values
# y_test = test_data["label"].values

# # Load trained model
# model = joblib.load("data/models/model.pkl")

# # Predict on test data
# y_pred = model.predict(x_test)

# # Evaluate performance
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)

# # Store metrics
# metrics = {
#     'accuracy': accuracy,
#     'precision': precision,
#     'recall': recall,
#     'roc_auc': roc_auc
# }

# # Ensure 'reports/' directory exists
# os.makedirs("data/reports", exist_ok=True)

# # Save metrics to JSON
# with open("data/reports/metrics.json", "w") as f:
#     json.dump(metrics, f, indent=4)



import os
import json
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report


def load_test_data(file_path="data/features/test.csv"):
    """Load the test dataset from a CSV file."""
    test_data = pd.read_csv(file_path)
    x_test = test_data.drop(columns=["label"]).values
    y_test = test_data["label"].values
    return x_test, y_test


def load_model(model_path="data/models/model.pkl"):
    """Load the trained model from a file."""
    return joblib.load(model_path)


def predict(model, x_test):
    """Use the trained model to make predictions on the test data."""
    return model.predict(x_test)


def evaluate_model(y_test, y_pred):
    """Evaluate the model performance using various metrics."""
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }
    
    return metrics


def save_metrics(metrics, output_dir="data/reports", output_file="metrics.json"):
    """Save the evaluation metrics to a JSON file."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w") as f:
        json.dump(metrics, f, indent=4)


def main():
    # Load test data and model
    x_test, y_test = load_test_data()
    model = load_model()

    # Predict and evaluate
    y_pred = predict(model, x_test)
    metrics = evaluate_model(y_test, y_pred)

    # Save the evaluation metrics to a JSON file
    save_metrics(metrics)


if __name__ == "__main__":
    main()

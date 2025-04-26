# import pickle
# import os
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, classification_report
# import yaml

# n_estimators = yaml.safe_load(open('params.yaml','r'))['model_trainer']['n_estimators']
# learning_rate = yaml.safe_load(open('params.yaml','r'))['model_trainer']['learning_rate']

# # Load features
# train_data = pd.read_csv("data/features/train.csv")
# test_data = pd.read_csv("data/features/test.csv")

# # Split into features and labels
# x_train = train_data.drop(columns=["label"]).values
# y_train = train_data["label"].values

# x_test = test_data.drop(columns=["label"]).values
# y_test = test_data["label"].values

# # Train XGBoost model
# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=n_estimators, learning_rate=learning_rate)

# xgb_model.fit(x_train, y_train)

# # Create model directory if it doesn't exist
# model_path = 'data/models'
# os.makedirs(model_path, exist_ok=True)

# # Save model
# with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
#     pickle.dump(xgb_model, f)



import os
import pickle
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb


def load_params(config_file='params.yaml'):
    """Load parameters from the configuration YAML file."""
    with open(config_file, 'r') as file:
        params = yaml.safe_load(file)
    return params


def load_data(train_file="data/features/train.csv", test_file="data/features/test.csv"):
    """Load train and test datasets."""
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    return train_data, test_data


def preprocess_data(train_data, test_data):
    """Preprocess the datasets by separating features and labels."""
    x_train = train_data.drop(columns=["label"]).values
    y_train = train_data["label"].values

    x_test = test_data.drop(columns=["label"]).values
    y_test = test_data["label"].values

    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train, n_estimators, learning_rate):
    """Train the XGBoost model."""
    xgb_model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    xgb_model.fit(x_train, y_train)
    return xgb_model


def save_model(model, model_path="data/models", model_name="model.pkl"):
    """Save the trained model to a file."""
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, model_name), 'wb') as f:
        pickle.dump(model, f)


def main():
    # Load configuration parameters
    params = load_params()
    n_estimators = params['model_trainer']['n_estimators']
    learning_rate = params['model_trainer']['learning_rate']

    # Load data
    train_data, test_data = load_data()

    # Preprocess data
    x_train, y_train, x_test, y_test = preprocess_data(train_data, test_data)

    # Train model
    model = train_model(x_train, y_train, n_estimators, learning_rate)

    # Save the trained model
    save_model(model)


if __name__ == "__main__":
    main()
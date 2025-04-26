# import numpy as np 
# import pandas as pd 
# from sklearn.feature_extraction.text import CountVectorizer
# import os
# import yaml

# max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

# train_data = pd.read_csv("data/processed/train.csv")
# test_data = pd.read_csv("data/processed/test.csv")

# # Fill NaNs and convert to list of strings
# X_train = train_data['content'].fillna("").tolist()
# y_train = train_data['sentiment'].values

# X_test = test_data['content'].fillna("").tolist()
# y_test = test_data['sentiment'].values

# #bow vectorizer
# vectorizer = CountVectorizer(max_features=max_features)

# x_train_bow = vectorizer.fit_transform(X_train)
# x_test_bow = vectorizer.transform(X_test)

# train_df = pd.DataFrame(x_train_bow.toarray())
# train_df['label'] = y_train

# test_df = pd.DataFrame(x_test_bow.toarray())
# test_df['label'] = y_test

# data_path = os.path.join('data', 'features')
# os.makedirs(data_path, exist_ok=True)

# train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import os
import yaml


def load_params(file_path="params.yaml"):
    """Load parameters from a YAML file."""
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def load_data(train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    """Load train and test datasets."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data


def preprocess_data(train_data, test_data):
    """Preprocess the train and test data, fill NaNs, and convert to list of strings."""
    X_train = train_data['content'].fillna("").tolist()
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].fillna("").tolist()
    y_test = test_data['sentiment'].values

    return X_train, y_train, X_test, y_test


def vectorize_data(X_train, X_test, max_features):
    """Apply CountVectorizer to the data and transform it."""
    vectorizer = CountVectorizer(max_features=max_features)
    
    x_train_bow = vectorizer.fit_transform(X_train)
    x_test_bow = vectorizer.transform(X_test)
    
    return x_train_bow, x_test_bow


def create_dataframe(x_train_bow, x_test_bow, y_train, y_test):
    """Convert the BOW matrices to DataFrames and add labels."""
    train_df = pd.DataFrame(x_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(x_test_bow.toarray())
    test_df['label'] = y_test

    return train_df, test_df


def save_data(train_df, test_df, data_path="data/features"):
    """Save the processed data to CSV files."""
    os.makedirs(data_path, exist_ok=True)
    
    train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


def main():
    # Load parameters
    params = load_params()
    max_features = params['feature_engineering']['max_features']

    # Load train and test data
    train_data, test_data = load_data()

    # Preprocess data
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)

    # Vectorize data
    x_train_bow, x_test_bow = vectorize_data(X_train, X_test, max_features)

    # Create dataframes
    train_df, test_df = create_dataframe(x_train_bow, x_test_bow, y_train, y_test)

    # Save the results
    save_data(train_df, test_df)


if __name__ == "__main__":
    main()
import numpy as np 
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import os

train_data = pd.read_csv("data/processed/train.csv")
test_data = pd.read_csv("data/processed/test.csv")

# Fill NaNs and convert to list of strings
X_train = train_data['content'].fillna("").tolist()
y_train = train_data['sentiment'].values

X_test = test_data['content'].fillna("").tolist()
y_test = test_data['sentiment'].values

#bow vectorizer
vectorizer = CountVectorizer(max_features=500)

x_train_bow = vectorizer.fit_transform(X_train)
x_test_bow = vectorizer.transform(X_test)

train_df = pd.DataFrame(x_train_bow.toarray())
train_df['label'] = y_train

test_df = pd.DataFrame(x_test_bow.toarray())
test_df['label'] = y_test

data_path = os.path.join('data', 'features')
os.makedirs(data_path, exist_ok=True)

train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)
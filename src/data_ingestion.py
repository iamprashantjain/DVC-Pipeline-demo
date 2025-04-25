import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
df = pd.read_csv(url)

# Drop unnecessary column
df.drop(columns=['tweet_id'], inplace=True)

# Filter for binary classification and encode labels
final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

# Split the data
train_data, test_data = train_test_split(final_df, test_size=0.2, random_state=42)

# Create directory if it doesn't exist
data_path = os.path.join('data', 'raw')
os.makedirs(data_path, exist_ok=True)

# Save to CSV
train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

print(f"Train and test data saved in: {data_path}")
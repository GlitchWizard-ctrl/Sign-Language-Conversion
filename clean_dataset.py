# Step 1: Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Step 2: Load dataset (replace with your actual file path)
df = pd.read_csv(r"C:\Users\Pavitra K\OneDrive\Documents\Project\major Project\archive\data.csv")

# Step 3: Inspect data
print("Initial dataset shape:", df.shape)
print(df.head())
print(df.info())

# Step 4: Handle missing values
# Fill numeric columns with median
numeric_cols = [col for col in df.columns if df[col].dtype in ['int64', 'float64']]
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Drop rows where gesture label (A–Z) is missing
df = df.dropna(subset=['gesture_label'])

# Step 5: Remove duplicates
df = df.drop_duplicates()

# Step 6: Encode categorical variables (gesture labels)
df = pd.get_dummies(df, columns=['gesture_label'], drop_first=True)

# Step 7: Normalize numeric features
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Step 8: Split dataset into train/test with reproducibility
train, test = train_test_split(df, test_size=0.2, random_state=42)

print("Cleaned dataset shape:", df.shape)
print("Training set shape:", train.shape)
print("Test set shape:", test.shape)

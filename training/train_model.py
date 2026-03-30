"""
Diabetes Prediction Model Training Script
Uses Logistic Regression to predict diabetes with Pima Indian Dataset
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import pickle
import os

# Create output directory if it doesn't exist
os.makedirs('model', exist_ok=True)

print("=" * 60)
print("DIABETES PREDICTION MODEL TRAINING")
print("Pima Indian Diabetes Dataset")
print("=" * 60)

# Step 1: Load the Pima Indian diabetes dataset
print("\n1. Loading dataset...")
df = pd.read_csv('data/diabetes.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names: {list(df.columns)}")
print(f"\nDataset info:")
print(df.info())

# Step 2: Prepare features (X) and target (y)
print("\n2. Preparing features and target...")
# Assuming last column is target, all others are features
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Step 3: Split the data into training and testing sets
print("\n3. Splitting data (80-20 split)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 4: Standardize the features (IMPORTANT for Logistic Regression)
print("\n4. Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Features standardized successfully")

# Step 5: Train logistic regression model
print("\n5. Training Logistic Regression model...")
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

print("Model training completed")

# Step 6: Make predictions and evaluate
print("\n6. Evaluating model...")
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

# Calculate metrics
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test)
recall = recall_score(y_test, y_pred_test)
f1 = f1_score(y_test, y_pred_test)

print(f"\nTRAINING METRICS:")
print(f"  Training Accuracy: {train_accuracy:.4f}")
print(f"  Testing Accuracy:  {test_accuracy:.4f}")
print(f"  Precision:         {precision:.4f}")
print(f"  Recall:            {recall:.4f}")
print(f"  F1-Score:          {f1:.4f}")

print(f"\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred_test))

print(f"\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred_test))

# Step 7: Save the model, scaler, and feature names
print("\n7. Saving model artifacts...")
model_path = 'model/diabetes_model.pkl'
scaler_path = 'model/scaler.pkl'
features_path = 'model/feature_names.pkl'

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

# Save feature names for later use
feature_names = list(X.columns)
with open(features_path, 'wb') as f:
    pickle.dump(feature_names, f)

print(f"\n✓ Model saved to: {model_path}")
print(f"✓ Scaler saved to: {scaler_path}")
print(f"✓ Feature names saved to: {features_path}")

print("\n" + "=" * 60)
print("TRAINING COMPLETE!")
print("=" * 60)
print(f"\nNext steps:")
print(f"1. Create Gradio app (huggingface-space/app.py)")
print(f"2. Deploy to Hugging Face Space")
print(f"3. Build the frontend website")
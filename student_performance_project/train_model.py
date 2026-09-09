"""
train_model.py
--------------
Purpose:
This script trains a Random Forest Classifier to predict student performance
('Good', 'Average', 'Low') based on study hours, attendance, previous marks,
assignment score, sleep hours, and extracurricular activities.

Steps Included:
1. Load dataset from CSV file.
2. Check for missing values and inspect data types.
3. Preprocess features (e.g., encode categorical variables).
4. Split dataset into Training (80%) and Testing (20%) sets.
5. Train the Random Forest Classifier model.
6. Evaluate model accuracy, classification report, and confusion matrix.
7. Save the trained model to a file ('model.pkl') for deployment in the Streamlit app.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_student_model():
    print("=" * 60)
    print("STEP 1: LOADING DATASET")
    print("=" * 60)
    
    csv_file = "student_data.csv"
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Dataset '{csv_file}' not found. Please run generate_dataset.py first.")
    
    df = pd.read_csv(csv_file)
    print(f"Dataset successfully loaded. Total rows: {df.shape[0]}, Columns: {df.shape[1]}")
    print("\nDataset Preview:")
    print(df.head())

    print("\n" + "=" * 60)
    print("STEP 2: DATA INSPECTION & PREPROCESSING")
    print("=" * 60)
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    print("Missing values per column:")
    print(missing_counts)
    
    # Preprocess categorical column: Extracurricular_Activities ('Yes' -> 1, 'No' -> 0)
    # This converts text into numbers so the machine learning model can understand it.
    df['Extracurricular_Activities'] = df['Extracurricular_Activities'].map({'Yes': 1, 'No': 0})
    
    print("\nProcessed 'Extracurricular_Activities' to numeric (1 = Yes, 0 = No):")
    print(df[['Extracurricular_Activities']].head())

    print("\n" + "=" * 60)
    print("STEP 3: SEPARATING FEATURES (X) AND TARGET (y)")
    print("=" * 60)
    
    # Feature columns used for prediction
    feature_columns = [
        'Study_Hours',
        'Attendance',
        'Previous_Marks',
        'Assignment_Score',
        'Sleep_Hours',
        'Extracurricular_Activities'
    ]
    
    X = df[feature_columns]
    y = df['Performance']
    
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print("\nClass distribution in target:")
    print(y.value_counts())

    print("\n" + "=" * 60)
    print("STEP 4: TRAIN-TEST SPLIT")
    print("=" * 60)
    
    # 80% data for training the model, 20% data for testing model performance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples:  {X_test.shape[0]}")

    print("\n" + "=" * 60)
    print("STEP 5: TRAINING RANDOM FOREST CLASSIFIER")
    print("=" * 60)
    
    # Random Forest combines multiple decision trees (ensemble learning)
    # n_estimators=100 means 100 decision trees will vote on the final prediction.
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        max_depth=6
    )
    
    # Train the model on training data
    model.fit(X_train, y_train)
    print("Random Forest Classifier model training completed!")

    print("\n" + "=" * 60)
    print("STEP 6: MODEL EVALUATION ON TEST DATA")
    print("=" * 60)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy * 100:.2f}%\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\n" + "=" * 60)
    print("STEP 7: FEATURE IMPORTANCE ANALYSIS")
    print("=" * 60)
    
    # Inspect which features influenced the predictions most
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print(importance_df)

    # Plot and save feature importance chart
    try:
        plt.figure(figsize=(8, 4.5))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color='#3B82F6')
        plt.xlabel('Importance Score')
        plt.title('Random Forest Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        chart_filename = "feature_importance.png"
        plt.savefig(chart_filename, dpi=150)
        plt.close()
        print(f"\nFeature importance chart saved as '{chart_filename}'.")
    except Exception as e:
        print(f"Note: Could not save chart: {e}")

    print("\n" + "=" * 60)
    print("STEP 8: SAVING TRAINED MODEL")
    print("=" * 60)
    
    model_filename = "model.pkl"
    joblib.dump(model, model_filename)
    print(f"Trained model saved successfully as '{model_filename}'!")
    print("You can now run 'streamlit run app.py' to launch the web application.")

if __name__ == "__main__":
    train_student_model()

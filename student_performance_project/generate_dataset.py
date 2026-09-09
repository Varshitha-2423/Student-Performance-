"""
generate_dataset.py
-------------------
Purpose:
This script creates a synthetic dataset representing student academic records.
It generates realistic data for 6 features and calculates a target label
('Good', 'Average', 'Low') based on a logical combination of these features with some noise.

Features:
1. Study_Hours: Daily study hours (1.0 to 10.0)
2. Attendance: Percentage of classes attended (45.0% to 100.0%)
3. Previous_Marks: Marks scored in previous exams (30.0 to 100.0)
4. Assignment_Score: Marks scored in internal assignments (30.0 to 100.0)
5. Sleep_Hours: Daily average sleep hours (4.0 to 9.5)
6. Extracurricular_Activities: Whether student participates in activities (Yes/No)

Target:
- Performance: 'Good', 'Average', or 'Low'
"""

import numpy as np
import pandas as pd

def generate_student_dataset(n_samples=600, random_seed=42):
    # Set seed for reproducibility so the same dataset is generated every time
    np.random.seed(random_seed)

    # 1. Study Hours (e.g., between 1 and 10 hours per day)
    study_hours = np.round(np.random.uniform(1.0, 10.0, n_samples), 1)

    # 2. Attendance percentage (e.g., between 45% and 100%)
    attendance = np.round(np.random.uniform(45.0, 100.0, n_samples), 1)

    # 3. Previous Marks out of 100 (e.g., between 30 and 100)
    previous_marks = np.round(np.random.uniform(30.0, 100.0, n_samples), 1)

    # 4. Assignment Score out of 100 (correlated somewhat with previous marks)
    assignment_noise = np.random.normal(0, 5, n_samples)
    assignment_score = np.clip(np.round(0.7 * previous_marks + 0.3 * attendance + assignment_noise, 1), 30.0, 100.0)

    # 5. Sleep Hours (e.g., between 4 and 9.5 hours per night)
    sleep_hours = np.round(np.random.uniform(4.0, 9.5, n_samples), 1)

    # 6. Extracurricular Activities (Yes or No with 50% probability)
    extracurricular = np.random.choice(['Yes', 'No'], size=n_samples, p=[0.5, 0.5])
    extracurricular_numeric = np.where(extracurricular == 'Yes', 1.0, 0.0)

    # Composite Score calculation with realistic weights:
    # Academics (Previous marks & assignments) have highest weight, followed by attendance & study hours
    academic_score = (
        0.30 * previous_marks +
        0.25 * assignment_score +
        0.25 * attendance +
        1.80 * study_hours +
        0.80 * sleep_hours +
        3.00 * extracurricular_numeric
    )

    # Add realistic random variation (noise) representing real-life factors
    random_noise = np.random.normal(0, 3.5, n_samples)
    final_score = academic_score + random_noise

    # Assign Performance categories based on realistic score thresholds:
    # Below 62 -> 'Low'
    # 62 to 80 -> 'Average'
    # Above 80 -> 'Good'
    performance = []
    for score in final_score:
        if score >= 80.0:
            performance.append('Good')
        elif score >= 62.0:
            performance.append('Average')
        else:
            performance.append('Low')

    # Construct the pandas DataFrame
    df = pd.DataFrame({
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Marks': previous_marks,
        'Assignment_Score': assignment_score,
        'Sleep_Hours': sleep_hours,
        'Extracurricular_Activities': extracurricular,
        'Performance': performance
    })

    return df

if __name__ == "__main__":
    print("Generating synthetic student dataset...")
    df = generate_student_dataset(n_samples=600, random_seed=42)

    # Save to CSV
    output_filename = "student_data.csv"
    df.to_csv(output_filename, index=False)

    print(f"Dataset successfully saved as '{output_filename}'!")
    print(f"Total Records: {len(df)}")
    print("\nClass Distribution:")
    print(df['Performance'].value_counts())
    print("\nFirst 5 Sample Records:")
    print(df.head())

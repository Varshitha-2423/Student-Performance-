# 🎓 Student Performance Prediction Using Machine Learning

An end-to-end, beginner-friendly Machine Learning project that predicts whether a student's academic performance will be **Good**, **Average**, or **Low** based on their study habits, attendance, and lifestyle factors.

---

## 📌 Project Overview

- **Problem Statement:** Educational institutions often struggle to identify struggling or at-risk students before final examinations occur. Early prediction enables timely academic interventions and personalized mentoring.
- **Solution:** A Machine Learning classification system using the **Random Forest Classifier** algorithm, integrated with an intuitive **Streamlit** web user interface.
- **Target Audience:** College freshers, machine learning beginners, teachers, and academic counselors.

---

## 🗂️ Project Structure & File Explanations

```text
student_performance_project/
│
├── student_data.csv          # The dataset containing 600 student records
├── generate_dataset.py       # Script to generate realistic synthetic student data
├── train_model.py            # Script to preprocess data, train Random Forest, and save model
├── model.pkl                 # Serialized trained Machine Learning model
├── feature_importance.png    # Chart showing which features influence predictions most
├── app.py                    # Interactive Streamlit web application
├── requirements.txt          # Python dependencies required to run the project
└── README.md                 # Project documentation & interview preparation guide
```

### Detailed Purpose of Each File:
1. **`student_data.csv`**: Contains student records with 6 input features (`Study_Hours`, `Attendance`, `Previous_Marks`, `Assignment_Score`, `Sleep_Hours`, `Extracurricular_Activities`) and 1 target label (`Performance`: Good, Average, Low).
2. **`generate_dataset.py`**: A Python utility that creates a realistic dataset using statistical distributions and real-world academic correlations.
3. **`train_model.py`**: The machine learning pipeline. It reads the CSV, cleans missing values, encodes categories (Yes/No to 1/0), splits data into train and test sets, trains a `RandomForestClassifier`, evaluates test accuracy, and exports `model.pkl`.
4. **`model.pkl`**: The saved binary file of the trained Random Forest model created using `joblib`. It allows the web app to make instant predictions without retraining every time.
5. **`feature_importance.png`**: An automatically generated bar chart illustrating the contribution of each feature to the model's decision-making.
6. **`app.py`**: A frontend web dashboard built with Streamlit allowing users to enter student details using sliders and dropdowns, then view the predicted performance, confidence levels, and actionable advice.
7. **`requirements.txt`**: A clean list of required Python packages (`pandas`, `numpy`, `scikit-learn`, `matplotlib`, `streamlit`, `joblib`).

---

## 💻 Step-by-Step Instructions to Run on Windows

Follow these exact steps to run the project on any Windows 10 or 11 laptop:

### Step 1: Open Terminal (PowerShell or Command Prompt)
Press `Win + R`, type `cmd` or `powershell`, and press **Enter**.

### Step 2: Navigate to Project Directory
```powershell
cd c:\Users\varsh\.antigravity-ide\student_performance_project
```
*(Or navigate to wherever you have placed the `student_performance_project` folder)*

### Step 3: (Optional but Recommended) Create & Activate a Virtual Environment
```powershell
# Create a virtual environment named 'venv'
python -m venv venv

# Activate on Windows Command Prompt:
venv\Scripts\activate.bat

# Or activate on Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### Step 4: Install Required Packages
```powershell
pip install -r requirements.txt
```

### Step 5: (Optional) Re-generate Dataset or Train Model
The dataset and trained model are already included. If you want to re-run:
```powershell
# To generate a fresh dataset:
python generate_dataset.py

# To train the Random Forest model:
python train_model.py
```

### Step 6: Launch the Streamlit Web Application
```powershell
streamlit run app.py
```
After running this command, your default web browser will automatically open at:
```text
http://localhost:8501
```

---

## 🧠 The Machine Learning Algorithm in Simple English

### What is a Decision Tree?
Imagine making a decision:
- *Did the student study more than 5 hours?*
  - **Yes** -> *Are previous marks > 70?* -> If Yes, predict **Good**.
  - **No** -> *Is attendance < 60%?* -> If Yes, predict **Low**.

A single tree makes choices by asking a series of yes/no questions. However, a single decision tree can be easily biased or make mistakes on unseen data (called **overfitting**).

### What is a Random Forest?
A **Random Forest** is an ensemble of many decision trees (in our project, 100 trees):
1. **Forest of Trees:** Instead of asking just one teacher, you ask a panel of 100 teachers.
2. **Random Subsets:** Each decision tree is trained on a slightly different random subset of students and features.
3. **Majority Voting:** When a new student's data is entered, all 100 trees cast their individual votes. If 85 trees vote "Good", 10 vote "Average", and 5 vote "Low", the final output is **Good** with 85% confidence!

**Why Random Forest is great for beginners:**
- High accuracy without complex mathematical tuning.
- Resilient to noise and outliers.
- Shows feature importance clearly.

---

## 🔄 Project Workflow for an Interview

When asked *"Walk me through your project workflow"*, describe this 5-stage pipeline:

```text
[ Raw Student Data ]
        ↓
[ 1. Data Preprocessing & Cleaning ]
   - Handled missing values
   - Converted categorical strings (Yes/No) to numeric (1/0)
        ↓
[ 2. Train-Test Split ]
   - 80% data for training the model
   - 20% reserved unseen data for testing
        ↓
[ 3. Model Training & Evaluation ]
   - Trained RandomForestClassifier with 100 estimators
   - Evaluated accuracy, precision, recall, and confusion matrix
        ↓
[ 4. Model Serialization ]
   - Serialized trained model to model.pkl using joblib
        ↓
[ 5. Web Deployment with Streamlit ]
   - Created intuitive UI with sliders & inputs
   - Real-time inference with probability breakdown & recommendations
```

---

## 📊 Sample Inputs and Outputs

### Sample 1: High Performing Student
- **Inputs:**
  - Study Hours: `8.5 hrs`
  - Attendance: `94%`
  - Previous Marks: `88`
  - Assignment Score: `90`
  - Sleep Hours: `7.5 hrs`
  - Extracurricular Activities: `Yes`
- **Output:**
  - **Predicted Performance:** 🟢 **Good**
  - **Confidence:** ~98% Good

---

### Sample 2: Average Performing Student
- **Inputs:**
  - Study Hours: `4.5 hrs`
  - Attendance: `72%`
  - Previous Marks: `65`
  - Assignment Score: `68`
  - Sleep Hours: `7.0 hrs`
  - Extracurricular Activities: `Yes`
- **Output:**
  - **Predicted Performance:** 🟡 **Average**
  - **Confidence:** ~78% Average

---

### Sample 3: At-Risk / Low Performing Student
- **Inputs:**
  - Study Hours: `1.5 hrs`
  - Attendance: `48%`
  - Previous Marks: `35`
  - Assignment Score: `38`
  - Sleep Hours: `5.0 hrs`
  - Extracurricular Activities: `No`
- **Output:**
  - **Predicted Performance:** 🔴 **Low**
  - **Confidence:** ~98% Low

---

## 🎯 10 Common Fresher Interview Questions & Answers

### Q1: What is the main objective of this project?
> **Answer:** The objective is to proactively predict student academic performance into three tiers: Good, Average, or Low. This helps educational institutions identify students who need mentoring before final exams, improving overall academic outcomes.

### Q2: Why did you choose Random Forest instead of a single Decision Tree?
> **Answer:** A single decision tree tends to overfit the training data and can be overly sensitive to noise. Random Forest creates an ensemble of 100 trees with bagging (bootstrap aggregating) and feature randomness, leading to higher generalization and stability through majority voting.

### Q3: What is the difference between Classification and Regression? Which one is this?
> **Answer:** Regression predicts continuous numerical values (such as predicting exact marks e.g., 78.5), whereas Classification categorizes inputs into distinct classes. This project is a **multi-class classification** problem because it categorizes performance into three discrete labels: Good, Average, and Low.

### Q4: Why do we split the dataset into Training and Testing sets?
> **Answer:** We split data (80% training, 20% testing) to evaluate how well the model performs on unseen data. If we test on the same data we trained on, the model could simply memorize the inputs (overfitting) rather than learning true underlying patterns.

### Q5: How did you handle categorical variables in the data?
> **Answer:** The `Extracurricular_Activities` feature had text values ('Yes' and 'No'). Machine learning models require numerical inputs, so I applied binary label encoding: mapping 'Yes' to `1` and 'No' to `0`.

### Q6: What is `random_state=42` used for in scikit-learn?
> **Answer:** `random_state` is a seed value for pseudo-random number generators. Setting a fixed value (like 42) ensures that data splits and tree generation are reproducible—meaning anyone running the code will get the exact same results.

### Q7: What metric did you use to evaluate the model and why?
> **Answer:** I evaluated the model using **Accuracy**, along with **Precision**, **Recall**, **F1-score**, and a **Confusion Matrix**. Because student performance categories can have real consequences (missing an at-risk student is costly), inspecting recall for the 'Low' performance class ensures we minimize false negatives.

### Q8: What is `joblib` and why do we use it in this project?
> **Answer:** `joblib` is a Python library optimized for serializing large Python objects containing NumPy arrays. We use it to save our trained model as `model.pkl`. This decouples training from deployment, allowing the Streamlit web application to load the model instantly without retraining it on every page refresh.

### Q9: Which feature was the most important in predicting performance?
> **Answer:** Based on Random Forest's feature importance analysis, **Assignment Score**, **Previous Marks**, and **Daily Study Hours** had the highest relative importance, followed by **Attendance**.

### Q10: How could you improve this project in the future?
> **Answer:** 
> 1. Collect larger, real-world data from school management systems.
> 2. Implement hyperparameter tuning using `GridSearchCV`.
> 3. Add student feedback loops and exportable PDF counseling reports.
> 4. Deploy the application to cloud platforms such as Streamlit Community Cloud or AWS.

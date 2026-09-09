"""
app.py
------
Student Performance Prediction Web Application
Built with Streamlit and Scikit-Learn

This web application allows teachers, students, or evaluators to enter
academic and lifestyle metrics of a student and predict their overall
performance category ('Good', 'Average', or 'Low') using a trained Random Forest model.
"""

import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .card {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #E2E8F0;
        margin-bottom: 20px;
    }
    .result-good {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
    }
    .result-average {
        background: linear-gradient(135deg, #F59E0B 0%, #D97706 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.2);
    }
    .result-low {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 24px;
        border-radius: 12px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: bold;
        box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2);
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        background: #E2E8F0;
        color: #334155;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model
@st.cache_resource
def load_model():
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)

model = load_model()

# Sidebar: Project Details & Quick Presets
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523240795612-9a054b0db644?w=500&auto=format&fit=crop&q=60", use_container_width=True)
    st.markdown("### 🎓 About Project")
    st.write(
        "This project uses a **Random Forest Classifier** trained on academic & lifestyle "
        "factors to predict if a student's performance will be **Good**, **Average**, or **Low**."
    )
    
    st.markdown("---")
    st.markdown("### ⚡ Quick Presets")
    preset = st.radio(
        "Load example student profiles:",
        ["Custom Input", "High Achiever Profile", "Average Student Profile", "At-Risk Student Profile"]
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - **Language:** Python 3.12
    - **Libraries:** Scikit-learn, Pandas, NumPy
    - **Algorithm:** Random Forest Classifier
    - **Frontend:** Streamlit
    """)

# Preset values configuration
if preset == "High Achiever Profile":
    def_study, def_att, def_prev, def_assign, def_sleep, def_activity = 8.0, 95.0, 90.0, 92.0, 7.5, "Yes"
elif preset == "Average Student Profile":
    def_study, def_att, def_prev, def_assign, def_sleep, def_activity = 5.0, 78.0, 70.0, 72.0, 7.0, "Yes"
elif preset == "At-Risk Student Profile":
    def_study, def_att, def_prev, def_assign, def_sleep, def_activity = 2.0, 52.0, 42.0, 45.0, 5.0, "No"
else:
    def_study, def_att, def_prev, def_assign, def_sleep, def_activity = 5.5, 82.0, 74.0, 76.0, 7.0, "Yes"

# Main Page Header
st.markdown('<div class="main-title">🎓 Student Performance Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Predict academic outcomes based on study habits, attendance, and lifestyle metrics using Machine Learning.</div>', unsafe_allow_html=True)

# Warning if model file is missing
if model is None:
    st.error("⚠️ Model file (`model.pkl`) not found! Please run `python train_model.py` in your terminal to train and save the model first.")
    st.stop()

# Input Form in Two Columns
with st.container():
    st.markdown("#### 📝 Enter Student Information")
    col1, col2 = st.columns(2)

    with col1:
        study_hours = st.slider(
            "📖 Daily Study Hours",
            min_value=1.0,
            max_value=12.0,
            value=float(def_study),
            step=0.5,
            help="Average hours spent studying per day outside class"
        )
        
        attendance = st.slider(
            "📅 Attendance Percentage (%)",
            min_value=40.0,
            max_value=100.0,
            value=float(def_att),
            step=1.0,
            help="Overall attendance in current semester"
        )
        
        previous_marks = st.number_input(
            "📊 Previous Exam Marks (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=float(def_prev),
            step=1.0,
            help="Marks obtained in previous semester / exams"
        )

    with col2:
        assignment_score = st.number_input(
            "📑 Assignment Score (out of 100)",
            min_value=0.0,
            max_value=100.0,
            value=float(def_assign),
            step=1.0,
            help="Average score scored in internal continuous assignments"
        )
        
        sleep_hours = st.slider(
            "😴 Daily Sleep Hours",
            min_value=4.0,
            max_value=10.0,
            value=float(def_sleep),
            step=0.5,
            help="Average sleeping hours per night"
        )
        
        activity_choice = st.selectbox(
            "⚽ Participation in Extracurricular Activities",
            options=["Yes", "No"],
            index=0 if def_activity == "Yes" else 1,
            help="Active participation in sports, clubs, or cultural events"
        )

st.markdown("<br>", unsafe_allow_html=True)

# Prediction Button
predict_btn = st.button("🔮 Predict Performance", type="primary", use_container_width=True)

if predict_btn:
    # 1. Preprocess inputs
    # Encode activity: Yes -> 1, No -> 0
    activity_encoded = 1 if activity_choice == "Yes" else 0
    
    # Feature columns order:
    # ['Study_Hours', 'Attendance', 'Previous_Marks', 'Assignment_Score', 'Sleep_Hours', 'Extracurricular_Activities']
    input_data = pd.DataFrame([{
        'Study_Hours': study_hours,
        'Attendance': attendance,
        'Previous_Marks': previous_marks,
        'Assignment_Score': assignment_score,
        'Sleep_Hours': sleep_hours,
        'Extracurricular_Activities': activity_encoded
    }])
    
    # 2. Predict class & probabilities
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    prob_dict = {cls: prob * 100 for cls, prob in zip(classes, probabilities)}

    st.markdown("---")
    st.subheader("🎯 Prediction Outcome")
    
    # Display Result Card
    res_col, details_col = st.columns([1.2, 1.8])
    
    with res_col:
        if prediction == "Good":
            st.markdown("""
            <div class="result-good">
                🌟 Predicted: GOOD
            </div>
            """, unsafe_allow_html=True)
            st.success("The student is on track for outstanding academic achievement!")
        elif prediction == "Average":
            st.markdown("""
            <div class="result-average">
                ⚖️ Predicted: AVERAGE
            </div>
            """, unsafe_allow_html=True)
            st.warning("The student maintains steady performance with solid growth potential.")
        else:
            st.markdown("""
            <div class="result-low">
                ⚠️ Predicted: LOW
            </div>
            """, unsafe_allow_html=True)
            st.error("The student is currently at risk and requires timely guidance and support.")

    with details_col:
        st.markdown("##### 📈 Model Confidence Distribution")
        for cls in ['Good', 'Average', 'Low']:
            if cls in prob_dict:
                val = prob_dict[cls]
                st.write(f"**{cls}**: {val:.1f}%")
                st.progress(int(round(val)))

    # Actionable Insights & Recommendations
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("💡 Actionable Recommendations for this Student", expanded=True):
        if prediction == "Good":
            st.markdown("""
            - 🏆 **Maintain Routine:** Continue balanced study routines and assignment completion consistency.
            - 🚀 **Peer Mentorship:** Encourage mentoring fellow classmates to reinforce conceptual mastery.
            - 🔬 **Advanced Challenges:** Explore competitive exams, advanced projects, or research papers.
            """)
        elif prediction == "Average":
            st.markdown("""
            - 📈 **Incremental Study Hours:** Aim to increase productive study time by 1 to 1.5 hours daily.
            - 🎯 **Target Weak Areas:** Focus on assignments and previous examination mistakes.
            - 📅 **Improve Attendance:** Raising attendance to 85%+ significantly improves retention and scores.
            """)
        else:
            st.markdown("""
            - 🚨 **Immediate Academic Mentoring:** Schedule one-on-one sessions with subject teachers.
            - ⏰ **Structured Study Schedule:** Create a realistic daily study timetable starting with 3-4 focused hours.
            - 📋 **Assignment Catch-up:** Complete backlog internal assignments to secure essential passing marks.
            - 😴 **Sleep & Wellness:** Ensure at least 7 hours of regular sleep to improve focus and cognitive retention.
            """)

# Expandable Section for Dataset & Model Inspection
st.markdown("---")
with st.expander("🔍 View Dataset & Model Information"):
    tabs = st.tabs(["Dataset Sample", "Feature Importance Chart", "Model Architecture"])
    
    with tabs[0]:
        if os.path.exists("student_data.csv"):
            sample_df = pd.read_csv("student_data.csv")
            st.write(f"Total records in dataset: **{len(sample_df)}**")
            st.dataframe(sample_df.head(10), use_container_width=True)
        else:
            st.info("Dataset file `student_data.csv` is not present in the current folder.")
            
    with tabs[1]:
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", caption="Feature Importance in Random Forest Model", use_container_width=True)
        else:
            st.info("Run `python train_model.py` to generate the feature importance chart.")
            
    with tabs[2]:
        st.markdown("""
        - **Model:** `RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)`
        - **Target Classes:** `Good`, `Average`, `Low`
        - **Features (6):** `Study_Hours`, `Attendance`, `Previous_Marks`, `Assignment_Score`, `Sleep_Hours`, `Extracurricular_Activities`
        - **Ensemble Method:** Bagging (Bootstrap Aggregation) combining 100 Decision Trees.
        """)

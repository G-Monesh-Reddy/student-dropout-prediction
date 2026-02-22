import streamlit as st
import pandas as pd
import joblib
import io
import os
from dotenv import load_dotenv
import google.generativeai as genai



load_dotenv()

API_KEY = None
USE_GEMINI = False

# Check if running on Streamlit Cloud
if os.path.exists(".streamlit/secrets.toml"):
    try:
        API_KEY = st.secrets.get("GEMINI_API_KEY")
    except Exception:
        API_KEY = None

# Fallback to local .env
if not API_KEY:
    API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
if API_KEY:
    genai.configure(api_key=API_KEY)
    gemini_model = genai.GenerativeModel("gemini-2.5-flash")
    USE_GEMINI = True
# ===============================
# Load Model
# ===============================
model_data = joblib.load("final_best_model.joblib")
model = model_data["pipeline"]
threshold = model_data["threshold"]

# ===============================
# Page Config
# ===============================
st.set_page_config(page_title="Student Dropout Prediction", layout="wide")
st.title("🎓 Student Dropout Prediction System")

mode = st.sidebar.radio(
    "Select Usage Mode",
    ["Faculty Mode (Upload File)", "Student Mode (Manual Input)"]
)

# ===============================
# Demo Dataset Loader
# ===============================
def generate_demo_file():
    try:
        return pd.read_csv("test_students1.csv")
    except Exception:
        st.error("Demo dataset not found. Ensure test_students1.csv is in project root.")
        return pd.DataFrame()

# ===============================
# Counselling Logic
# ===============================
def rule_based_counselling(prediction):
    if prediction == "Dropout":
        return (
            "Student is at risk of dropout. Improve grades, clear pending subjects, "
            "seek mentoring support, and maintain fee regularity."
        )
    return (
        "Student performance is satisfactory. Maintain consistency and stay engaged academically."
    )

def gemini_counselling(student_data, prediction, probability):
    if not USE_GEMINI:
        return rule_based_counselling(prediction)

    prompt = f"""
    You are an academic counselor.

    Student details:
    {student_data}

    Dropout prediction: {prediction}
    Probability: {round(probability, 2)}

    Explain the main risk factors in simple English
    and give 3 practical academic improvement suggestions.
    """

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception:
        return rule_based_counselling(prediction)

# ===============================
# Faculty Mode
# ===============================
if mode == "Faculty Mode (Upload File)":

    st.header("📂 Faculty Mode – Upload Student File")

    demo_df = generate_demo_file()

    if not demo_df.empty:
        excel_buffer = io.BytesIO()
        demo_df.to_excel(excel_buffer, index=False, engine="xlsxwriter")

        st.download_button(
            label="📥 Download Full Demo Excel File",
            data=excel_buffer.getvalue(),
            file_name="student_dropout_full_demo.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

        st.subheader("📊 Demo Dataset Preview")
        st.dataframe(demo_df.head())

    uploaded_file = st.file_uploader(
        "Upload CSV or Excel file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)

        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        if st.button("🔍 Predict Dropout"):
            try:
                probs = model.predict_proba(df)[:, 1]
                preds = (probs >= threshold).astype(int)

                df["dropout_probability"] = probs.round(3)
                df["prediction"] = ["Dropout" if p else "Not Dropout" for p in preds]
                df["counselling"] = df["prediction"].apply(rule_based_counselling)

                st.subheader("✅ Prediction Results")
                st.dataframe(df)

                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)

                st.download_button(
                    "📥 Download Prediction CSV",
                    csv_buffer.getvalue(),
                    "student_dropout_predictions.csv",
                    "text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ===============================
# Student Mode
# ===============================
else:
    st.header("🧑‍🎓 Student Mode – Individual Prediction")

    col1, col2 = st.columns(2)

    with col1:
        g1 = st.number_input("1st Semester Grade", 0.0, 10.0, 7.0)
        a1 = st.number_input("1st Semester Approved Subjects", 0, 10, 4)
        g2 = st.number_input("2nd Semester Grade", 0.0, 10.0, 7.0)
        a2 = st.number_input("2nd Semester Approved Subjects", 0, 10, 3)
        age = st.number_input("Age at Enrollment", 15, 60, 19)

    with col2:
        tuition = st.selectbox("Tuition Fees Status", [1, 0], format_func=lambda x: "Up to Date" if x else "Pending")
        debtor = st.selectbox("Outstanding Dues", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        scholarship = st.selectbox("Scholarship Holder", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
        app_mode = st.number_input("Application Mode Code", 1, 20, 1)

    if st.button("🎯 Predict"):
        input_df = pd.DataFrame([{
            "Curricular units 1st sem (grade)": g1,
            "Curricular units 1st sem (approved)": a1,
            "Curricular units 2nd sem (grade)": g2,
            "Curricular units 2nd sem (approved)": a2,
            "Tuition fees up to date": tuition,
            "Debtor": debtor,
            "Age at enrollment": age,
            "Scholarship holder": scholarship,
            "Application mode": app_mode
        }])

        try:
            prob = model.predict_proba(input_df)[0][1]
            label = "Dropout" if prob >= threshold else "Not Dropout"

            st.metric("Dropout Probability", f"{prob:.2f}")
            st.success(f"Prediction: {label}")

            st.subheader("🧠 AI-Based Academic Counselling")
            st.info(gemini_counselling(input_df.to_dict("records")[0], label, prob))

        except Exception as e:
            st.error(f"Prediction failed: {e}")
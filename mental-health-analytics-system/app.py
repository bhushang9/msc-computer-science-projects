import streamlit as st
import pandas as pd
import joblib
import os
from textblob import TextBlob
import warnings
from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings("ignore", category=DataConversionWarning)

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(
    page_title="Predictive Analytics for Mental Health",
    layout="centered"
)

# --------------------------------------------------
# Load paths
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS_DIR = os.path.join(BASE_DIR, "models")
ARTIFACTS_DIR = os.path.join(MODELS_DIR, "artifacts")

MODEL_PATH = os.path.join(ARTIFACTS_DIR, "best_model.pkl")
FEATURE_NAMES_PATH = os.path.join(ARTIFACTS_DIR, "feature_names.pkl")

ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")

# --------------------------------------------------
# Load artifacts
# --------------------------------------------------
model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURE_NAMES_PATH)

encoders = joblib.load(ENCODERS_PATH)
scaler = joblib.load(SCALER_PATH)

# --------------------------------------------------
# Title
# --------------------------------------------------
st.title("Predictive Analytics for Mental Health")
st.write("Fill in the details below to assess mental health risk.")

# --------------------------------------------------
# Inputs
# --------------------------------------------------
name = st.text_input("Name")
age = st.number_input("Age", min_value=18, max_value=100, value=25)

gender = st.selectbox(
    "Gender",
    ["male", "female", "trans"]
)

family_history = st.selectbox(
    "Family history of mental health issues?",
    ["Yes", "No"]
)

benefits = st.selectbox(
    "Employer provides mental health benefits?",
    ["Yes", "No"]
)

care_options = st.selectbox(
    "Aware of mental health care options?",
    ["Yes", "No"]
)

anonymity = st.selectbox(
    "Anonymity protected?",
    ["Yes", "No"]
)

leave = st.selectbox(
    "Ease of taking mental health leave",
    ["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult"]
)

work_interfere = st.selectbox(
    "Mental health affecting work?",
    ["Never", "Sometimes", "Often"]
)

journal_entry = st.text_area("Optional: Write how you're feeling")

# --------------------------------------------------
# Sentiment analysis
# --------------------------------------------------
def get_sentiment(text):
    if not text.strip():
        return "Neutral", 0.0
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "Positive", round(polarity, 2)
    elif polarity < -0.1:
        return "Negative", round(polarity, 2)
    return "Neutral", round(polarity, 2)

sentiment_label, sentiment_score = get_sentiment(journal_entry)

st.write(f"**Journal Sentiment:** {sentiment_label} ({sentiment_score})")

# --------------------------------------------------
# Prepare input
# --------------------------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "Gender": gender.lower(),
    "family_history": family_history,
    "benefits": benefits,
    "care_options": care_options,
    "anonymity": anonymity,
    "leave": leave,
    "work_interfere": work_interfere
}])

# Ensure correct column order
input_data = input_data[feature_names]

# --------------------------------------------------
# Encode categorical features (SAFE)
# --------------------------------------------------
for col in feature_names:
    if col in encoders:
        le = encoders[col]
        value = input_data[col].iloc[0]

        if value not in le.classes_:
            st.error(
                f"Invalid value '{value}' for '{col}'. "
                f"Expected one of {list(le.classes_)}"
            )
            st.stop()

        input_data[col] = le.transform([value])

# --------------------------------------------------
# Scale
# --------------------------------------------------
input_data["Age"] = scaler.transform(input_data[["Age"]])

input_scaled = input_data.copy()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Mental health treatment is advised.")
    else:
        st.success("Treatment may not be necessary.")

    st.write(f"**Confidence:**")
    st.write(f"Yes: {proba[1] * 100:.2f}%")
    st.write(f"No: {proba[0] * 100:.2f}%")

    report = pd.DataFrame({
        "Name": [name],
        "Age": [age],
        "Prediction": ["Yes" if prediction == 1 else "No"],
        "Confidence Yes (%)": [round(proba[1] * 100, 2)],
        "Confidence No (%)": [round(proba[0] * 100, 2)],
        "Sentiment": [sentiment_label],
        "Sentiment Score": [sentiment_score]
    })

    st.download_button(
        "Download Report",
        report.to_csv(index=False),
        file_name="mental_health_report.csv",
        mime="text/csv"
    )

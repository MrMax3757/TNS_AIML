import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "diabetes_model.pkl"

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error("Trained model not found. Please run train.py first.")
    st.stop()


# ==========================================
# DATASET ENCODING
# ==========================================

YES_NO = {
    "No": 0,
    "Yes": 1
}

SEX_OPTIONS = {
    "Female": 0,
    "Male": 1
}

AGE_OPTIONS = {
    "18–24 years": 1,
    "25–29 years": 2,
    "30–34 years": 3,
    "35–39 years": 4,
    "40–44 years": 5,
    "45–49 years": 6,
    "50–54 years": 7,
    "55–59 years": 8,
    "60–64 years": 9,
    "65–69 years": 10,
    "70–74 years": 11,
    "75–79 years": 12,
    "80 years or older": 13
}

EDUCATION_OPTIONS = {
    "Never attended school / Kindergarten only": 1,
    "Elementary school": 2,
    "Some high school": 3,
    "High school graduate": 4,
    "Some college or technical education": 5,
    "College graduate or higher": 6
}

INCOME_OPTIONS = {
    "Below $10,000": 1,
    "$10,000 – $14,999": 2,
    "$15,000 – $19,999": 3,
    "$20,000 – $24,999": 4,
    "$25,000 – $34,999": 5,
    "$35,000 – $49,999": 6,
    "$50,000 – $74,999": 7,
    "$75,000 or more": 8
}

GENERAL_HEALTH_OPTIONS = {
    "Excellent": 1,
    "Very Good": 2,
    "Good": 3,
    "Fair": 4,
    "Poor": 5
}


# ==========================================
# APP DESIGN
# ==========================================

st.title("🩺 Diabetes Prediction System")

st.markdown(
    """
    Enter the patient's information below to generate a
    machine learning prediction.

    This application is an educational project and does not
    replace professional medical advice.
    """
)

st.divider()


# ==========================================
# PATIENT INFORMATION FORM
# ==========================================

with st.form("patient_information"):

    # --------------------------------------
    # SECTION 1: PERSONAL INFORMATION
    # --------------------------------------

    st.header("👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:

        age_label = st.selectbox(
            "Age Group",
            list(AGE_OPTIONS.keys())
        )

        sex_label = st.selectbox(
            "Sex",
            list(SEX_OPTIONS.keys())
        )

        education_label = st.selectbox(
            "Education Level",
            list(EDUCATION_OPTIONS.keys())
        )

    with col2:

        income_label = st.selectbox(
            "Annual Household Income",
            list(INCOME_OPTIONS.keys())
        )

        bmi = st.number_input(
            "Body Mass Index (BMI)",
            min_value=10.0,
            max_value=100.0,
            value=28.0,
            step=0.1,
            help="Enter the patient's BMI."
        )

    st.divider()

    # --------------------------------------
    # SECTION 2: MEDICAL HISTORY
    # --------------------------------------

    st.header("🏥 Medical History")

    col1, col2 = st.columns(2)

    with col1:

        high_bp = st.selectbox(
            "Has High Blood Pressure?",
            list(YES_NO.keys())
        )

        high_chol = st.selectbox(
            "Has High Cholesterol?",
            list(YES_NO.keys())
        )

        chol_check = st.selectbox(
            "Cholesterol Checked Within the Last 5 Years?",
            list(YES_NO.keys())
        )

        stroke = st.selectbox(
            "History of Stroke?",
            list(YES_NO.keys())
        )

    with col2:

        heart_disease = st.selectbox(
            "History of Heart Disease or Heart Attack?",
            list(YES_NO.keys())
        )

        healthcare = st.selectbox(
            "Has Healthcare Coverage?",
            list(YES_NO.keys())
        )

        no_doc_cost = st.selectbox(
            "Unable to Visit a Doctor Due to Cost?",
            list(YES_NO.keys())
        )

        diff_walk = st.selectbox(
            "Has Serious Difficulty Walking or Climbing Stairs?",
            list(YES_NO.keys())
        )

    st.divider()

    # --------------------------------------
    # SECTION 3: LIFESTYLE INFORMATION
    # --------------------------------------

    st.header("🏃 Lifestyle Information")

    col1, col2 = st.columns(2)

    with col1:

        smoker = st.selectbox(
            "Smoked at Least 100 Cigarettes in Lifetime?",
            list(YES_NO.keys())
        )

        physical_activity = st.selectbox(
            "Physical Activity in the Last 30 Days?",
            list(YES_NO.keys())
        )

        fruits = st.selectbox(
            "Consumes Fruit at Least Once Per Day?",
            list(YES_NO.keys())
        )

    with col2:

        veggies = st.selectbox(
            "Consumes Vegetables at Least Once Per Day?",
            list(YES_NO.keys())
        )

        alcohol = st.selectbox(
            "Heavy Alcohol Consumption?",
            list(YES_NO.keys()),
            help="Uses the dataset's definition of heavy alcohol consumption."
        )

    st.divider()

    # --------------------------------------
    # SECTION 4: HEALTH ASSESSMENT
    # --------------------------------------

    st.header("❤️ Health Assessment")

    col1, col2 = st.columns(2)

    with col1:

        general_health_label = st.selectbox(
            "How Would You Describe Your General Health?",
            list(GENERAL_HEALTH_OPTIONS.keys())
        )

        mental_health = st.slider(
            "Days Mental Health Was Not Good (Last 30 Days)",
            min_value=0,
            max_value=30,
            value=5
        )

    with col2:

        physical_health = st.slider(
            "Days Physical Health Was Not Good (Last 30 Days)",
            min_value=0,
            max_value=30,
            value=5
        )

    st.divider()

    # --------------------------------------
    # SUBMIT
    # --------------------------------------

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


# ==========================================
# PREDICTION
# ==========================================

if submitted:

    # Convert user-friendly selections
    # into the original dataset encodings.

    patient_data = {

        "HighBP": YES_NO[high_bp],

        "HighChol": YES_NO[high_chol],

        "CholCheck": YES_NO[chol_check],

        "BMI": bmi,

        "Smoker": YES_NO[smoker],

        "Stroke": YES_NO[stroke],

        "HeartDiseaseorAttack": YES_NO[heart_disease],

        "PhysActivity": YES_NO[physical_activity],

        "Fruits": YES_NO[fruits],

        "Veggies": YES_NO[veggies],

        "HvyAlcoholConsump": YES_NO[alcohol],

        "AnyHealthcare": YES_NO[healthcare],

        "NoDocbcCost": YES_NO[no_doc_cost],

        "GenHlth": GENERAL_HEALTH_OPTIONS[general_health_label],

        "MentHlth": mental_health,

        "PhysHlth": physical_health,

        "DiffWalk": YES_NO[diff_walk],

        "Sex": SEX_OPTIONS[sex_label],

        "Age": AGE_OPTIONS[age_label],

        "Education": EDUCATION_OPTIONS[education_label],

        "Income": INCOME_OPTIONS[income_label]
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([patient_data])

    # Match the exact feature order used during training.
    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)

        missing_features = [
            feature for feature in expected_features
            if feature not in input_df.columns
        ]

        if missing_features:
            st.error(
                f"Missing model features: {missing_features}"
            )
            st.stop()

        input_df = input_df[expected_features]

    # Generate prediction
    prediction = model.predict(input_df)[0]

    # Generate class probabilities
    probabilities = model.predict_proba(input_df)[0]

    classes = list(model.classes_)

    # Find the probability corresponding to each class.
    probability_map = dict(zip(classes, probabilities))

    diabetes_probability = probability_map.get(1, 0)
    no_diabetes_probability = probability_map.get(0, 0)

    st.divider()
    st.header("📊 Prediction Results")

    col1, col2 = st.columns(2)

    with col1:

        if prediction == 1:

            st.error("Diabetes Predicted")

            st.markdown(
                "The model classified this patient's "
                "information into the diabetes category."
            )

        else:

            st.success("No Diabetes Predicted")

            st.markdown(
                "The model classified this patient's "
                "information into the no-diabetes category."
            )

    with col2:

        st.metric(
            "Model Probability — Diabetes",
            f"{diabetes_probability * 100:.2f}%"
        )

        st.metric(
            "Model Probability — No Diabetes",
            f"{no_diabetes_probability * 100:.2f}%"
        )

    st.subheader("Model Probability Distribution")

    st.progress(float(diabetes_probability))

    st.caption(
        "These percentages represent the model's estimated "
        "class probabilities, not clinically validated "
        "individual diabetes risk."
    )

# ==========================================
# DISCLAIMER
# ==========================================

st.divider()

st.caption(
    """
    **Medical Disclaimer:** This application is developed
    for educational and machine learning demonstration
    purposes only. It is not a medical diagnostic tool.
    Predictions should not be used to make healthcare
    decisions. Consult a qualified healthcare professional
    for medical assessment.
    """
)

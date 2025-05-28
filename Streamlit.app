import streamlit as st
import numpy as np
import joblib

# Load model and scaler
clf = joblib.load("malware_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Malware Classifier", layout="centered")
st.title("🛡️ Malware Detection Classifier")

st.markdown("""
Upload feature values from malware data and predict if it's malicious or benign.
Please ensure the feature count matches the trained model's input size.
""")

# Create input form
with st.form("malware_form"):
    input_values = st.text_area("Enter feature values (comma-separated):", "0.5, -1.2, 3.1, ...")
    submitted = st.form_submit_button("Predict")

if submitted:
    try:
        # Process input
        values = np.array([float(i) for i in input_values.strip().split(",")]).reshape(1, -1)
        values_scaled = scaler.transform(values)

        # Prediction
        prediction = clf.predict(values_scaled)[0]
        prediction_label = "Malware" if prediction == 1 else "Benign"

        st.success(f"Prediction: **{prediction_label}**")

    except Exception as e:
        st.error(f"Error: {str(e)}")

st.markdown("---")
st.caption("Developed with ❤️ using Streamlit and scikit-learn")

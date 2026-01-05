import pandas as pd
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# -----------------------------
# Load dataset
# -----------------------------
movie_info = pd.read_csv("Movie_Interests_DecisionTree.csv")

X = movie_info.drop(columns=["Interest"])
y = movie_info["Interest"]

MODEL_FILE = "movie_interest_model.pkl"

# -----------------------------
# Train & save model (only once)
# -----------------------------
if not os.path.exists(MODEL_FILE):
    model = DecisionTreeClassifier()
    model.fit(X, y)
    joblib.dump(model, MODEL_FILE)

# -----------------------------
# Load model
# -----------------------------
movie_model = joblib.load(MODEL_FILE)

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🎬 Movie Interest Prediction App")

st.write("Enter user details to predict movie interest")

age = st.number_input("Age", min_value=1, max_value=100, value=25)
gender = st.selectbox("Gender", ["Male", "Female"])

# Convert gender to numeric
gender_value = 1 if gender == "Male" else 0

if st.button("Predict Interest"):
    prediction = movie_model.predict([[age, gender_value]])

    if prediction[0] == 1:
        st.success("✅ User is interested in movies!")
    else:
        st.warning("❌ User is not interested in movies.")

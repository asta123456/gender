import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Gender Classification")

st.title("Gender Classification App")

st.write("App Loaded Successfully")

uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read CSV
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        # Lowercase columns
        df.columns = df.columns.str.lower()

        # Check gender column
        if "gender" not in df.columns:
            st.error("Dataset must contain gender column")
            st.stop()

        # Convert gender values
        df["gender"] = df["gender"].map({
            "Male": 1,
            "Female": 0
        })

        # Remove null values
        df.dropna(inplace=True)

        # Features and target
        X = df.drop("gender", axis=1)
        y = df["gender"]

        # Convert categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # Scaling
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Model
        model = LogisticRegression()

        model.fit(X_train, y_train)

        # Prediction
        y_pred = model.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)

        st.success(f"Model Accuracy: {accuracy:.2f}")

    except Exception as e:
        st.error(f"Error: {e}")

else:
    st.info("Please upload CSV file")

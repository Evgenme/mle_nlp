import streamlit as st
from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

st.title("ML Project with Streamlit")

data_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if data_file is not None:
    data = load_data(data_file)
    st.write("Raw Data", data.head())

    processed_data = preprocess_data(data)
    st.write("Processed Data", processed_data.head())

    target_column = st.selectbox("Select target column", processed_data.columns)
    if st.button("Train Model"):
        model, X_test, y_test = train_model(processed_data, target_column)
        accuracy = evaluate_model(model, X_test, y_test)
        st.write(f"Model Accuracy: {accuracy}")

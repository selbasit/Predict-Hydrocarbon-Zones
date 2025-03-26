import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import numpy as np

st.title("Hydrocarbon Zone Predictor")

uploaded_file = st.file_uploader("Upload Well Log CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Sample", df.head())

    if st.button("Predict Hydrocarbon Zones"):
        # Load model
        model = load_model("hc_cnn_model.h5")

        # Preprocess input
        X = df.drop(columns=["DEPTH"])  # make sure only features are kept
        X_cnn = X.values.reshape((X.shape[0], 1, X.shape[1]))

        # Predict
        preds = model.predict(X_cnn)
        df['Predicted_Prob'] = preds
        df['Predicted_Label'] = (preds > 0.5).astype(int)

        st.success("Prediction complete!")
        st.write(df[['DEPTH', 'Predicted_Prob', 'Predicted_Label']])


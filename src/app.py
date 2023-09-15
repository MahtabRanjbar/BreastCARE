import os
import pickle

import numpy as np
import pandas as pd
import streamlit as st

import model as md
import preprocessing
import visualization


def add_sidebar():
    st.sidebar.header("Nuclear Size Quantification")

    data = preprocessing.get_clean_data()
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    return {
        key: st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )
        for label, key in slider_labels
    }


def add_predictions(input_data):
    model, scaler = md.load_model()

    input_array = np.array(list(input_data.values())).reshape(1, -1)

    input_array_scaled = scaler.transform(input_array)

    prediction = model.predict(input_array_scaled)

    st.subheader("Cell prediction")
    st.write("The cell Type is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True)
    else:
        st.write(
            "<span class='diagnosis malicious'>Malicious</span>", unsafe_allow_html=True
        )

    st.write(
        "Probability of being Benign: ", model.predict_proba(input_array_scaled)[0][0]
    )
    st.write(
        "Probability of being Malicious: ",
        model.predict_proba(input_array_scaled)[0][1],
    )

    st.write(
        "This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for a professional diagnosis."
    )


def main():
    st.set_page_config(
        page_title="BreastCARE",
        page_icon=":heart:",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("BreastCARE: Breast Cancer Predictor")
    st.markdown(
        "This application predicts whether a breast tumor is malignant or benign based on various features."
    )

    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("BreastCARE: Breast Cancer Predictor")
        st.write(
            "This application combines cell nuclei measurement with machine learning\
            techniquesto predict breast cancer likelihood accurately. It provides healthcare \
            professionals with advanced tools for precise cell nuclei analysis, contributing to\
            improved breast cancer prediction and diagnosis.You can also update the measurement \
            by hand in the sidebar."
        )

    col1, col2 = st.columns([4, 1])
    with col1:
        radar_chart = visualization.get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)

    st.title("Report Generator")

    # Generate reports on button click
    st.markdown(
        """
    Here, you can see the evaluation metrics for my model. The **Confusion Matrix** provides a summary of the model's predictions, showing the number of true positives, true negatives, false positives, and false negatives. The **Classification Report** includes precision, recall, F1-score, and support for each class. Lastly, the **ROC Curve** displays the trade-off between the true positive rate and false positive rate, indicating the model's performance in distinguishing between classes. These reports help assess the model's accuracy and effectiveness in making predictions.
    """
    )
    if st.button("Generate Reports"):

        # Display reports
        report_folder = "report"
        st.subheader("Classification Report")
        with open(os.path.join(report_folder, "classification_report.txt"), "r") as f:
            classification_rep = f.read()
        st.text(classification_rep)

        st.subheader("Confusion Matrix")
        st.image(os.path.join(report_folder, "confusion_matrix.png"))

        st.subheader("ROC Curve")
        st.image(os.path.join(report_folder, "roc_curve.png"))


if __name__ == "__main__":
    main()

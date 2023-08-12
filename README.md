# BreastCARE: Breast Cancer Predictor

BreastCARE is a web application that predicts whether a breast tumor is malignant or benign based on various features. It uses machine learning techniques and cell nuclei measurements to accurately predict breast cancer likelihood.

[Here](./assets/images/) you can see a demo of my app


## Dataset

The dataset used for training and testing the model in BreastCARE is the **Breast Cancer Wisconsin (Diagnostic) Data Set** from the UCI Machine Learning Repository. This dataset contains measurements of various features extracted from digitized images of fine needle aspirate (FNA) of breast masses. The features describe characteristics of the cell nuclei present in the images, such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension. Each sample in the dataset is labeled as either benign or malignant.

For more information about the dataset, you can refer to the [Kaggle Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data).

## Features

- Interactive Sidebar: Allows users to input cell nuclei measurements and update them manually.
- Radar Chart Visualization: Displays a radar chart representing the input data and its distribution across different features.
- Predictions: Provides the predicted cell cluster (benign or malicious) and the probabilities of being benign or malicious.
- Evaluation Reports: Generates evaluation metrics reports for the model, including a confusion matrix, classification report, and ROC curve.

## Evaluation Reports

After generating the evaluation reports, you can find them in the `report` folder. The reports provide insights into the model's performance and accuracy in predicting breast cancer likelihood. The available reports include:

- **confusion_matrix.png**: A visual representation of the confusion matrix, summarizing the model's predictions and the actual labels.
- **classification_report.txt**: A text file containing the classification report, which includes precision, recall, F1-score, and support for each class.
- **roc_curve.png**: A visual representation of the ROC curve, showing the trade-off between the true positive rate and false positive rate.

You can also find a detailed report in the `report.md` file, which provides an overview of the evaluation metrics and their interpretation.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MahtabRanjbar/BreastCARE.git ↗
```
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run main.py
```

## Usage

- Access the application by opening the provided URL in your web browser.
- Update the cell nuclei measurements in the sidebar to see the prediction and probabilities change accordingly.
- Click on the "Generate Reports" button to generate evaluation metrics reports for the model.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvement, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
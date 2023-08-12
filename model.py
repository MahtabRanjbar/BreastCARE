import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import evaluation
import preprocessing


def create_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


def predict(X_test, model):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    return y_pred, y_prob


def save_model(model, scaler):
    with open("Model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("Model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load_model():
    with open("Model/model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("Model/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def main():
    data = preprocessing.get_clean_data()

    X_train, X_test, y_train, y_test, scaler = preprocessing.prepare_data(data)

    model = create_model(X_train, y_train)
    y_pred, y_prob = predict(X_test, model)
    print(y_pred.shape, y_test.shape)
    print(y_pred[:10])
    print(y_test[:10].tolist())
    print(y_prob[:10])
    evaluation.save_report(y_pred, y_prob, y_test.tolist())
    save_model(model, scaler)


if __name__ == "__main__":
    main()

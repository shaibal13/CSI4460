from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np


class LogisticRegressionModel:
    def __init__(self):
        # class_weight='balanced' handles class imbalance
        self.model = LogisticRegression(max_iter=5000, class_weight="balanced")

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)

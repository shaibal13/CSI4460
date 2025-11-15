import xgboost as xgb
import numpy as np


class XGBoostModel:
    def __init__(self, pos_weight_ratio: float):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=10,
            learning_rate=0.001,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=pos_weight_ratio,
            n_jobs=-1
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str):
        self.model.save_model(path)

    def load(self, path: str):
        self.model.load_model(path)

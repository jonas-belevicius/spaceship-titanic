import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from typing import List

from sklearn.pipeline import make_pipeline
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
)
from catboost import CatBoostClassifier
from sklearn.model_selection import cross_val_score


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

import optuna

X_train = pd.read_csv("../data/processed_x_train_data.csv")
y_train = pd.read_csv("../data/y_train.csv")


def optuna_objective(trial):
    model_name = trial.suggest_categorical(
        "model",
        [
            "XGBoost_optuned",
            "LGBMClassifier_optuned",
            "RandomForestClassifier_optuned",
            "AdaBoostClassifier_optuned",
            "BaggingClassifier_optuned",
            "CatBoostClassifier_optuned",
        ],
    )

    if model_name == "XGBoost_optuned":
        param = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("xgb_max_depth", 3, 10),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample_bytree", 0.5, 1.0),
        }
        model = XGBClassifier(**param, verbose=0)

    elif model_name == "LGBMClassifier_optuned":
        param = {
            "n_estimators": trial.suggest_int("lgbm_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("lgbm_max_depth", -1, 15),
            "learning_rate": trial.suggest_float("lgbm_learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("lgbm_num_leaves", 20, 150),
        }
        model = LGBMClassifier(**param, random_state=1, verbose=-1)

    elif model_name == "RandomForestClassifier_optuned":
        param = {
            "n_estimators": trial.suggest_int("rf_n_estimators", 50, 200),
            "max_depth": trial.suggest_int("rf_max_depth", 3, 15),
            "max_features": trial.suggest_categorical(
                "rf_max_features", ["sqrt", "log2"]
            ),
            "min_samples_split": trial.suggest_int("rf_min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("rf_min_samples_leaf", 1, 4),
        }
        model = RandomForestClassifier(**param, random_state=1)

    elif model_name == "AdaBoostClassifier_optuned":
        param = {
            "n_estimators": trial.suggest_int("ada_n_estimators", 50, 200),
            "learning_rate": trial.suggest_float("ada_learning_rate", 0.01, 1.0),
        }
        model = AdaBoostClassifier(**param, random_state=1)

    elif model_name == "BaggingClassifier_optuned":
        param = {
            "n_estimators": trial.suggest_int("bag_n_estimators", 10, 100),
            "max_samples": trial.suggest_float("bag_max_samples", 0.1, 1.0),
            "max_features": trial.suggest_float("bag_max_features", 0.1, 1.0),
        }
        model = BaggingClassifier(**param, random_state=1)

    elif model_name == "CatBoostClassifier_optuned":
        param = {
            "iterations": trial.suggest_int("iterations", 100, 1000),
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "bagging_temperature": trial.suggest_loguniform(
                "bagging_temperature", 0.01, 1.0
            ),
        }
        model = CatBoostClassifier(**param, verbose=0, random_state=1)

    model_pipeline = make_pipeline(model)
    accuracy = cross_val_score(
        model_pipeline, X_train, y_train, n_jobs=-1, cv=5, scoring="accuracy"
    ).mean()

    return accuracy


if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(optuna_objective, n_trials=100)
    print(f"Best Model and Parameters found: {study.best_params}")

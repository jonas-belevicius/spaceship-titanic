import sys
import os

sys.path.append(os.path.abspath(os.path.join("..", "utils")))

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import functions as fu
import plotters as pl

import warnings
from typing import List

from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from transformers import (
    ColumnRemover,
    ColumnImputer,
    ColumnBinner,
    ColumnSplitter,
    PassengerGroupper,
    NoServiceUsers,
    RegressionImputer,
    SimpleColumnImputer,
    ChangeDtype,
    DropMissing,
    FamilyMemberCounter,
    CombinedSpendingCounter,
    LogTransformer,
)

# define columns to split, bin and remove during preprocessing of the dataset
columns_split_into1 = ["PassengerGroup", "PassengerNumber"]
columns_split_into2 = ["CabinDeck", "CabinNum", "CabinSide"]
columns_split_into3 = ["Name", "Surname"]

columns_to_log = [
    "RoomService",
    "FoodCourt",
    "ShoppingMall",
    "Spa",
    "VRDeck",
    "CombinedSpending",
]
numeric_cols_to_transform = [
    "Age",
    "CabinNum",
    "PassengersPerGroup",
    "FamilyMembers",
    "RoomServiceLog",
    "FoodCourtLog",
    "ShoppingMallLog",
    "SpaLog",
    "VRDeckLog",
]
cat_cols_to_transform = [
    "HomePlanet",
    "CryoSleep",
    "Destination",
    #   "VIP",
    "CabinDeck",
    "CabinSide",
]
columns_to_remove = (
    columns_to_log + columns_split_into1 + ["PassengerId", "Cabin", "Name"]
)


def split_columns():
    return Pipeline(
        [
            (
                "split-passenger-id",
                ColumnSplitter("PassengerId", columns_split_into1, "_"),
            ),
            ("split-cabin", ColumnSplitter("Cabin", columns_split_into2, "/")),
            ("split-name", ColumnSplitter("Name", columns_split_into3, " ")),
        ]
    )


def bin_age():
    return Pipeline([("bin-age", ColumnBinner(["Age"], bins=5, apply_log=False))])


def log_transform():
    return Pipeline([("log-transform-services", LogTransformer(columns_to_log))])


def add_new_column():
    return Pipeline(
        [
            (
                "add-no-service-users",
                NoServiceUsers(
                    ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"],
                    "NoServicesUsed",
                ),
            ),
        ]
    )


def count_family_members():
    return Pipeline([("family-member-counter", FamilyMemberCounter())])


def count_total_spending():
    return Pipeline([("total-spending", CombinedSpendingCounter())])


def group_passengers():
    return Pipeline(
        [
            ("passenger-groupper", PassengerGroupper()),
        ]
    )


def change_dtype():
    return Pipeline(
        [("passenger-group-type-change", ChangeDtype("PassengerGroup", "float64"))]
    )


def impute_missing_values():
    return Pipeline(
        [
            (
                "age-impute-by-deck",
                ColumnImputer(["CabinDeck"], "Age", strategy="median"),
            ),
            ("deck-impute-by-group", ColumnImputer(["PassengerGroup"], "CabinDeck")),
            ("deck-impute-by-planet", ColumnImputer(["HomePlanet"], "CabinDeck")),
            (
                "cabin-num-impute-group-deck",
                RegressionImputer("CabinNum", "PassengerGroup", "CabinDeck"),
            ),
            ("side-impute-group", ColumnImputer(["PassengerGroup"], "CabinSide")),
            (
                "homeplanet-impute-group",
                ColumnImputer(["PassengerGroup"], "HomePlanet"),
            ),
            ("homeplanet-impute-deck", ColumnImputer(["CabinDeck"], "HomePlanet")),
            (
                "cryosleep-impute-no-services",
                ColumnImputer(["NoServicesUsed"], "CryoSleep"),
            ),
            (
                "room-service-impute-combined",
                ColumnImputer(
                    ["CryoSleep", "HomePlanet", "AgeCut", "CabinDeck"],
                    "RoomService",
                    strategy="mean",
                ),
            ),
            (
                "food-court-impute-combined",
                ColumnImputer(
                    ["CryoSleep", "HomePlanet", "AgeCut", "CabinDeck"],
                    "FoodCourt",
                    strategy="mean",
                ),
            ),
            (
                "shopping-mall-impute-combined",
                ColumnImputer(
                    ["CryoSleep", "HomePlanet", "AgeCut", "CabinDeck"],
                    "ShoppingMall",
                    strategy="mean",
                ),
            ),
            (
                "spa-impute-combined",
                ColumnImputer(
                    ["CryoSleep", "HomePlanet", "AgeCut", "CabinDeck"],
                    "Spa",
                    strategy="mean",
                ),
            ),
            (
                "vr-deck-impute-combined",
                ColumnImputer(
                    ["CryoSleep", "HomePlanet", "AgeCut", "CabinDeck"],
                    "VRDeck",
                    strategy="mean",
                ),
            ),
        ]
    )


def impute_simple_imputer():
    return Pipeline(
        [
            ("age-impute-by-median", SimpleColumnImputer("Age", strategy="median")),
            (
                "deck-impute-frequent",
                SimpleColumnImputer("CabinDeck", strategy="most_frequent"),
            ),
            (
                "side-impute-frequent",
                SimpleColumnImputer("CabinSide", strategy="most_frequent"),
            ),
            (
                "destination-impute-most-frequent",
                SimpleColumnImputer("Destination", strategy="most_frequent"),
            ),
            # (
            #     "vip-impute-most-frequent",
            #     SimpleColumnImputer("VIP", strategy="most_frequent"),
            # ),
            (
                "roomservice-impute-median",
                SimpleColumnImputer("RoomService", strategy="median"),
            ),
            (
                "foodcourt-impute-median",
                SimpleColumnImputer("FoodCourt", strategy="median"),
            ),
            (
                "shoppingmall-impute-median",
                SimpleColumnImputer("ShoppingMall", strategy="median"),
            ),
            (
                "spa-impute-median",
                SimpleColumnImputer("Spa", strategy="median"),
            ),
            (
                "crdeck-impute-median",
                SimpleColumnImputer("VRDeck", strategy="median"),
            ),
            (
                "cabin-num-impute-median",
                SimpleColumnImputer("CabinNum", strategy="median"),
            ),
            (
                "homplanet-impute-most-frequent",
                SimpleColumnImputer("HomePlanet", strategy="most_frequent"),
            ),
            (
                "family-members-impute-median",
                SimpleColumnImputer("FamilyMembers", strategy="median"),
            ),
        ]
    )


def remove_columns():
    return Pipeline(
        [
            ("column-remover", ColumnRemover(columns_to_remove)),
        ]
    )


def numeric_preprocessing():
    return Pipeline([("robust-scaler", RobustScaler())])


def preprocess_columns():
    return ColumnTransformer(
        [
            (
                "numeric-preprocessor",
                numeric_preprocessing(),
                numeric_cols_to_transform,
            ),
            (
                "categorical-preprocessor",
                OneHotEncoder(drop="if_binary", sparse_output=False),
                cat_cols_to_transform,
            ),
        ]
    )


def create_full_preprocessing_pipeline():
    return Pipeline(
        [
            ("remove-vip", ColumnRemover(["VIP"])),
            ("split-columns", split_columns()),
            ("add-new-column", add_new_column()),
            ("group-passengers", group_passengers()),
            ("count_family_members", count_family_members()),
            ("change-dtype", change_dtype()),
            ("bin-age", bin_age()),
            ("impute-missing-values", impute_missing_values()),
            ("impute-simple-imputer", impute_simple_imputer()),
            ("count_total_spending", count_total_spending()),
            ("log-services", log_transform()),
            ("remove-columns", remove_columns()),
            ("preprocess-columns", preprocess_columns()),
            ("drop-missing", DropMissing()),
        ]
    )


if __name__ == "__main__":

    # Load data
    x_train = pd.read_csv("../data/x_train.csv")
    x_val = pd.read_csv("../data/x_val.csv")
    x_test = pd.read_csv("../data/test.csv")

    datasets = {"x_train": x_train, "x_val": x_val, "x_test": x_test}

    for data_name, dataset in datasets.items():

        # Create the preprocessing pipeline
        pipeline = create_full_preprocessing_pipeline()

        # Fit and transform the dataset
        processed_data = pipeline.fit_transform(dataset)

        # get the feature names that are los during transformations
        one_hot_encoder = pipeline.named_steps[
            "preprocess-columns"
        ].named_transformers_["categorical-preprocessor"]

        categorical_feature_names = one_hot_encoder.get_feature_names_out(
            input_features=cat_cols_to_transform
        )

        transformed_feature_names = numeric_cols_to_transform + list(
            categorical_feature_names
        )  # + ['cluster_label']

        processed_df = pd.DataFrame(processed_data, columns=transformed_feature_names)

        # Remove CabinDeck_T column
        processed_df = processed_df.drop("CabinDeck_T", axis=1)

        # Save or use the processed data
        processed_df.to_csv(f"../data/processed_{data_name}_data.csv", index=False)

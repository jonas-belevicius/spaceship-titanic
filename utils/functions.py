import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import pickle
import pandas as pd
from scipy.stats import chi2_contingency
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from pandas import DataFrame, Series
from sklearn.pipeline import Pipeline
import shap
from typing import Union, List, Dict, Type, Any

from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


def _initiate_plot(nrows=1, ncols=1, figsize=(4, 3), sharey=False, sharex=False):
    """
    Create a new matplotlib figure and axes with optional sharing and styling
    """
    fig, ax = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize, sharey=sharey, sharex=sharex
    )

    if nrows == 1 and ncols == 1:
        ax = [ax]

    for a in ax:
        sns.despine(ax=a)

    return fig, ax


def search_best_params(
    pipeline: Pipeline,
    search_grid: dict,
    X_train: DataFrame,
    y_train: Series,
    X_val: DataFrame,
    y_val: Series,
    name: str,
    scoring: str,
    method: str = "random",
    file_name_addition: str = None,
) -> tuple:
    """
    Apply the pipeline and grid/random search to find and display the best model
    parameters with confusion matrix.

    Parameters:
        pipeline: sklearn.pipeline.Pipeline
            The pipeline containing the preprocessing steps and model.
        search_grid: dict
            The parameter grid to search over.
        X_train, X_val: pandas.DataFrame
            Training and validation feature data.
        y_train, y_val: pandas.Series
            Training and validation target data.
        name: str
            Name of the model for display purposes.
        scoring: str
            The scoring metric to optimize.
        method: str, optional
            The optimization method: 'random' (default) or 'grid'.

    Returns:
        best_search: RandomizedSearchCV or GridSearchCV
            The best search object.
        best_params: dict
            The best parameters found by the search.
    """
    if method == "random":
        search = RandomizedSearchCV(
            pipeline,
            search_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            n_iter=50,
            verbose=1,
            refit=scoring,
        )
    else:
        search = GridSearchCV(
            pipeline,
            search_grid,
            scoring=scoring,
            cv=5,
            n_jobs=-1,
            verbose=1,
            refit=scoring,
        )

    best_search = search.fit(X_train, y_train)
    best_params = search.best_params_

    pickle.dump(best_search, open(f"./models/{name}_{file_name_addition}.sav", "wb"))

    return best_search, best_params


def cramers_V(var1, var2):
    """
    Calculate Cramer's V statistic for categorical-categorical association.

    Parameters:
        var1 (array-like): First categorical variable.
        var2 (array-like): Second categorical variable.

    Returns:
        float: Cramer's V statistic.
    """
    crosstab = np.array(pd.crosstab(var1, var2, rownames=None, colnames=None))
    stat = chi2_contingency(crosstab)[0]
    obs = np.sum(crosstab)
    mini = min(crosstab.shape) - 1
    return np.sqrt(stat / (obs * mini))


def outliers_zscore(df: pd.DataFrame, threshold: int = 3) -> pd.DataFrame:
    """
    Identify outliers in a DataFrame using z-score method.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        threshold (int, optional): Z-score threshold for identifying outliers. Defaults to 3.

    Returns:
        pd.DataFrame: DataFrame indicating outlier status for each numerical column.
    """
    numerical_cols = df.select_dtypes(include=["float64"]).columns
    df_num = df[numerical_cols]

    z_scores = (df_num - df_num.mean(axis=0)) / df_num.std(axis=0)

    df_outliers = pd.DataFrame(index=df.index)

    for i, name in enumerate(numerical_cols):
        df_outliers[name + "_outlier"] = np.where(
            np.abs(z_scores.iloc[:, i]) > threshold, "outlier", "non-outlier"
        )
    return df_outliers


def train_validate_test_split(
    X: pd.DataFrame, y: pd.Series, stratify: pd.Series, test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataframe into training, validation, and test sets.

    Parameters:
    - X (pd.DataFrame): The feature DataFrame.
    - y (pd.Series): The target Series.
    - stratify (pd.Series): The stratification target Series.
    - test_size (float, optional): The proportion of the dataset to include
                                    in the test split. Default is 0.2.

    Returns:
    - X_train (pd.DataFrame): The feature training set.
    - X_val (pd.DataFrame): The feature validation set.
    - X_test (pd.DataFrame): The feature test set.
    - y_train (pd.Series): The target training set.
    - y_val (pd.Series): The target validation set.
    - y_test (pd.Series): The target test set.
    """
    X_train, X_vt, y_train, y_vt = train_test_split(
        X, y, test_size=2 * test_size, random_state=1, stratify=stratify
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_vt, y_vt, test_size=0.5, random_state=1, stratify=y_vt
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_new_feature_names_from_ohe(continuous_cols, boolean_cols, ohe):

    ohe_feature_names = ohe.get_feature_names_out(boolean_cols[1:])
    return list(continuous_cols) + list(ohe_feature_names) + ["cluster_label"]


def transform_full_preprocessor(df, continuous_cols, boolean_cols, clf):
    """
    Transforms the dataframe using the full preprocessor pipeline and returns
    the transformed dataframe with appropriate column names.

    Parameters:
    - df: pd.DataFrame, the input dataframe to transform
    - continuous_cols: list, the names of continuous/numerical columns
    - boolean_cols: list, the names of categorical columns
    - clf: sklearn Pipeline, the pipeline with the full preprocessor and model

    Returns:
    - df_transformed: pd.DataFrame, the transformed dataframe with new feature names
    """
    full_preprocessor = clf.named_steps["full-preprocessor"]
    preprocessor = full_preprocessor.named_steps["preprocessor"]
    ohe = preprocessor.transformers_[1][1]["one-hot-encoder"]

    new_feature_names = get_new_feature_names_from_ohe(
        continuous_cols, boolean_cols, ohe
    )
    df_processed = pd.DataFrame(
        full_preprocessor.transform(df), columns=new_feature_names, index=df.index
    )

    return df_processed


def count_unique(data):
    for column in data.columns:
        unique_count = len(data[column].unique())
        print(f"{column} unique entries: {unique_count}")


def compute_shap(
    model: Pipeline, model_name: str, X_train: pd.DataFrame, sample: int = 50
) -> Union[List, Dict]:
    """
    Compute SHAP values for a given model and training data.

    Args:
        model (Pipeline): The machine learning model wrapped in a sklearn Pipeline.
        model_name (str): The name of the model, expected to follow the format 'ModelName_SomeIdentifier'.
        X_train (pd.DataFrame): The training data used to compute SHAP values.
        sample (int): The number of samples from training data to use in SHAP explanation.

    Returns:
        Union[List, Dict]: The SHAP values computed for the given model and sample of the training data.

    Raises:
        ValueError: If the model_name is not supported.
    """
    # Extract the base name of the model
    name_start = model_name.split("_")[0]

    # List of models that can be directly explained by shap.Explainer
    explainer_models = ["XGBClassifier", "LGBMClassifier", "CatBoostClassifier"]

    # List of models that require a prediction wrapper for shap.Explainer
    wrapped_explainer_models = [
        "AdaBoostClassifier",
        "BaggingClassifier",
        "RandomForestClassifier",
    ]

    X_sample = X_train.sample(sample, random_state=1)

    def predict_wrapper(data: pd.DataFrame) -> pd.Series:
        return model.predict(data)

    if name_start in explainer_models:
        explainer = shap.Explainer(model.named_steps[name_start.lower()])
    elif name_start in wrapped_explainer_models:
        model_step = model.named_steps[name_start.lower()]
        explainer = shap.Explainer(predict_wrapper, X_sample)
    else:
        raise ValueError(f"Model {model_name} is not supported")

    # Compute and return SHAP values
    return explainer.shap_values(X_sample)


def train_and_store_model(
    optuna_best_params: Dict[str, Union[int, float]],
    model_name: str,
    model: Type[
        Union[CatBoostClassifier, LGBMClassifier, XGBClassifier, AdaBoostClassifier]
    ],
    x_train: Union[np.ndarray, pd.DataFrame],
    y_train: Union[np.ndarray, pd.Series],
    x_val: Union[np.ndarray, pd.DataFrame],
    model_cache: Dict[str, Pipeline],
    model_predictions: Dict[str, np.ndarray],
    model_predicted_probabilities: Dict[str, np.ndarray],
    model_params: Dict[str, Dict[str, Any]],
) -> None:
    """
    Train a model with given parameters and store the model, predictions, and probabilities.

    Parameters:
    - optuna_best_params: dict
        Dictionary of the best hyperparameters.
    - model_name: str
        The name of the model.
    - model: Type[Union[CatBoostClassifier, LGBMClassifier, XGBClassifier, AdaBoostClassifier]]
        The model class to be instantiated with the given parameters.
    - x_train: Union[np.ndarray, pd.DataFrame]
        The preprocessed training data.
    - y_train: Union[np.ndarray, pd.Series]
        The target values for training.
    - x_val: Union[np.ndarray, pd.DataFrame]
        The preprocessed validation data.
    - model_cache: Dict[str, Pipeline]
        Dictionary to store the trained model.
    - model_predictions: Dict[str, np.ndarray]
        Dictionary to store the predictions.
    - model_predicted_probabilities: Dict[str, np.ndarray]
        Dictionary to store the predicted probabilities.
    - model_params: Dict[str, Dict[str, Any]]
        Dictionary to store the model parameters.

    Returns:
    - None
    """
    # Create and train the model
    pipeline = Pipeline(
        [
            (
                model.__name__.lower(),
                model(**optuna_best_params, random_state=1, verbose=0),
            )
        ]
    )
    pipeline.fit(x_train, y_train)

    # Store the trained model
    model_cache[model_name] = pipeline

    # Predict on validation data
    predictions = pipeline.predict(x_val)
    model_predictions[model_name] = predictions

    # Predict probabilities on validation data
    predicted_probabilities = pipeline.predict_proba(x_val)[:, 1]
    model_predicted_probabilities[model_name] = predicted_probabilities

    # Store model parameters
    model_params[model_name] = pipeline.get_params


def find_proportion_threshold(
    y_true: np.ndarray, predicted_probabilities: np.ndarray, proportion: float = 0.5
) -> Tuple[float, float]:
    """
    Find the threshold at which the selected positive proportion for classification is reached
    and calculate the accuracy at that threshold.

    Parameters:
    - y_true: np.ndarray of shape (n_samples,)
        True binary labels in range {0, 1} or {-1, 1}.
    - predicted_probabilities: np.ndarray of shape (n_samples,)
        Predicted probabilities for the positive class.
    - proportion: float
        The desired proportion of positive classifications (default is 0.5).

    Returns:
    - threshold: float
        The threshold at which the selected positive proportion is reached.
    - accuracy: float
        The accuracy achieved at the calculated threshold.
    """
    # Ensure the proportion is between 0 and 1
    if not (0 <= proportion <= 1):
        raise ValueError("Proportion must be between 0 and 1")

    # Initialize threshold and predictions
    threshold = 0.1
    y_pred = (predicted_probabilities >= threshold).astype(int)

    # Adjust the threshold until the proportion condition is met
    while sum(y_pred) / len(y_pred) > proportion:
        threshold += 0.00005
        y_pred = (predicted_probabilities >= threshold).astype(int)

    # Calculate accuracy at the determined threshold
    accuracy = accuracy_score(y_true, y_pred)

    return round(threshold, 4), round(accuracy, 4)


def find_best_threshold(y_true, predicted_probabilities):
    """
    Find the best threshold for classification based on the best accuracy.

    Parameters:
    - y_true: array-like of shape (n_samples,)
        True binary labels in range {0, 1} or {-1, 1}.
    - predicted_probabilities: array-like of shape (n_samples,)
        Predicted probabilities for the positive class.

    Returns:
    - best_threshold: float
        The threshold that provides the best accuracy.
    - best_accuracy: float
        The accuracy achieved at the best threshold.
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_threshold = 0.0
    best_accuracy = 0.0

    for threshold in thresholds:
        # Convert probabilities to binary predictions based on the current threshold
        y_pred = (predicted_probabilities >= threshold).astype(int)

        # Calculate the accuracy
        accuracy = accuracy_score(y_true, y_pred)

        # Update the best threshold and best accuracy if current accuracy is higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy

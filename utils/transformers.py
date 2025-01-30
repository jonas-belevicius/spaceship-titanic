import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, DBSCAN
from typing import List, Optional, Any, Union
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LinearRegression


class ColumnRemover(BaseEstimator, TransformerMixin):
    """
    Transformer that removes the columns from a DataFrame.
    """

    def __init__(self, columns_to_remove: List[str]):
        self.columns_to_remove = columns_to_remove

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnRemover":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        ColumnRemover: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, removes columns from the DataFrame.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with the first column removed.
        """
        X = X.copy()
        X.drop(self.columns_to_remove, axis=1, inplace=True)
        return X


class ColumnSplitter(BaseEstimator, TransformerMixin):
    """
    Transformer that splits a column into multiple columns based on a delimiter.
    """

    def __init__(
        self, column_to_split: str, columns_split_into: List[str], symbol: str
    ):
        """
        Initialize the ColumnSplitter.

        Parameters:
        column_to_split (str): The name of the column to split.
        columns_split_into (List[str]): List of new column names to split into.
        symbol (str): The delimiter to split by.
        """
        self.column_to_split = column_to_split
        self.columns_split_into = columns_split_into
        self.symbol = symbol

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnSplitter":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        ColumnSplitter: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, splits the specified column into new columns.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with the specified column split into new columns.
        """
        X = X.copy()
        X[self.columns_split_into] = X[self.column_to_split].str.split(
            self.symbol, expand=True
        )
        return X


class PassengerGroupper(BaseEstimator, TransformerMixin):

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "PassengerGroupper":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        PassengerGroupper: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, applies passengers per group calculation.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with log-transformed and binned columns.
        """
        X = X.copy()
        X["PassengersPerGroup"] = X["PassengerGroup"].map(
            lambda x: X["PassengerGroup"].value_counts()[x]
        )
        return X


class FamilyMemberCounter(BaseEstimator, TransformerMixin):

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "FamilyMemberCounter":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        FamilyMemberCounter: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, applies family members calculation.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with log-transformed and binned columns.
        """
        X = X.copy()
        X["Surname"].fillna("Unknown", inplace=True)
        X["FamilyMembers"] = X["Surname"].map(lambda x: X["Surname"].value_counts()[x])
        X["Surname"] = X["Surname"].replace({"Unknown": np.nan})
        X["FamilyMembers"] = X["FamilyMembers"].where(X["FamilyMembers"] <= 50, np.nan)
        return X


class CombinedSpendingCounter(BaseEstimator, TransformerMixin):

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "CombinedSpendingCounter":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        CombinedSpendingCounter: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, applies total spending amount calculation.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with log-transformed and binned columns.
        """
        X = X.copy()
        X["CombinedSpending"] = X[
            ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
        ].sum(axis=1)
        return X


class ColumnBinner(BaseEstimator, TransformerMixin):
    """
    Transformer that applies log transformation and binning to specified columns.
    """

    def __init__(self, columns_to_bin: List[str], bins: int, apply_log: bool = True):
        """
        Initialize the ColumnBinner.

        Parameters:
        columns_to_bin (List[str]): List of columns to apply transformations to.
        bins (int): Number of bins for discretization.
        apply_log (bool): Whether to apply log transformation before binning.
        """
        self.columns_to_bin = columns_to_bin
        self.bins = bins
        self.apply_log = apply_log

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ColumnBinner":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        ColumnBinner: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, applies log transformation and binning to the specified columns.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with log-transformed and binned columns.
        """
        X = X.copy()
        for col in self.columns_to_bin:
            if self.apply_log:
                log_col = f"{col}Log"
                X[log_col] = np.log10(X[col] + 0.01)
                X[f"{col}Cut"] = pd.cut(
                    X[log_col], self.bins, labels=np.linspace(1, self.bins, self.bins)
                )
            else:
                X[f"{col}Cut"] = pd.cut(
                    X[col], self.bins, labels=np.linspace(1, self.bins, self.bins)
                )
        return X


class LogTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies log transformation to specified columns.
    """

    def __init__(self, columns_to_transform: List[str]):
        """
        Initialize the LogTransformer.

        Parameters:
        columns_to_transform (List[str]): List of columns to apply log transformations to.
        """
        self.columns_to_transform = columns_to_transform

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "LogTransformer":
        """
        Fit method, does nothing.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        LogTransformer: Returns self.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, applies log transformation to the specified columns.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with log-transformed columns.
        """
        X = X.copy()
        for col in self.columns_to_transform:
            log_col = f"{col}Log"
            X[log_col] = np.log10(X[col] + 0.01)
        return X


class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Transformer that removes outliers based on a z-score threshold.
    """

    def __init__(self, threshold: int = 3):
        """
        Initialize the OutlierRemover.

        Parameters:
        threshold (int): The z-score threshold to identify outliers.
        """
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "OutlierRemover":
        """
        Fit method, calculates the mean and standard deviation of the DataFrame.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        OutlierRemover: Returns self.
        """
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, removes outliers by setting them to NaN.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with outliers removed.
        """
        z_scores = (X - self.mean_) / self.std_
        mask = np.abs(z_scores) <= self.threshold
        X_outliers_removed = X.copy()
        X_outliers_removed[~mask] = np.nan
        return X_outliers_removed


class KMeansTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies KMeans clustering and appends the cluster labels to the data.
    """

    def __init__(self, n_clusters: int = 2, random_state: Optional[int] = None):
        """
        Initialize the KMeansTransformer.

        Parameters:
        n_clusters (int): The number of clusters for KMeans.
        random_state (Optional[int]): The random state for KMeans.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "KMeansTransformer":
        """
        Fit method, applies KMeans clustering.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        KMeansTransformer: Returns self.
        """
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """
        Transform method, predicts cluster labels and appends them to the data.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        np.ndarray: Data array with appended cluster labels.
        """
        cluster_labels = self.kmeans.predict(X)
        X_with_clusters = np.hstack([X, cluster_labels.reshape(-1, 1)])
        return X_with_clusters


class DBSCANTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer that applies DBSCAN clustering and appends the cluster labels to the data.
    """

    def __init__(
        self,
        eps=0.5,
        min_samples=5,
        metric="euclidean",
        metric_params=None,
        algorithm="auto",
        leaf_size=30,
        p=None,
        n_jobs=None,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.metric_params = metric_params
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.n_jobs = n_jobs
        self.dbscan = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )

    def fit(
        self, X: pd.DataFrame, y: Optional[pd.Series] = None
    ) -> "DBSCANTransformer":
        """
        Fit method, applies DBSCAN clustering.

        Parameters:
        X (pd.DataFrame): Input DataFrame.
        y (pd.Series, optional): Ignored.

        Returns:
        DBSCANTransformer: Returns self.
        """
        self.dbscan.fit(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform method, predicts cluster labels and appends them to the data.

        Parameters:
        X (pd.DataFrame): Input DataFrame.

        Returns:
        pd.DataFrame: DataFrame with appended cluster labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        labels = self.dbscan.fit_predict(X)
        X_transformed = X.copy()
        X_transformed["cluster"] = labels
        return X_transformed


class DFKNNImputer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.imputer = KNNImputer(**kwargs)

    def fit(self, X, y=None):
        self.imputer.fit(X)
        return self

    def transform(self, X):
        X_imputed = self.imputer.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_imputed, columns=X.columns, index=X.index)
        else:
            return pd.DataFrame(X_imputed)


class ColumnImputer(BaseEstimator, TransformerMixin):
    """
    Imputer for the target column based on the most common value in each group of specified columns.

    Parameters
    ----------
    group_cols : List[str]
        The column names indicating the grouping.
    target_col : str
        The column name to impute.
    strategy : str
        The strategy to use for imputation ('mode', 'median', or 'mean').

    Attributes
    ----------
    fill_values : pd.Series
        The imputed values for each group in 'group_cols'.
    """

    def __init__(
        self, group_cols: List[str], target_col: str, strategy: str = "mode"
    ) -> None:
        self.group_cols = group_cols
        self.target_col = target_col
        self.strategy = strategy
        self.fill_values: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "HomePlanetImputer":
        """
        Fit the imputer on the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer.
        y : None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : HomePlanetImputer
            The fitted imputer.
        """
        if self.strategy == "mode":
            self.fill_values = X.groupby(self.group_cols)[self.target_col].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
            )
        elif self.strategy == "median":
            self.fill_values = X.groupby(self.group_cols)[self.target_col].median()
        elif self.strategy == "mean":
            self.fill_values = X.groupby(self.group_cols)[self.target_col].mean()
        else:
            raise ValueError("Strategy not supported. Use 'mode', 'median' or 'mean'.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data by imputing missing values.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with missing values imputed.
        """
        X = X.copy()

        if len(self.group_cols) == 1:
            # Handle single column grouping
            group_col = self.group_cols[0]
            for group, value in self.fill_values.items():
                mask = (X[group_col] == group) & (X[self.target_col].isnull())
                X.loc[mask, self.target_col] = value
        else:
            # Handle multiple column grouping
            grouped = X.groupby(self.group_cols)
            for name, group in grouped:
                mask = group[self.target_col].isnull()
                condition = True
                if isinstance(name, tuple):
                    for col, val in zip(self.group_cols, name):
                        condition &= X[col] == val
                else:
                    condition = X[self.group_cols[0]] == name
                X.loc[condition & mask, self.target_col] = self.fill_values[name]

        return X


class CryoSleepImputer(BaseEstimator, TransformerMixin):
    def __init__(self, group_col: str, target_col: str) -> None:
        self.group_col = group_col
        self.target_col = target_col
        self.fill_values: Optional[pd.Series] = None

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "CryoSleepImputer":
        """
        Fit the imputer on the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer.
        y : None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : CryoSleepImputer
            The fitted imputer.
        """
        self.fill_values = X.groupby(self.group_col)[self.target_col].agg(
            lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
        )
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the data by imputing missing values.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with missing values imputed.
        """
        X = X.copy()
        for group, value in self.fill_values.items():
            mask = (X[self.group_col] == group) & (X[self.target_col].isnull())
            X.loc[mask, self.target_col] = value

        return X


class NoServiceUsers(BaseEstimator, TransformerMixin):
    """
    Transformer to add a column indicating if all specified columns have values greater than 0.

    Parameters
    ----------
    columns_in : List[str]
        List of column names to check.
    column_out : str
        Name of the new column to be added.
    """

    def __init__(self, columns_in: List[str], column_out: str) -> None:
        self.columns_in = columns_in
        self.column_out = column_out

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "NoServiceUsers":
        """
        Fit the transformer.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the transformer.
        y : Optional[Any]
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : NoServiceUsers
            The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the data by adding a new column indicating if all specified columns have values greater than 0.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with the new column added.
        """
        X = X.copy()
        X[self.column_out] = X[self.columns_in].le(0).all(axis=1)
        return X


class RegressionImputer(BaseEstimator, TransformerMixin):

    def __init__(
        self, column_to_impute: str, dependent_column: str, group_col: str
    ) -> None:
        self.column_to_impute = column_to_impute
        self.dependent_column = dependent_column
        self.group_col = group_col
        self.models = {}

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "RegressionImputer":
        """
        Fit the imputer on the data.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer.
        y : None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : RegressionImputer
            The fitted imputer.
        """
        # Fit a regression model for each group
        for group in X[self.group_col].unique():
            group_data = X[X[self.group_col] == group]
            not_null = group_data[group_data[self.column_to_impute].notnull()]
            if not not_null.empty:
                model = LinearRegression()
                model.fit(
                    not_null[[self.dependent_column]], not_null[self.column_to_impute]
                )
                self.models[group] = model
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the data by imputing missing values using the fitted regression models.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with missing values imputed.
        """
        X = X.copy()
        # Impute missing values using the appropriate model for each group
        for group, model in self.models.items():
            group_data = X[X[self.group_col] == group]
            null = group_data[group_data[self.column_to_impute].isnull()]
            if not null.empty:
                X.loc[null.index, self.column_to_impute] = model.predict(
                    null[[self.dependent_column]]
                )
        return X


class SimpleColumnImputer(BaseEstimator, TransformerMixin):
    """
    Imputer for a single column in a DataFrame using a specified strategy.

    Parameters
    ----------
    column_to_impute : str
        The column name to impute.
    strategy : str, default="most_frequent"
        The strategy to use for imputation.
        Supported strategies: "mean", "median", "most_frequent", and "constant".

    Attributes
    ----------
    column_to_impute : str
        The column name to impute.
    strategy : str
        The strategy to use for imputation.
    imputer : SimpleImputer
        The SimpleImputer instance used for imputation.
    """

    def __init__(self, column_to_impute: str, strategy: str = "most_frequent") -> None:
        self.column_to_impute = column_to_impute
        self.strategy = strategy
        self.imputer = SimpleImputer(strategy=self.strategy)

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "SimpleColumnImputer":
        """
        Fit the imputer on the data. This step is necessary to adhere to the scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the imputer.
        y : None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : SimpleColumnImputer
            The fitted imputer.
        """
        self.imputer.fit(X[[self.column_to_impute]])
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the data by imputing missing values using the fitted imputer.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with missing values imputed.
        """
        X = X.copy()
        X[self.column_to_impute] = self.imputer.transform(X[[self.column_to_impute]])[
            :, 0
        ]
        return X


class ChangeDtype(BaseEstimator, TransformerMixin):
    """
    Transformer that changes the dtype of a specified column in a DataFrame.

    Parameters
    ----------
    column : str
        The name of the column to change the dtype.
    dtype_to_change : str, default='float64'
        The dtype to change the column to.

    Attributes
    ----------
    column : str
        The name of the column to change the dtype.
    dtype_to_change : str
        The dtype to change the column to.
    """

    def __init__(self, column: str, dtype_to_change: str = "float64") -> None:
        self.column = column
        self.dtype_to_change = dtype_to_change

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "ChangeDtype":
        """
        Fit method, does nothing but is required by the scikit-learn API.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to fit the transformer.
        y : None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        self : ChangeDtype
            The fitted transformer.
        """
        return self

    def transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """
        Transform the data by changing the dtype of the specified column.

        Parameters
        ----------
        X : pd.DataFrame
            The input data to transform.
        y : None
            Ignored. This parameter exists only for compatibility with the scikit-learn API.

        Returns
        -------
        X : pd.DataFrame
            The transformed data with the column dtype changed.
        """
        X = X.copy()
        X[self.column] = X[self.column].astype(self.dtype_to_change)
        return X


class DropMissing(BaseEstimator, TransformerMixin):
    """
    Transformer that drops rows with any missing values.

    Methods
    -------
    fit(X, y=None)
        Fits the transformer.
    transform(X)
        Transforms the data by dropping rows with missing values.
    """

    def fit(
        self, X: Union[pd.DataFrame, np.ndarray], y: Optional[pd.Series] = None
    ) -> "DropMissing":
        """
        Fit method. Does nothing as no fitting is required.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input DataFrame or numpy array.
        y : Optional[pd.Series], optional
            Ignored.

        Returns
        -------
        DropMissing
            Returns self.
        """
        return self

    def transform(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform method. Drops rows with any missing values.

        Parameters
        ----------
        X : Union[pd.DataFrame, np.ndarray]
            Input DataFrame or numpy array.

        Returns
        -------
        Union[pd.DataFrame, np.ndarray]
            Data with rows containing missing values removed.
        """
        if isinstance(X, pd.DataFrame):
            return X.dropna()
        elif isinstance(X, np.ndarray):
            return X[~np.isnan(X).any(axis=1)]
        else:
            raise ValueError("Input should be a pandas DataFrame or numpy array")

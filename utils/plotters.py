import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu, kruskal
from sklearn.metrics import precision_recall_curve, auc, roc_curve, roc_auc_score
from typing import List, Optional
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report


def initiate_plot(nrows=1, ncols=1, figsize=(4, 3), sharey=False, sharex=False):
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


def show_distributions(df: pd.DataFrame) -> None:
    """
    Visualize the distributions of numerical columns in the DataFrame

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        None
    """
    num_cols = df.select_dtypes(exclude="object").columns
    num_num_cols = len(num_cols)

    fig, ax = initiate_plot(1, num_num_cols, figsize=(num_num_cols * 3.5, 4))

    for i, col in enumerate(num_cols):
        if (
            df[col].dtype == "int64"
            or df[col].dtype == "int32"
            or df[col].dtype == "int"
        ):
            sns.countplot(data=df, x=col, ax=ax[i])
        else:
            sns.histplot(data=df, x=col, ax=ax[i])
        ax[i].set_title(f'"{col}" parameter')

    plt.tight_layout()
    plt.suptitle("Distribution of Data in Numerical Features", size="large", y=1.05)


def plot_association_hist(x, data, hue, ax, title="Title", shrink=0.8, discrete=True):
    sns.histplot(
        data=data,
        x=x,
        hue=hue,
        multiple="fill",
        shrink=shrink,
        discrete=discrete,
        ax=ax,
    )

    ax.set_title(title, size="medium", y=1.05)


def boxplot_cat(
    df: pd.DataFrame,
    categoric_cols: list,
    numerical_col: str,
    text_addition: str,
    title: str,
    delaxes: list = None,
    nrows: int = 2,
    ncols: int = 3,
    figsize: tuple = (9, 6),
) -> None:
    """
    Plot categorical features with numerical column on y axis and
    apply test to identify if there is difference between categories in
    distribution of numerical column.

    df (pd.DataFrame): input dataset
    numerical_col (str): numerical column, based on which categories are compared.
    text_addition (str): text part that is added to individual subplots to explain
    categorical features displayed.
    title (str): title of all subplots.
    delaxes (list, optional): list of axes indices to delete
    nrows (int): number of subplot rows.
    ncols (int): number of subplot columns.
    figsize (tuple): size of all subplots in a figure.
    """

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    axes = axes.ravel()

    for col, ax in zip(categoric_cols, axes):
        sns.boxplot(data=df, x=col, y=numerical_col, ax=ax)

        ax.set_title(f"'{col}' {text_addition}", size="small")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymax=ymax * 1.2)

        categories = df[col].unique()
        if len(categories) == 2:
            group1 = df[df[col] == categories[0]][numerical_col]
            group2 = df[df[col] == categories[1]][numerical_col]
            _, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            ax.text(
                0.8,
                0.85,
                f"Mann-Whitney \np-value: {p_value:.3f}",
                transform=ax.transAxes,
                ha="center",
                size="x-small",
            )
        else:
            groups = [df[df[col] == category][numerical_col] for category in categories]
            _, p_value = kruskal(*groups)
            ax.text(
                0.8,
                0.85,
                f"Kruskal-Wallis \np-value: {p_value:.3f}",
                transform=ax.transAxes,
                ha="center",
                size="x-small",
            )
    if delaxes is not None:
        for idx in delaxes:
            fig.delaxes(axes[idx])

    plt.tight_layout()
    plt.suptitle(title, size="large", y=1.05)


def plot_scatter(
    df: pd.DataFrame,
    categoric_cols: list,
    numerical_colx: str,
    numerical_coly: str,
    title: str,
) -> None:
    """
    Plot numerical features in the scatterplot with hue as categorical column.

    df (pd.DataFrame): input dataset

    numerical_colx (str): numerical column, on x axis
    numerical_coly (str): numerical column, on y axis

    text_addition (str): text part that is added to individual subplots to
    explain features displayed.

    title (str): title of all subplots
    """
    col_len = len(categoric_cols)

    fig, ax = initiate_plot(1, col_len, figsize=(3.5 * col_len, 4))

    for i, col in enumerate(categoric_cols):
        sns.scatterplot(
            data=df,
            x=numerical_colx,
            y=numerical_coly,
            hue=col,
            ax=ax[i],
            palette="crest",
        )

        ax[i].set_title(
            f"'{col}' by \n'{numerical_colx}' and '{numerical_coly}'", size="medium"
        )

        ax[i].set_xticklabels(ax[i].get_xticklabels(), rotation=45, ha="right")

        ax[i].legend(
            loc="upper right",
            fontsize="small",
            frameon=False,
            bbox_to_anchor=(0.7, -0.2),
            title_fontsize="small",
        )

    plt.suptitle(title, size="large", y=1.05)


def review_col_counts(df: pd.DataFrame, selected_cols: List) -> None:
    """
    Review and visualize counts of unique entries in columns.

    Parameters:
       df (pd.DataFrame): Input DataFrame.

    Returns:
       None
    """

    fig, ax = initiate_plot(1, selected_cols, figsize=(selected_cols * 3.5, 3))

    for i, col in enumerate(selected_cols):
        counts = df.groupby(selected_cols)[col].value_counts().reset_index()

        sns.barplot(x=counts.iloc[:, 1], y=counts.iloc[:, 0], ax=ax[i])
        ax[i].set_title(f'"{col}" parameter', size="medium")

    plt.tight_layout()
    plt.suptitle("Unique Features of Selected Parameters", size="medium", y=1.05)


def plot_roc_curve(
    y_val: int, predicted_probabilities: float, model_name: str, ax: plt.Axes
) -> None:
    """
    Plots the ROC curve for a binary classification model.

    Parameters:
    y_val (array-like): True binary labels.
    predicted_probabilities (array-like): Predicted probabilities of the
    positive class.
    model_name (str): Name of the model for labeling the plot.
    ax (matplotlib.axes.Axes): Axes object to plot on.

    Returns:
    None
    """
    fpr, tpr, _ = roc_curve(y_val, predicted_probabilities)
    roc_auc = roc_auc_score(y_val, predicted_probabilities)

    sns.lineplot(x=fpr, y=tpr, label=f"{model_name} AUC = {roc_auc:.2f}", ax=ax)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve", size="medium")


def plot_precision_recall_curve(
    y_val: int, predicted_probabilities: float, model_name: str, ax: plt.Axes
) -> None:
    """
    Plots the precision-recall curve for a binary classification model.

    Parameters:
    y_val (array-like): True binary labels.
    predicted_probabilities (array-like): Predicted probabilities of the positive class.
    model_name (str): Name of the model for labeling the plot.
    ax (matplotlib.axes.Axes): Axes object to plot on.

    Returns:
    None
    """
    precision, recall, _ = precision_recall_curve(y_val, predicted_probabilities)
    pr_auc = auc(recall, precision)

    sns.lineplot(x=recall, y=precision, label=f"{model_name} AUC = {pr_auc:.2f}", ax=ax)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve", size="medium")


def evaluate_model(
    model,
    x_test,
    y_test,
    labels: List[str] = ["0", "1"],
    addition_to_title: Optional[str] = None,
) -> None:
    """
    Evaluates model by showing confusion matrix, precision, recall and f1
    scores, false positive rate, false negative rate, draws ROC and
    precision-recall curves.

    Parameters:
        model: object
            The trained model to evaluate.
        x_test: array-like
            Test features.
        y_test: array-like
            True labels.
        labels: list of str, optional
            Labels for the confusion matrix. Default is ['0', '1'].
        addition_to_title: str, optional
            Additional text to include in the title of the plots. Default is None.
    """
    y_pred = model.predict(x_test)
    predicted_probabilities = model.predict_proba(x_test)[:, 1]

    fig, ax = initiate_plot(1, 5, figsize=(18, 3))

    # Confusion Matrix
    sns.heatmap(
        confusion_matrix(y_test, y_pred), cmap="crest", annot=True, fmt=".0f", ax=ax[0]
    )

    ax[0].set_xticklabels(labels)
    ax[0].set_yticklabels(labels)
    ax[0].set_xlabel("Actual")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title(f"Confusion Matrix \n{addition_to_title}", size="medium")

    # Classification Report
    report = classification_report(y_test, y_pred, output_dict=True)["1"]
    report = list(report.items())[:3]
    scores = [score for metric, score in report]  # Extract scores
    metrics = [metric for metric, score in report]

    sns.barplot(x=metrics, y=scores, ax=ax[1])
    ax[1].set_ylim(0, 1)
    ax[1].set_title("Performance Metrics", size="medium")

    for i, score in enumerate(scores):
        ax[1].text(i, score, f"{score:.2f}", ha="center", va="bottom")

    # False positives and false negatives
    confusion = confusion_matrix(y_test, y_pred)
    fpr = confusion[0, 1] / np.sum(confusion[0])
    fnr = confusion[1, 0] / np.sum(confusion[1])
    fnr_fpr_dict = {"False positives": fpr, "False negatives": fnr}

    for i, (name, score) in enumerate(fnr_fpr_dict.items()):
        sns.barplot(x=[name], y=[score], ax=ax[2])
        ax[2].set_title("False negatives/positives")
        ax[2].text(i, score, f"{score:.2f}", ha="center", va="bottom")

    # ROC Curve
    plot_roc_curve(y_test, predicted_probabilities, model_name="", ax=ax[3])

    # Precision-Recall Curve
    plot_precision_recall_curve(
        y_test, predicted_probabilities, model_name="", ax=ax[4]
    )

    plt.tight_layout()
    plt.suptitle(f"{addition_to_title} Performace Results", size="large", y=1.05)

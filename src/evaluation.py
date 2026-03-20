from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.pipeline import Pipeline


def evaluate_classification(
    model, X_test, y_test
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Compute standard classification metrics and the confusion matrix.

    Returns:
    - metrics: dict with accuracy, precision, recall, f1 (weighted).
    - cm: confusion matrix array.
    - classes: sorted unique class labels (for plotting).
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    classes = np.unique(y_test)
    cm = confusion_matrix(y_test, y_pred, labels=classes)

    metrics = {
        "Accuracy": accuracy,
        "Precision (weighted)": precision,
        "Recall (weighted)": recall,
        "F1-score (weighted)": f1,
    }
    return metrics, cm, classes


def evaluate_regression(model, X_test, y_test) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Returns a dict with RMSE, MAE, and R^2.
    """
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "RMSE": rmse,
        "MAE": mae,
        "R^2": r2,
    }


def plot_confusion_matrix_figure(
    cm: np.ndarray, classes: np.ndarray, cmap: str = "Blues"
):
    """
    Create a matplotlib Figure for the confusion matrix using seaborn heatmap.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    return fig


def get_feature_importance(model) -> Optional[pd.DataFrame]:
    """
    Try to extract feature importance or coefficients from a fitted model pipeline.

    Supports:
    - Tree-based models with `feature_importances_`.
    - Linear models with `coef_` (absolute value used as importance).

    Returns a DataFrame with columns: Feature, Importance; or None if not available.
    """
    # Unwrap sklearn Pipeline if needed
    preprocessor = None
    final_estimator = model
    if isinstance(model, Pipeline):
        preprocessor = model.named_steps.get("preprocessor")
        final_estimator = model.named_steps.get("model", model)

    # Determine feature names from preprocessor if possible
    feature_names = None
    if preprocessor is not None:
        try:
            feature_names = preprocessor.get_feature_names_out()
        except Exception:
            feature_names = None

    if hasattr(final_estimator, "feature_importances_"):
        importances = np.asarray(final_estimator.feature_importances_)
    elif hasattr(final_estimator, "coef_"):
        coef = np.asarray(final_estimator.coef_)
        # For multi-class linear models, coef_ has shape (n_classes, n_features).
        # Aggregate across classes so we end up with one importance per feature.
        if coef.ndim > 1:
            importances = np.mean(np.abs(coef), axis=0)
        else:
            importances = np.abs(coef)
    else:
        return None

    importances = importances.ravel()

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]
    else:
        # Ensure feature_names length matches importances; if not, fall back to generic names.
        if len(feature_names) != len(importances):
            feature_names = [f"Feature {i}" for i in range(len(importances))]

    importance_df = pd.DataFrame(
        {"Feature": feature_names, "Importance": importances}
    ).sort_values("Importance", ascending=False)

    return importance_df


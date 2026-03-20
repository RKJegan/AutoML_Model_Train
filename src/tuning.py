from __future__ import annotations

from typing import Dict, Any


def get_classification_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Hyperparameter search spaces for classification models.

    Keys correspond to model names used in `model_training.get_classification_models`.
    """
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__max_iter": [500, 1000],
        },
        "Decision Tree": {
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "Random Forest": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "SVM": {
            "model__C": [0.1, 1.0, 10.0],
            "model__kernel": ["rbf", "linear"],
            "model__gamma": ["scale", "auto"],
        },
    }


def get_regression_param_grids() -> Dict[str, Dict[str, Any]]:
    """
    Hyperparameter search spaces for regression models.

    Keys correspond to model names used in `model_training.get_regression_models`.
    """
    return {
        "Linear Regression": {
            "model__fit_intercept": [True, False],
        },
        "Ridge Regression": {
            "model__alpha": [0.1, 1.0, 10.0],
            "model__fit_intercept": [True, False],
        },
        "Lasso Regression": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0],
            "model__max_iter": [1000, 5000],
        },
        "Random Forest Regressor": {
            "model__n_estimators": [50, 100, 200],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    }


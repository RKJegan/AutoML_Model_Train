from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .preprocessing import build_full_pipeline
from .tuning import get_classification_param_grids, get_regression_param_grids


@dataclass
class ModelResult:
    name: str
    best_estimator: BaseEstimator
    best_score: float
    best_params: Dict[str, Any]


def get_classification_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Return a dictionary of classification model instances."""
    return {
        "Logistic Regression": LogisticRegression(random_state=random_state),
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(random_state=random_state),
        "SVM": SVC(probability=True, random_state=random_state),
    }


def get_regression_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Return a dictionary of regression model instances."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(random_state=random_state),
        "Lasso Regression": Lasso(random_state=random_state),
        "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
    }


def _choose_models(
    problem_type: str, selected_model_name: Optional[str] = None, random_state: int = 42
) -> Tuple[Dict[str, BaseEstimator], Dict[str, Dict[str, Any]]]:
    """Return the model dict and corresponding param grids for the chosen problem type."""
    if problem_type == "classification":
        all_models = get_classification_models(random_state=random_state)
        grids = get_classification_param_grids()
    else:
        all_models = get_regression_models(random_state=random_state)
        grids = get_regression_param_grids()

    if selected_model_name is not None:
        # Filter down to the manually selected model
        chosen_models = {selected_model_name: all_models[selected_model_name]}
        chosen_grids = {selected_model_name: grids.get(selected_model_name, {})}
        return chosen_models, chosen_grids

    return all_models, grids


def train_and_tune_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    problem_type: str,
    selected_model_name: Optional[str] = None,
    manual_params: Optional[Dict[str, Any]] = None,
    n_iter: int = 10,
    cv_folds: int = 3,
    random_state: int = 42,
) -> Tuple[ModelResult, List[ModelResult], pd.DataFrame]:
    """
    Train and tune models using RandomizedSearchCV.

    Returns:
    - best_model_result: the best-performing model and its information.
    - all_model_results: list of results for each tried model.
    - comparison_df: table summarizing performance of all models.
    """
    models, param_grids = _choose_models(
        problem_type, selected_model_name=selected_model_name, random_state=random_state
    )

    # Keep the scores meaningful and user-friendly:
    # - classification: accuracy (higher is better)
    # - regression: RMSE (lower is better). We compute RMSE as a positive value.
    scoring = "accuracy" if problem_type == "classification" else "neg_root_mean_squared_error"

    all_results: List[ModelResult] = []

    for name, base_model in models.items():
        # If in manual mode for this single model, apply user-provided hyperparameters
        # directly on the underlying estimator and skip RandomizedSearchCV.
        use_manual = manual_params is not None and selected_model_name == name
        if use_manual:
            base_model = base_model.set_params(**manual_params)

        pipeline = build_full_pipeline(X_train, base_model)
        param_distributions = {} if use_manual else param_grids.get(name, {})

        if param_distributions:
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                n_jobs=-1,
                cv=cv_folds,
                random_state=random_state,
                refit=True,
            )
            search.fit(X_train, y_train)
            best_estimator = search.best_estimator_
            if problem_type == "classification":
                best_score = float(search.best_score_)  # accuracy in [0, 1]
            else:
                best_score = float(-search.best_score_)  # convert neg RMSE -> RMSE
            best_params = dict(search.best_params_)
        else:
            # No hyperparameters to tune; just fit a plain pipeline.
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            # Compute a CV score so comparison is consistent and doesn't show negatives.
            cv_scores = cross_val_score(
                best_estimator,
                X_train,
                y_train,
                cv=cv_folds,
                scoring=scoring,
                n_jobs=-1,
            )
            if problem_type == "classification":
                best_score = float(np.mean(cv_scores))  # accuracy
            else:
                best_score = float(-np.mean(cv_scores))  # RMSE (positive)
            best_params = manual_params if use_manual else {}

        all_results.append(
            ModelResult(
                name=name,
                best_estimator=best_estimator,
                best_score=best_score,
                best_params=best_params,
            )
        )

    # Determine the best model.
    if problem_type == "classification":
        best_model = max(all_results, key=lambda r: r.best_score)  # maximize accuracy
    else:
        best_model = min(all_results, key=lambda r: r.best_score)  # minimize RMSE

    # Build comparison table with user-friendly scores.
    comparison_rows = []
    if problem_type == "classification":
        metric_label = "CV Accuracy"
        for r in all_results:
            display_score = r.best_score
            comparison_rows.append(
                {
                    "Model": r.name,
                    metric_label: display_score,
                    "Best Params": r.best_params,
                }
            )
    else:
        metric_label = "CV RMSE"
        for r in all_results:
            comparison_rows.append(
                {
                    "Model": r.name,
                    metric_label: r.best_score,
                    "Best Params": r.best_params,
                }
            )

    comparison_df = pd.DataFrame(comparison_rows)

    return best_model, all_results, comparison_df


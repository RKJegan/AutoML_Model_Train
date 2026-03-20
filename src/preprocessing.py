from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetInfo:
    """Container with basic dataset information for display in the UI."""

    shape: Tuple[int, int]
    dtypes: pd.Series
    head: pd.DataFrame


def load_dataset(file) -> pd.DataFrame:
    """
    Read a CSV file (uploaded via Streamlit) into a pandas DataFrame.

    `file` can be a path-like object or a Streamlit UploadedFile.
    """
    return pd.read_csv(file)


def get_dataset_info(df: pd.DataFrame, head_rows: int = 5) -> DatasetInfo:
    """Return a lightweight summary of the dataset for quick inspection."""
    return DatasetInfo(
        shape=df.shape,
        dtypes=df.dtypes,
        head=df.head(head_rows),
    )


def detect_problem_type(target: pd.Series) -> str:
    """
    Infer whether the problem is classification or regression.

    Heuristic:
    - If the target dtype is object/category/bool, treat as classification.
    - Otherwise, if the number of unique values is relatively small
      compared to the dataset size, treat as classification.
    - Else, treat as regression.
    """
    if target.dtype == "O" or pd.api.types.is_categorical_dtype(target) or pd.api.types.is_bool_dtype(
        target
    ):
        return "classification"

    # Numeric target: decide based on number of unique values
    n_unique = target.nunique(dropna=True)
    n_total = len(target)
    unique_ratio = n_unique / max(n_total, 1)

    if n_unique <= 20 or unique_ratio < 0.05:
        return "classification"
    return "regression"


def build_preprocessor(
    X: pd.DataFrame,
) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a ColumnTransformer that:
    - Imputes numeric features with median and scales them.
    - Imputes categorical features with most-frequent and one-hot encodes them.

    Returns the preprocessor and the lists of numeric and categorical feature names.
    """
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features


def train_test_split_data(
    df: pd.DataFrame,
    target_column: str,
    problem_type: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the dataset into train and test sets.

    Uses stratification for classification problems when possible.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    stratify = None
    if problem_type == "classification":
        # Only stratify if there is more than one class.
        if y.nunique(dropna=True) > 1:
            stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test


def build_full_pipeline(
    X: pd.DataFrame,
    model,
) -> Pipeline:
    """
    Attach the preprocessing ColumnTransformer in front of a model
    to create a full sklearn Pipeline.
    """
    preprocessor, _, _ = build_preprocessor(X)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )
    return pipeline


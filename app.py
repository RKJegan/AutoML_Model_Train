from __future__ import annotations

import textwrap
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score

from src.evaluation import (
    evaluate_regression,
)
from src.model_training import (
    ModelResult,
    get_classification_models,
    get_regression_models,
    train_and_tune_models,
)
from src.preprocessing import (
    DatasetInfo,
    detect_problem_type,
    get_dataset_info,
    load_dataset,
    train_test_split_data,
)


st.set_page_config(
    page_title="AutoML Streamlit App",
    layout="wide",
)

# Persist trained model and feature info across reruns (e.g. when clicking Predict)
if "trained_model" not in st.session_state:
    st.session_state["trained_model"] = None
if "feature_columns" not in st.session_state:
    st.session_state["feature_columns"] = None
if "problem_type" not in st.session_state:
    st.session_state["problem_type"] = None
if "target_column" not in st.session_state:
    st.session_state["target_column"] = None


@st.cache_data
def cached_load_dataset(uploaded_file) -> pd.DataFrame:
    """Wrapper around load_dataset with Streamlit caching."""
    return load_dataset(uploaded_file)


def render_sidebar() -> None:
    st.sidebar.title("AutoML Settings")
    st.sidebar.markdown(
        textwrap.dedent(
            """
            Upload a CSV dataset, choose the target column,
            and optionally pick a specific algorithm.

            - Automatic mode: tries multiple models and picks the best.
            - Manual mode: you choose the algorithm to train.
            """
        )
    )


def select_mode() -> str:
    return st.sidebar.radio(
        "Select mode",
        options=["Automatic Model Selection", "Manual Model Selection"],
        index=0,
    )


def get_manual_model_choice(problem_type: str) -> Optional[str]:
    """Let the user pick a specific model name in manual mode."""
    if problem_type == "classification":
        model_names = list(get_classification_models().keys())
    else:
        model_names = list(get_regression_models().keys())

    return st.sidebar.selectbox("Choose algorithm", options=model_names)


def get_manual_hyperparameters(problem_type: str, model_name: Optional[str]) -> Optional[Dict]:
    """
    Render hyperparameter controls in the sidebar for the selected model (manual mode)
    and return the chosen values as a dict that can be passed to the estimator.
    """
    if model_name is None:
        return None

    with st.sidebar.expander("Model Hyperparameters", expanded=True):
        if problem_type == "classification":
            if model_name == "Logistic Regression":
                C = st.number_input("C (inverse regularization strength)", 0.0001, 1000.0, 1.0, step=0.1)
                max_iter = st.number_input("max_iter", 100, 5000, 1000, step=100)
                penalty = st.selectbox("penalty", ["l2"])
                solver = st.selectbox("solver", ["lbfgs", "liblinear"])
                return {"C": C, "max_iter": int(max_iter), "penalty": penalty, "solver": solver}

            if model_name == "Decision Tree":
                criterion = st.selectbox("criterion", ["gini", "entropy", "log_loss"])
                max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20])
                max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
                min_samples_split = st.number_input("min_samples_split", 2, 50, 2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1, step=1)
                return {
                    "criterion": criterion,
                    "max_depth": max_depth,
                    "min_samples_split": int(min_samples_split),
                    "min_samples_leaf": int(min_samples_leaf),
                }

            if model_name == "Random Forest":
                n_estimators = st.number_input("n_estimators", 10, 1000, 100, step=10)
                max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20])
                max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
                min_samples_split = st.number_input("min_samples_split", 2, 50, 2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1, step=1)
                return {
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth,
                    "min_samples_split": int(min_samples_split),
                    "min_samples_leaf": int(min_samples_leaf),
                }

            if model_name == "SVM":
                C = st.number_input("C", 0.0001, 1000.0, 1.0, step=0.1)
                kernel = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"])
                gamma = st.selectbox("gamma", ["scale", "auto"])
                return {"C": C, "kernel": kernel, "gamma": gamma}

        else:
            # Regression models
            if model_name == "Linear Regression":
                fit_intercept = st.checkbox("fit_intercept", value=True)
                return {"fit_intercept": fit_intercept}

            if model_name == "Ridge Regression":
                alpha = st.number_input("alpha", 0.0001, 1000.0, 1.0, step=0.1)
                fit_intercept = st.checkbox("fit_intercept", value=True)
                return {"alpha": alpha, "fit_intercept": fit_intercept}

            if model_name == "Lasso Regression":
                alpha = st.number_input("alpha", 0.0001, 1000.0, 1.0, step=0.1)
                max_iter = st.number_input("max_iter", 100, 5000, 1000, step=100)
                return {"alpha": alpha, "max_iter": int(max_iter)}

            if model_name == "Random Forest Regressor":
                n_estimators = st.number_input("n_estimators", 10, 1000, 100, step=10)
                max_depth_opt = st.selectbox("max_depth", ["None", 3, 5, 10, 20])
                max_depth = None if max_depth_opt == "None" else int(max_depth_opt)
                min_samples_split = st.number_input("min_samples_split", 2, 50, 2, step=1)
                min_samples_leaf = st.number_input("min_samples_leaf", 1, 50, 1, step=1)
                return {
                    "n_estimators": int(n_estimators),
                    "max_depth": max_depth,
                    "min_samples_split": int(min_samples_split),
                    "min_samples_leaf": int(min_samples_leaf),
                }

    return None


def display_dataset_info(info: DatasetInfo) -> None:
    st.subheader("Dataset Preview")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Shape** (rows, columns):", info.shape)
        st.write("**Column types:**")
        st.dataframe(info.dtypes.to_frame("dtype"))

    with col2:
        st.write("**Head:**")
        st.dataframe(info.head)


def display_model_results(
    best_result: ModelResult,
    comparison_df: pd.DataFrame,
    problem_type: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    X_train: pd.DataFrame,
    target_column: str,
) -> None:
    st.subheader("Best Model Summary")
    st.write(f"**Best algorithm:** {best_result.name}")

    if problem_type == "classification":
        y_pred = best_result.best_estimator.predict(X_test)
        acc = float(accuracy_score(y_test, y_pred))
        st.write("**Accuracy (test set):**")
        st.metric("Accuracy", f"{max(0.0, acc) * 100:.2f}%")
    else:
        metrics = evaluate_regression(best_result.best_estimator, X_test, y_test)
        st.write("**Evaluation metrics (test set):**")
        st.json({k: float(v) for k, v in metrics.items()})

    st.subheader("Best Hyperparameters")
    if best_result.best_params:
        st.json(best_result.best_params)
    else:
        st.write("No hyperparameters were tuned for this model.")

    st.subheader("Model Comparison")
    # For classification, show accuracy as a positive percentage in the table.
    if problem_type == "classification" and "CV Accuracy" in comparison_df.columns:
        df_display = comparison_df.copy()
        df_display["CV Accuracy"] = (df_display["CV Accuracy"].astype(float).clip(0, 1) * 100).map(lambda v: f"{v:.2f}%")
        st.dataframe(df_display)
    else:
        st.dataframe(comparison_df)


def render_prediction_section(df: pd.DataFrame) -> None:
    """
    Show prediction form only when a trained model exists in session_state.
    Uses feature_columns for correct column order; validates against current df when building inputs.
    """
    st.subheader("Prediction")
    if st.session_state["trained_model"] is None:
        st.warning("Please train the model first.")
        return

    model_result: ModelResult = st.session_state["trained_model"]
    feature_columns = st.session_state["feature_columns"]
    problem_type = st.session_state["problem_type"]
    target_column = st.session_state["target_column"]

    if not feature_columns:
        st.warning("No feature columns saved. Please run AutoML again.")
        return

    # Ensure target is excluded and we have a matching feature set from current df
    X_ref = df.drop(columns=[target_column], errors="ignore") if target_column in df.columns else df
    missing = [c for c in feature_columns if c not in X_ref.columns]
    if missing:
        st.warning(
            "The current dataset does not contain the same features used for training "
            f"(missing: {missing}). Please upload the same dataset and run AutoML again."
        )
        return

    with st.form("prediction_form"):
        user_input: Dict = {}
        cols = st.columns(2)
        for i, col_name in enumerate(feature_columns):
            container = cols[i % 2]
            series = X_ref[col_name]
            with container:
                if pd.api.types.is_numeric_dtype(series):
                    default_val = float(series.median()) if series.dropna().shape[0] else 0.0
                    user_input[col_name] = st.number_input(f"{col_name}", value=default_val)
                elif pd.api.types.is_bool_dtype(series):
                    user_input[col_name] = st.selectbox(f"{col_name}", options=[False, True], index=0)
                else:
                    uniq = series.dropna().astype(str).unique().tolist()
                    uniq = uniq[:50]
                    if len(uniq) > 0 and series.nunique(dropna=True) <= 50:
                        user_input[col_name] = st.selectbox(f"{col_name}", options=uniq)
                    else:
                        user_input[col_name] = st.text_input(f"{col_name}", value="")

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            # Build DataFrame with columns in the same order as training
            input_row = [user_input[c] for c in feature_columns]
            input_df = pd.DataFrame([input_row], columns=feature_columns)
            # Coerce dtypes to match training (e.g. numeric columns as float)
            for c in feature_columns:
                if pd.api.types.is_numeric_dtype(X_ref[c]):
                    input_df[c] = pd.to_numeric(input_df[c], errors="coerce")
            pred = model_result.best_estimator.predict(input_df)[0]
            if problem_type == "classification":
                st.success(f"**Predicted Class:** {pred}")
            else:
                st.success(f"**Predicted Value:** {pred}")
        except (ValueError, TypeError) as e:
            st.error("Invalid input or feature type mismatch. Please check your values and try again.")
        except Exception as e:
            st.error(f"Prediction failed. Please ensure inputs match the training features and try again.")


def main() -> None:
    st.title("End-to-End AutoML Web Application")
    st.markdown(
        """
        Upload your tabular CSV dataset, select the target column,
        and let this app automatically preprocess the data, train multiple models
        with hyperparameter tuning, and show you the best-performing model.
        """
    )

    render_sidebar()
    mode = select_mode()

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file is None:
        st.info("Please upload a CSV file to get started.")
        return

    # Load and preview dataset
    try:
        df = cached_load_dataset(uploaded_file)
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return

    if df.empty:
        st.error("The uploaded CSV appears to be empty.")
        return

    dataset_info = get_dataset_info(df)
    display_dataset_info(dataset_info)

    # Target column selection
    target_column = st.selectbox("Select target column (what you want to predict)", df.columns)
    if not target_column:
        st.info("Select a target column to continue.")
        return

    target_series = df[target_column]
    problem_type = detect_problem_type(target_series)

    if problem_type == "classification":
        st.success("Detected problem type: **Classification**")
    else:
        st.success("Detected problem type: **Regression**")

    selected_model_name: Optional[str] = None
    manual_params: Optional[Dict] = None
    if mode == "Manual Model Selection":
        selected_model_name = get_manual_model_choice(problem_type)
        manual_params = get_manual_hyperparameters(problem_type, selected_model_name)

    if st.button("Run AutoML"):
        with st.spinner("Training models. This may take a moment..."):
            try:
                # Target column is excluded from features: X = df.drop(target), y = df[target]
                X_train, X_test, y_train, y_test = train_test_split_data(
                    df, target_column, problem_type
                )

                best_result, all_results, comparison_df = train_and_tune_models(
                    X_train=X_train,
                    y_train=y_train,
                    problem_type=problem_type,
                    selected_model_name=selected_model_name,
                    manual_params=manual_params,
                )
            except Exception as e:
                st.error(f"An error occurred during training: {e}")
                return

        # Persist trained model and feature info so Predict button works after reruns
        st.session_state["trained_model"] = best_result
        st.session_state["feature_columns"] = X_train.columns.tolist()
        st.session_state["problem_type"] = problem_type
        st.session_state["target_column"] = target_column

        display_model_results(
            best_result=best_result,
            comparison_df=comparison_df,
            problem_type=problem_type,
            X_test=X_test,
            y_test=y_test,
            X_train=X_train,
            target_column=target_column,
        )

    # Prediction section: show only when a trained model exists; otherwise "Please train first"
    render_prediction_section(df)


if __name__ == "__main__":
    main()


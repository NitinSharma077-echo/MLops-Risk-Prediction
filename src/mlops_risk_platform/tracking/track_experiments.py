import json
from inspect import signature
from pathlib import Path
from typing import Dict, Tuple

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from mlops_risk_platform.config import load_config, get_project_root
from mlops_risk_platform.utils.logger import get_logger


logger = get_logger(__name__)


def load_train_test_data(
    train_path: Path,
    test_path: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads train and test datasets.

    Args:
        train_path: Path to train CSV.
        test_path: Path to test CSV.

    Returns:
        Train and test DataFrames.
    """

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train data not found at {train_path}. Run ingest_data.py first."
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. Run ingest_data.py first."
        )

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    logger.info("Train data loaded from: %s", train_path)
    logger.info("Test data loaded from: %s", test_path)

    return train_data, test_data


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Splits data into X and y.

    Args:
        data: Input DataFrame.
        target_column: Target column.
        id_column: ID column.

    Returns:
        X and y.
    """

    X = data.drop(columns=[target_column, id_column])
    y = data[target_column]

    return X, y


def create_one_hot_encoder() -> OneHotEncoder:
    """
    Creates OneHotEncoder compatible with old and new Scikit-learn versions.

    Returns:
        OneHotEncoder.
    """

    encoder_params = signature(OneHotEncoder).parameters

    if "sparse_output" in encoder_params:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(
    numeric_columns: list,
    categorical_columns: list,
) -> ColumnTransformer:
    """
    Builds preprocessing pipeline.

    Args:
        numeric_columns: Numeric feature columns.
        categorical_columns: Categorical feature columns.

    Returns:
        ColumnTransformer.
    """

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", create_one_hot_encoder()),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )

    return preprocessor


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculates XGBoost scale_pos_weight for imbalance.

    Args:
        y_train: Training target.

    Returns:
        Weight value.
    """

    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        return 1.0

    return round(negative_count / positive_count, 4)


def build_logistic_regression(
    preprocessor: ColumnTransformer,
    random_seed: int,
) -> Pipeline:
    """
    Builds Logistic Regression pipeline.

    Args:
        preprocessor: Preprocessing object.
        random_seed: Random seed.

    Returns:
        Logistic Regression pipeline.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_seed,
                ),
            ),
        ]
    )

    return model


def build_random_forest(
    preprocessor: ColumnTransformer,
    random_seed: int,
) -> Pipeline:
    """
    Builds Random Forest pipeline.

    Args:
        preprocessor: Preprocessing object.
        random_seed: Random seed.

    Returns:
        Random Forest pipeline.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=8,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def build_xgboost(
    preprocessor: ColumnTransformer,
    random_seed: int,
    scale_pos_weight: float,
) -> Pipeline:
    """
    Builds XGBoost pipeline.

    Args:
        preprocessor: Preprocessing object.
        random_seed: Random seed.
        scale_pos_weight: Positive class weight.

    Returns:
        XGBoost pipeline.
    """

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=random_seed,
                    n_jobs=-1,
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )

    return model


def get_xgboost_search_space() -> Dict:
    """
    Returns XGBoost hyperparameter search space.

    Returns:
        Parameter distribution dictionary.
    """

    return {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.2, 0.5],
        "model__reg_lambda": [1, 3, 5, 10],
    }


def evaluate_model(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict:
    """
    Evaluates model.

    Args:
        model: Trained pipeline.
        X_test: Test features.
        y_test: Test target.

    Returns:
        Metrics dictionary.
    """

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = 0.0

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
            output_dict=True,
        ),
    }

    return metrics


def log_metrics_to_mlflow(metrics: Dict) -> None:
    """
    Logs core model metrics to MLflow.

    Args:
        metrics: Metrics dictionary.
    """

    mlflow.log_metric("accuracy", metrics["accuracy"])
    mlflow.log_metric("precision", metrics["precision"])
    mlflow.log_metric("recall", metrics["recall"])
    mlflow.log_metric("f1_score", metrics["f1_score"])
    mlflow.log_metric("roc_auc", metrics["roc_auc"])


def save_json(data: Dict, path: Path) -> None:
    """
    Saves dictionary as JSON.

    Args:
        data: Dictionary to save.
        path: Output path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def save_model(model: Pipeline, path: Path) -> None:
    """
    Saves model using joblib.

    Args:
        model: Trained model.
        path: Output path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def run_simple_model_experiment(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_dir: Path,
    report_dir: Path,
) -> Dict:
    """
    Trains, evaluates, saves, and logs a simple model experiment in MLflow.

    Args:
        model_name: Name of the model.
        model: Model pipeline.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        model_dir: Directory for saved models.
        report_dir: Directory for reports.

    Returns:
        Experiment result dictionary.
    """

    with mlflow.start_run(run_name=model_name):
        logger.info("Started MLflow run for: %s", model_name)

        model.fit(X_train, y_train)

        metrics = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
        )

        model_path = model_dir / f"{model_name}_mlflow.joblib"
        report_path = report_dir / f"{model_name}_mlflow_metrics.json"

        save_model(model, model_path)
        save_json(metrics, report_path)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", model_name)

        log_metrics_to_mlflow(metrics)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
        )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))

        logger.info("Completed MLflow run for: %s", model_name)

        result = {
            "model_name": model_name,
            "metrics": metrics,
            "model_path": str(model_path),
            "report_path": str(report_path),
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

    return result


def run_xgboost_tuning_experiment(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    scoring_metric: str,
    cv_folds: int,
    n_iter: int,
    random_seed: int,
    model_dir: Path,
    report_dir: Path,
) -> Dict:
    """
    Runs tuned XGBoost experiment with MLflow tracking.

    Args:
        model_name: Name of experiment.
        model: XGBoost pipeline.
        X_train: Training features.
        y_train: Training target.
        X_test: Test features.
        y_test: Test target.
        scoring_metric: Scoring metric for tuning.
        cv_folds: CV folds.
        n_iter: Number of random search iterations.
        random_seed: Random seed.
        model_dir: Model output directory.
        report_dir: Report output directory.

    Returns:
        Experiment result dictionary.
    """

    with mlflow.start_run(run_name=model_name):
        logger.info("Started MLflow run for: %s", model_name)

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=get_xgboost_search_space(),
            n_iter=n_iter,
            scoring=scoring_metric,
            cv=cv_folds,
            random_state=random_seed,
            n_jobs=-1,
            verbose=2,
        )

        search.fit(X_train, y_train)

        best_model = search.best_estimator_

        metrics = evaluate_model(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
        )

        model_path = model_dir / f"{model_name}_mlflow.joblib"
        report_path = report_dir / f"{model_name}_mlflow_metrics.json"
        tuning_path = report_dir / f"{model_name}_tuning_results.csv"

        tuning_results = pd.DataFrame(search.cv_results_)
        tuning_results = tuning_results.sort_values(by="rank_test_score")
        tuning_results.to_csv(tuning_path, index=False)

        experiment_report = {
            "model_name": model_name,
            "selection_metric": scoring_metric,
            "best_cv_score": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
            "test_metrics": metrics,
        }

        save_model(best_model, model_path)
        save_json(experiment_report, report_path)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("model_type", "xgboost")
        mlflow.log_param("selection_metric", scoring_metric)
        mlflow.log_param("cv_folds", cv_folds)
        mlflow.log_param("n_iter", n_iter)

        for param_name, param_value in search.best_params_.items():
            clean_name = param_name.replace("model__", "")
            mlflow.log_param(clean_name, param_value)

        mlflow.log_metric("best_cv_score", round(float(search.best_score_), 4))
        log_metrics_to_mlflow(metrics)

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model",
        )

        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(report_path))
        mlflow.log_artifact(str(tuning_path))

        logger.info("Completed MLflow run for: %s", model_name)

        result = {
            "model_name": model_name,
            "metrics": metrics,
            "best_cv_score": round(float(search.best_score_), 4),
            "best_params": search.best_params_,
            "model_path": str(model_path),
            "report_path": str(report_path),
            "tuning_path": str(tuning_path),
            "mlflow_run_id": mlflow.active_run().info.run_id,
        }

    return result


def select_best_experiment(results: Dict[str, Dict]) -> str:
    """
    Selects best experiment based on F1-score.

    Args:
        results: Experiment results.

    Returns:
        Best experiment name.
    """

    best_model_name = max(
        results,
        key=lambda name: results[name]["metrics"]["f1_score"],
    )

    return best_model_name


def main() -> None:
    """
    Runs MLflow experiment tracking for all models.
    """

    config = load_config()
    root_dir = get_project_root()

    tracking_uri = config["mlflow"]["tracking_uri"]
    experiment_name = config["mlflow"]["experiment_name"]

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    processed_dir = root_dir / config["dataset"]["processed_dir"]
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    target_column = config["dataset"]["target_column"]
    id_column = config["dataset"]["id_column"]
    numeric_columns = config["validation"]["numeric_columns"]
    categorical_columns = config["validation"]["categorical_columns"]

    random_seed = config["project"]["random_seed"]
    scoring_metric = config["model_training"]["scoring_metric"]
    cv_folds = config["model_training"]["cv_folds"]
    n_iter = config["model_training"]["n_iter"]

    model_dir = root_dir / "models"
    report_dir = root_dir / "reports" / "model"

    train_data, test_data = load_train_test_data(
        train_path=train_path,
        test_path=test_path,
    )

    X_train, y_train = split_features_target(
        data=train_data,
        target_column=target_column,
        id_column=id_column,
    )

    X_test, y_test = split_features_target(
        data=test_data,
        target_column=target_column,
        id_column=id_column,
    )

    scale_pos_weight = calculate_scale_pos_weight(y_train)

    results = {}

    logistic_preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    random_forest_preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    xgboost_preprocessor = build_preprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
    )

    logistic_model = build_logistic_regression(
        preprocessor=logistic_preprocessor,
        random_seed=random_seed,
    )

    random_forest_model = build_random_forest(
        preprocessor=random_forest_preprocessor,
        random_seed=random_seed,
    )

    xgboost_model = build_xgboost(
        preprocessor=xgboost_preprocessor,
        random_seed=random_seed,
        scale_pos_weight=scale_pos_weight,
    )

    results["logistic_regression"] = run_simple_model_experiment(
        model_name="logistic_regression",
        model=logistic_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_dir=model_dir,
        report_dir=report_dir,
    )

    results["random_forest"] = run_simple_model_experiment(
        model_name="random_forest",
        model=random_forest_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_dir=model_dir,
        report_dir=report_dir,
    )

    results["xgboost_tuned"] = run_xgboost_tuning_experiment(
        model_name="xgboost_tuned",
        model=xgboost_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        scoring_metric=scoring_metric,
        cv_folds=cv_folds,
        n_iter=n_iter,
        random_seed=random_seed,
        model_dir=model_dir,
        report_dir=report_dir,
    )

    best_model_name = select_best_experiment(results)
    best_model_source_path = Path(results[best_model_name]["model_path"])
    best_model_target_path = model_dir / "best_tracked_model.joblib"

    best_model = joblib.load(best_model_source_path)
    save_model(best_model, best_model_target_path)

    summary_rows = []

    for model_name, result in results.items():
        metrics = result["metrics"]

        summary_rows.append(
            {
                "model_name": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
                "roc_auc": metrics["roc_auc"],
                "mlflow_run_id": result["mlflow_run_id"],
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(by="f1_score", ascending=False)

    summary_path = report_dir / "mlflow_experiment_summary.csv"
    summary.to_csv(summary_path, index=False)

    final_report = {
        "best_model": best_model_name,
        "best_model_path": str(best_model_target_path),
        "selection_metric": "f1_score",
        "tracking_uri": tracking_uri,
        "experiment_name": experiment_name,
        "results": results,
    }

    final_report_path = report_dir / "mlflow_experiment_report.json"
    save_json(final_report, final_report_path)

    logger.info("MLflow experiment summary saved at: %s", summary_path)
    logger.info("MLflow experiment report saved at: %s", final_report_path)
    logger.info("Best tracked model saved at: %s", best_model_target_path)
    logger.info("Best model selected: %s", best_model_name)
    logger.info("MLflow experiment tracking completed successfully.")


if __name__ == "__main__":
    main()
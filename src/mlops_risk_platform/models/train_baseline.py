import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlops_risk_platform.utils import load_config, get_project_root
from mlops_risk_platform.utils.logger import get_logger

logger = get_logger(__name__)


def load_train_test(train_path: Path, test_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not train_path.exists():
        raise FileNotFoundError(f"Train data not found at {train_path}. Run ingest_data.py first.")
    if not test_path.exists():
        raise FileNotFoundError(f"Test data not found at {test_path}. Run ingest_data.py first.")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    logger.info(f"Train data shape: {train_df.shape}")
    logger.info(f"Test data shape: {test_df.shape}")

    return train_df, test_df


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features),
    ])


def build_models(preprocessor: ColumnTransformer, random_seed: int) -> Dict[str, Pipeline]:
    logistic = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(random_state=random_seed, max_iter=1000, solver="liblinear")),
    ])
    random_forest = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", RandomForestClassifier(
            random_state=random_seed,
            n_estimators=200,
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
        )),
    ])
    logger.info("Baseline models built successfully")
    return {"logistic": logistic, "random_forest": random_forest}


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }
    logger.info(f"Evaluation done — F1: {metrics['f1_score']}, ROC-AUC: {metrics['roc_auc']}")
    return metrics


def save_json(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON saved at {path}")


def save_model(model: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Model saved at {path}")


def get_best_model(models: Dict[str, Pipeline], metrics: Dict) -> Tuple[str, Pipeline]:
    best_name = max(metrics, key=lambda x: metrics[x]["f1_score"])
    logger.info(f"Best model: {best_name} — F1: {metrics[best_name]['f1_score']}")
    return best_name, models[best_name]


def plot_metrics(metrics: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(metrics.keys())
    f1_scores = [metrics[m]["f1_score"] for m in model_names]
    roc_aucs = [metrics[m]["roc_auc"] for m in model_names]

    x = range(len(model_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], f1_scores, width, label="F1 Score", color="steelblue")
    ax.bar([i + width / 2 for i in x], roc_aucs, width, label="ROC AUC", color="coral")
    ax.set_title("Model Comparison")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks(list(x))
    ax.set_xticklabels(model_names)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png")
    plt.close()
    logger.info(f"Metrics plot saved at {output_dir / 'model_comparison.png'}")


def main() -> None:
    config = load_config("config/settings.yaml")
    root = get_project_root()

    processed_dir = root / config["dataset"]["processed_dir"]
    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"
    models_dir = root / "models"
    reports_dir = root / "reports" / "models"

    target_col = config["dataset"]["target_column"]
    id_col = config["dataset"]["id_column"]
    numeric_features = config["validation"]["numeric_columns"]
    categorical_features = config["validation"]["categorical_columns"]
    random_seed = config["project"]["random_seed"]

    train_df, test_df = load_train_test(train_path, test_path)

    drop_cols = [target_col, id_col]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col]

    preprocessor = build_preprocessor(numeric_features, categorical_features)
    models = build_models(preprocessor, random_seed)

    metrics = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        metrics[name] = evaluate_model(model, X_test, y_test)

    save_json(metrics, reports_dir / "metrics.json")
    plot_metrics(metrics, reports_dir)

    best_name, best = get_best_model(models, metrics)
    save_model(best, models_dir / "best_model.joblib")

    logger.info("Baseline training completed successfully")


if __name__ == "__main__":
    main()

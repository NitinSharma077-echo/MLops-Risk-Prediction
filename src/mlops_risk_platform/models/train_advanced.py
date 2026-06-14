import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

from mlops_risk_platform.utils import load_config, get_project_root
from mlops_risk_platform.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    config = load_config("config/settings.yaml")
    root = get_project_root()

    processed_dir = root / config["dataset"]["processed_dir"]
    train_df = pd.read_csv(processed_dir / "train.csv")
    test_df = pd.read_csv(processed_dir / "test.csv")
    logger.info("Train shape: %s | Test shape: %s", train_df.shape, test_df.shape)

    target_col = config["dataset"]["target_column"]
    id_col = config["dataset"]["id_column"]
    numeric_cols = config["validation"]["numeric_columns"]
    categorical_cols = config["validation"]["categorical_columns"]
    random_seed = config["project"]["random_seed"]
    scoring_metric = config["model_training"]["scoring_metric"]
    cv_folds = config["model_training"]["cv_folds"]
    n_iter = config["model_training"]["n_iter"]

    drop_cols = [target_col, id_col]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col]
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col]

    scale_pos_weight = round((y_train == 0).sum() / (y_train == 1).sum(), 4)
    logger.info("scale_pos_weight: %s", scale_pos_weight)

    preprocessor = ColumnTransformer(transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_seed,
            n_jobs=-1,
            scale_pos_weight=scale_pos_weight,
        )),
    ])

    param_grid = {
        "model__n_estimators": [100, 200, 300, 500],
        "model__max_depth": [3, 4, 5, 6, 8],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.7, 0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5, 7],
        "model__gamma": [0, 0.1, 0.2, 0.5],
        "model__reg_lambda": [1, 3, 5, 10],
    }

    logger.info("Starting XGBoost hyperparameter tuning...")
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_grid,
        n_iter=n_iter,
        scoring=scoring_metric,
        cv=cv_folds,
        random_state=random_seed,
        n_jobs=-1,
        verbose=2,
    )
    search.fit(X_train, y_train)
    logger.info("Tuning complete. Best CV score: %s", round(search.best_score_, 4))

    best_model = search.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_proba)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test, y_pred, zero_division=0, output_dict=True
        ),
    }

    report = {
        "model_name": "xgboost_tuned",
        "selection_metric": scoring_metric,
        "best_cv_score": round(float(search.best_score_), 4),
        "scale_pos_weight": scale_pos_weight,
        "best_params": search.best_params_,
        "test_metrics": metrics,
    }

    report_dir = root / "reports" / "models"
    model_dir = root / "models"
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    with open(report_dir / "advanced_xgboost_metrics.json", "w") as f:
        json.dump(report, f, indent=4)
    logger.info("Report saved at: %s", report_dir / "advanced_xgboost_metrics.json")

    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv(
        report_dir / "xgboost_tuning_results.csv", index=False
    )
    logger.info("Tuning results saved at: %s", report_dir / "xgboost_tuning_results.csv")

    joblib.dump(best_model, model_dir / "advanced_xgboost_model.joblib")
    logger.info("Model saved at: %s", model_dir / "advanced_xgboost_model.joblib")

    logger.info("Test metrics: %s", metrics)
    logger.info("Advanced model training completed successfully.")


if __name__ == "__main__":
    main()

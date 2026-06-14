import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from mlops_risk_platform.config import load_config, get_project_root
from mlops_risk_platform.utils.logger import get_logger


logger = get_logger(__name__)


def load_model(model_path: Path) -> Pipeline:
    """
    Loads trained model pipeline.

    Args:
        model_path: Path to saved model.

    Returns:
        Loaded Scikit-learn pipeline.
    """

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            f"Please run Step 6: track_experiments.py first."
        )

    model = joblib.load(model_path)
    logger.info("Model loaded successfully from: %s", model_path)

    return model


def load_test_data(test_path: Path) -> pd.DataFrame:
    """
    Loads test dataset.

    Args:
        test_path: Path to test CSV.

    Returns:
        Test DataFrame.
    """

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_path}. "
            f"Please run Step 2: ingest_data.py first."
        )

    data = pd.read_csv(test_path)
    logger.info("Test data loaded from: %s", test_path)
    logger.info("Test data shape: %s", data.shape)

    return data


def split_features_target(
    data: pd.DataFrame,
    target_column: str,
    id_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Splits dataset into features, target, and IDs.

    Args:
        data: Input DataFrame.
        target_column: Target column name.
        id_column: ID column name.

    Returns:
        X, y, and IDs.
    """

    ids = data[id_column]
    X = data.drop(columns=[target_column, id_column])
    y = data[target_column]

    return X, y, ids


def get_pipeline_parts(model_pipeline: Pipeline):
    """
    Extracts preprocessor and estimator from model pipeline.

    Args:
        model_pipeline: Trained pipeline.

    Returns:
        Preprocessor and estimator.
    """

    if "preprocessor" not in model_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'preprocessor' step.")

    if "model" not in model_pipeline.named_steps:
        raise ValueError("Pipeline does not contain a 'model' step.")

    preprocessor = model_pipeline.named_steps["preprocessor"]
    estimator = model_pipeline.named_steps["model"]

    return preprocessor, estimator


def transform_features(
    model_pipeline: Pipeline,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Applies the fitted preprocessor and returns transformed features.

    Args:
        model_pipeline: Trained model pipeline.
        X: Raw feature DataFrame.

    Returns:
        Transformed feature array and feature names.
    """

    preprocessor, _ = get_pipeline_parts(model_pipeline)

    transformed_X = preprocessor.transform(X)

    if hasattr(preprocessor, "get_feature_names_out"):
        feature_names = preprocessor.get_feature_names_out().tolist()
    else:
        feature_names = [f"feature_{i}" for i in range(transformed_X.shape[1])]

    logger.info("Transformed feature shape: %s", transformed_X.shape)

    return transformed_X, feature_names


def prepare_shap_values(
    estimator,
    background_data: np.ndarray,
    explain_data: np.ndarray,
    feature_names: List[str],
) -> shap.Explanation:
    """
    Creates SHAP explanation object.

    This function uses SHAP's general Explainer interface.
    For tree models, SHAP automatically uses a suitable tree-based explainer
    where possible. For linear models, it uses a linear explainer.

    Args:
        estimator: Trained estimator.
        background_data: Background data used by SHAP.
        explain_data: Data to explain.
        feature_names: Feature names after preprocessing.

    Returns:
        SHAP Explanation object for the positive class.
    """

    logger.info("Creating SHAP explainer.")

    explainer = shap.Explainer(
        estimator,
        background_data,
        feature_names=feature_names,
    )

    shap_values = explainer(explain_data)

    if len(shap_values.values.shape) == 3:
        positive_class_values = shap.Explanation(
            values=shap_values.values[:, :, 1],
            base_values=shap_values.base_values[:, 1],
            data=shap_values.data,
            feature_names=feature_names,
        )
    else:
        positive_class_values = shap_values

    logger.info("SHAP values created successfully.")

    return positive_class_values


def save_global_bar_plot(
    shap_values: shap.Explanation,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """
    Saves SHAP global bar plot.

    Args:
        shap_values: SHAP Explanation object.
        output_path: Output path.
        max_display: Number of top features to display.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.bar(
        shap_values,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP global bar plot at: %s", output_path)


def save_beeswarm_plot(
    shap_values: shap.Explanation,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """
    Saves SHAP beeswarm plot.

    Args:
        shap_values: SHAP Explanation object.
        output_path: Output path.
        max_display: Number of top features to display.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.beeswarm(
        shap_values,
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("Saved SHAP beeswarm plot at: %s", output_path)


def save_local_waterfall_plot(
    shap_values: shap.Explanation,
    row_index: int,
    output_path: Path,
    max_display: int = 15,
) -> None:
    """
    Saves local SHAP waterfall plot for one prediction.

    Args:
        shap_values: SHAP Explanation object.
        row_index: Row index to explain.
        output_path: Output path.
        max_display: Number of top features to display.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure()
    shap.plots.waterfall(
        shap_values[row_index],
        max_display=max_display,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    logger.info("Saved local waterfall plot at: %s", output_path)


def create_global_feature_importance(
    shap_values: shap.Explanation,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Creates global SHAP feature importance table.

    Args:
        shap_values: SHAP Explanation object.
        feature_names: Feature names.

    Returns:
        Feature importance DataFrame.
    """

    mean_abs_values = np.abs(shap_values.values).mean(axis=0)

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap_value": mean_abs_values,
        }
    )

    importance = importance.sort_values(
        by="mean_absolute_shap_value",
        ascending=False,
    )

    return importance


def create_local_explanation(
    shap_values: shap.Explanation,
    row_index: int,
    feature_names: List[str],
    customer_id: str,
    prediction_probability: float,
    prediction_label: int,
    top_n: int = 10,
) -> Dict:
    """
    Creates local explanation for one customer.

    Args:
        shap_values: SHAP Explanation object.
        row_index: Row to explain.
        feature_names: Feature names.
        customer_id: Customer ID.
        prediction_probability: Predicted risk probability.
        prediction_label: Predicted class label.
        top_n: Number of top features.

    Returns:
        Local explanation dictionary.
    """

    row_shap_values = shap_values.values[row_index]

    local_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "shap_value": row_shap_values,
            "absolute_shap_value": np.abs(row_shap_values),
        }
    )

    local_importance = local_importance.sort_values(
        by="absolute_shap_value",
        ascending=False,
    ).head(top_n)

    risk_increasing_features = local_importance[
        local_importance["shap_value"] > 0
    ][["feature", "shap_value"]].to_dict(orient="records")

    risk_decreasing_features = local_importance[
        local_importance["shap_value"] < 0
    ][["feature", "shap_value"]].to_dict(orient="records")

    explanation = {
        "customer_id": customer_id,
        "prediction_probability": round(float(prediction_probability), 4),
        "prediction_label": int(prediction_label),
        "prediction_meaning": "high_risk" if prediction_label == 1 else "low_risk",
        "top_features": local_importance[
            ["feature", "shap_value"]
        ].to_dict(orient="records"),
        "risk_increasing_features": risk_increasing_features,
        "risk_decreasing_features": risk_decreasing_features,
    }

    return explanation


def save_dataframe(data: pd.DataFrame, path: Path) -> None:
    """
    Saves DataFrame as CSV.

    Args:
        data: DataFrame.
        path: Output path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False)
    logger.info("Saved DataFrame at: %s", path)


def save_json(data: Dict, path: Path) -> None:
    """
    Saves dictionary as JSON.

    Args:
        data: Dictionary.
        path: Output path.
    """

    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

    logger.info("Saved JSON at: %s", path)


def main() -> None:
    """
    Runs SHAP explainability pipeline.
    """

    config = load_config()
    root_dir = get_project_root()

    processed_dir = root_dir / config["dataset"]["processed_dir"]
    test_path = processed_dir / "test.csv"

    target_column = config["dataset"]["target_column"]
    id_column = config["dataset"]["id_column"]

    model_path = root_dir / "models" / "best_tracked_model.joblib"

    report_dir = root_dir / "reports" / "explainability"
    charts_dir = report_dir / "charts"
    local_dir = report_dir / "local"

    model_pipeline = load_model(model_path)

    test_data = load_test_data(test_path)

    X_test, y_test, customer_ids = split_features_target(
        data=test_data,
        target_column=target_column,
        id_column=id_column,
    )

    sample_size = min(300, len(X_test))
    background_size = min(100, len(X_test))

    X_sample = X_test.sample(
        n=sample_size,
        random_state=config["project"]["random_seed"],
    )

    X_background = X_test.sample(
        n=background_size,
        random_state=config["project"]["random_seed"],
    )

    transformed_background, feature_names = transform_features(
        model_pipeline=model_pipeline,
        X=X_background,
    )

    transformed_sample, feature_names = transform_features(
        model_pipeline=model_pipeline,
        X=X_sample,
    )

    _, estimator = get_pipeline_parts(model_pipeline)

    shap_values = prepare_shap_values(
        estimator=estimator,
        background_data=transformed_background,
        explain_data=transformed_sample,
        feature_names=feature_names,
    )

    save_global_bar_plot(
        shap_values=shap_values,
        output_path=charts_dir / "global_feature_importance_bar.png",
        max_display=15,
    )

    save_beeswarm_plot(
        shap_values=shap_values,
        output_path=charts_dir / "global_feature_impact_beeswarm.png",
        max_display=15,
    )

    global_importance = create_global_feature_importance(
        shap_values=shap_values,
        feature_names=feature_names,
    )

    save_dataframe(
        data=global_importance,
        path=report_dir / "global_feature_importance.csv",
    )

    probabilities = model_pipeline.predict_proba(X_sample)[:, 1]
    predictions = model_pipeline.predict(X_sample)

    local_row_index = 0
    original_customer_id = str(test_data.loc[X_sample.index[local_row_index], id_column])

    save_local_waterfall_plot(
        shap_values=shap_values,
        row_index=local_row_index,
        output_path=local_dir / "local_waterfall_explanation.png",
        max_display=15,
    )

    local_explanation = create_local_explanation(
        shap_values=shap_values,
        row_index=local_row_index,
        feature_names=feature_names,
        customer_id=original_customer_id,
        prediction_probability=float(probabilities[local_row_index]),
        prediction_label=int(predictions[local_row_index]),
        top_n=10,
    )

    save_json(
        data=local_explanation,
        path=local_dir / "local_prediction_explanation.json",
    )

    summary = {
        "model_path": str(model_path),
        "explained_rows": int(sample_size),
        "background_rows": int(background_size),
        "top_10_global_features": global_importance.head(10).to_dict(
            orient="records"
        ),
        "generated_files": {
            "global_bar_plot": str(charts_dir / "global_feature_importance_bar.png"),
            "beeswarm_plot": str(charts_dir / "global_feature_impact_beeswarm.png"),
            "global_importance_csv": str(report_dir / "global_feature_importance.csv"),
            "local_waterfall_plot": str(local_dir / "local_waterfall_explanation.png"),
            "local_explanation_json": str(
                local_dir / "local_prediction_explanation.json"
            ),
        },
    }

    save_json(
        data=summary,
        path=report_dir / "explainability_summary.json",
    )

    logger.info("Explainability pipeline completed successfully.")


if __name__ == "__main__":
    main()
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from mlops_risk_platform.utils import get_project_root, load_config
from mlops_risk_platform.utils.logger import get_logger

logger = get_logger(__name__)


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at {raw_path}")
    data = pd.read_csv(raw_path)
    logger.info(f"Raw data loaded successfully. Shape: {data.shape}")
    return data


def save_dataframe(data: Dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"Dataframe saved successfully at: {path}")


def plot_distribution(data: pd.DataFrame, target_column: str, output: Path) -> None:
    counts = data[target_column].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(x=counts.index, y=counts.values, palette='viridis')
    plt.title(f"Distribution of {target_column}")
    plt.xlabel("Target Variable")
    plt.ylabel("Frequency")
    plt.savefig(output)
    plt.close()
    logger.info(f"Distribution plot saved successfully at: {output}")


def plot_risk(data: pd.DataFrame, category: str, target: str, output: Path) -> None:
    group = data.groupby(category)[target].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=group.index, y=group.values, palette='magma')
    plt.title(f"Risk Event Rate by {category}")
    plt.xlabel(category)
    plt.ylabel("Risk Event Rate")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output)
    plt.close()
    logger.info(f"Risk plot saved successfully at {output}")


def plot_numeric(data: pd.DataFrame, numerical_features: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature].dropna(), kde=True, color='steelblue')
        plt.title(f"Distribution of {feature}")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(output_dir / f"{feature}_distribution.png")
        plt.close()
        logger.info(f"Numeric plot saved for feature: {feature}")


def target_summary(data: pd.DataFrame, target_column: str) -> Dict:
    summary = data[target_column].describe().to_dict()
    logger.info(f"Target summary: {summary}")
    return summary


def correlation_matrix(data: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(12, 10))
    sns.heatmap(data.select_dtypes(include="number").corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.savefig(output)
    plt.close()
    logger.info(f"Correlation matrix saved successfully at: {output}")


def business_insights(data: pd.DataFrame, output: Path, country: str, device: str) -> Dict:
    target_col = "risk_event"

    target_dist = data[target_col].value_counts(normalize=True).mul(100)
    risky_percentage = float(target_dist.get(1, 0))

    country_summary = data.groupby(country)[target_col].mean().sort_values(ascending=False)
    highest_risk_country = country_summary.index[0]
    highest_country_risk_rate = float(country_summary.iloc[0])

    device_summary = data.groupby(device)[target_col].mean().sort_values(ascending=False)
    highest_risk_device = device_summary.index[0]
    highest_device_risk_rate = float(device_summary.iloc[0])

    numeric_cols = [c for c in data.select_dtypes(include="number").columns if c != target_col]
    numeric_comparison = (
        data.groupby(target_col)[numeric_cols].mean().T
        .assign(absolute_difference=lambda df: (df[1] - df[0]).abs())
        .sort_values("absolute_difference", ascending=False)
    )
    top_numeric_difference = numeric_comparison.index[0]
    top_numeric_difference_value = float(numeric_comparison.iloc[0]["absolute_difference"])

    insights = {
        "target_distribution": {
            "risk_event_percentage": risky_percentage,
            "interpretation": (
                "This shows the percentage of records marked as risky. "
                "If this value is low, the dataset has class imbalance."
            ),
        },
        "highest_risk_country": {
            "country": str(highest_risk_country),
            "risk_rate": highest_country_risk_rate,
        },
        "highest_risk_device_type": {
            "device_type": str(highest_risk_device),
            "risk_rate": highest_device_risk_rate,
        },
        "strongest_numeric_difference": {
            "feature": str(top_numeric_difference),
            "difference_between_risk_and_non_risk": top_numeric_difference_value,
        },
        "business_summary": (
            f"The dataset has {risky_percentage:.2f}% risky records. "
            f"The highest observed country-level risk rate is for {highest_risk_country}. "
            f"The highest observed device-level risk rate is for {highest_risk_device}. "
            f"The feature with the strongest average difference between risky and "
            f"non-risky users is {top_numeric_difference}."
        ),
    }

    save_dataframe(insights, output)
    return insights


def main() -> None:
    from mlops_risk_platform.utils import get_project_root
    config = load_config("config/settings.yaml")

    root = get_project_root()
    charts_dir = root / "reports" / "eda" / "charts"
    insights_dir = root / "reports" / "eda" / "insights"
    charts_dir.mkdir(parents=True, exist_ok=True)
    insights_dir.mkdir(parents=True, exist_ok=True)



    raw_data = load_raw_data(root / config["dataset"]["raw_path"])
    target_col = config["dataset"]["target_column"]
    numeric_features = config["validation"]["numeric_columns"]
    categorical_cols = config["validation"]["categorical_columns"]

    plot_distribution(raw_data, target_col, charts_dir / "target_distribution.png")
    plot_risk(raw_data, categorical_cols[0], target_col, charts_dir / "risk_by_country.png")
    plot_risk(raw_data, categorical_cols[1], target_col, charts_dir / "risk_by_device.png")
    plot_numeric(raw_data, numeric_features, charts_dir / "numeric_distributions")
    correlation_matrix(raw_data, charts_dir / "correlation_matrix.png")
    target_summary(raw_data, target_col)
    business_insights(
        raw_data,
        insights_dir / "business_insights.json",
        categorical_cols[0],
        categorical_cols[1],
    )


if __name__ == "__main__":
    main()

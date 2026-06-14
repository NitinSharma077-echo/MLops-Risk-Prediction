import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split

from mlops_risk_platform.utils import load_config, get_project_root
from mlops_risk_platform.utils.logger import get_logger


logger = get_logger(__name__)


class DataValidationError(Exception):
    """
    Custom exception for data validation errors.
    """

    pass


def check_required_columns(data: pd.DataFrame, required_columns: List[str]) -> None:
    """
    Checks whether all required columns are present in the dataset.

    Args:
        data: Input DataFrame.
        required_columns: List of required columns.

    Raises:
        DataValidationError: If any required column is missing.
    """

    missing_columns = [col for col in required_columns if col not in data.columns]

    if missing_columns:
        raise DataValidationError(f"Missing required columns: {missing_columns}")


def check_missing_values(data: pd.DataFrame) -> Dict[str, int]:
    """
    Checks missing values in the dataset.

    Args:
        data: Input DataFrame.

    Returns:
        Dictionary containing missing value count for each column.
    """

    missing_values = data.isnull().sum()
    missing_dict = missing_values[missing_values > 0].to_dict()

    return missing_dict


def check_duplicate_ids(data: pd.DataFrame, id_column: str) -> int:
    """
    Checks duplicate IDs.

    Args:
        data: Input DataFrame.
        id_column: ID column name.

    Returns:
        Number of duplicate IDs.
    """

    duplicate_count = data[id_column].duplicated().sum()

    return int(duplicate_count)


def check_target_column(
    data: pd.DataFrame,
    target_column: str,
    allowed_values: List[int],
) -> None:
    """
    Validates target column values.

    Args:
        data: Input DataFrame.
        target_column: Target column name.
        allowed_values: Allowed binary values.

    Raises:
        DataValidationError: If target column contains invalid values.
    """

    unique_values = sorted(data[target_column].dropna().unique().tolist())

    invalid_values = [value for value in unique_values if value not in allowed_values]

    if invalid_values:
        raise DataValidationError(
            f"Invalid target values found: {invalid_values}. "
            f"Allowed values are: {allowed_values}"
        )


def check_numeric_columns(data: pd.DataFrame, numeric_columns: List[str]) -> None:
    """
    Checks whether numeric columns contain numeric data.

    Args:
        data: Input DataFrame.
        numeric_columns: List of numeric column names.

    Raises:
        DataValidationError: If any numeric column has non-numeric values.
    """

    for column in numeric_columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise DataValidationError(f"Column '{column}' must be numeric.")


def validate_data(data: pd.DataFrame, config: Dict) -> Dict:
    """
    Runs all validation checks.

    Args:
        data: Input DataFrame.
        config: Project configuration.

    Returns:
        Data validation report.
    """

    validation_config = config["validation"]
    dataset_config = config["dataset"]

    required_columns = validation_config["required_columns"]
    numeric_columns = validation_config["numeric_columns"]
    allowed_values = validation_config["allowed_binary_values"]

    id_column = dataset_config["id_column"]
    target_column = dataset_config["target_column"]

    check_required_columns(data, required_columns)
    check_numeric_columns(data, numeric_columns)
    check_target_column(data, target_column, allowed_values)

    missing_values = check_missing_values(data)
    duplicate_ids = check_duplicate_ids(data, id_column)

    if duplicate_ids > 0:
        raise DataValidationError(f"Duplicate IDs found: {duplicate_ids}")

    if missing_values:
        raise DataValidationError(f"Missing values found: {missing_values}")

    target_distribution = data[target_column].value_counts(normalize=True).to_dict()

    validation_report = {
        "status": "passed",
        "total_rows": int(data.shape[0]),
        "total_columns": int(data.shape[1]),
        "missing_values": missing_values,
        "duplicate_ids": duplicate_ids,
        "target_distribution": {
            str(key): round(float(value), 4)
            for key, value in target_distribution.items()
        },
    }

    return validation_report


def save_validation_report(report: Dict, report_path: Path) -> None:
    """
    Saves validation report as JSON.

    Args:
        report: Validation report dictionary.
        report_path: Path where report should be saved.
    """

    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)


def split_and_save_data(data: pd.DataFrame, config: Dict, root_dir: Path) -> None:
    """
    Splits data into train and test sets, then saves them.

    Args:
        data: Input DataFrame.
        config: Project configuration.
        root_dir: Project root directory.
    """

    target_column = config["dataset"]["target_column"]
    processed_dir = root_dir / config["dataset"]["processed_dir"]
    test_size = config["dataset"]["test_size"]
    random_seed = config["project"]["random_seed"]

    processed_dir.mkdir(parents=True, exist_ok=True)

    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_seed,
        stratify=data[target_column],
    )

    train_path = processed_dir / "train.csv"
    test_path = processed_dir / "test.csv"

    train_data.to_csv(train_path, index=False)
    test_data.to_csv(test_path, index=False)

    logger.info("Train data saved at: %s", train_path)
    logger.info("Test data saved at: %s", test_path)
    logger.info("Train shape: %s", train_data.shape)
    logger.info("Test shape: %s", test_data.shape)


def main() -> None:
    """
    Main function for data ingestion and validation.
    """

    config = load_config("config/settings.yaml")
    root_dir = get_project_root()

    raw_path = root_dir / config["dataset"]["raw_path"]

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_path}. "
            f"Run generate_sample_data.py first."
        )

    logger.info("Reading raw data from: %s", raw_path)

    data = pd.read_csv(raw_path)

    logger.info("Raw dataset loaded successfully.")
    logger.info("Raw data shape: %s", data.shape)

    validation_report = validate_data(data, config)

    report_path = root_dir / "reports" / "data" / "validation_report.json"
    save_validation_report(validation_report, report_path)

    logger.info("Data validation passed.")
    logger.info("Validation report saved at: %s", report_path)

    split_and_save_data(data, config, root_dir)

    logger.info("Data ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd

from mlops_risk_platform.utils import load_config, get_project_root
from mlops_risk_platform.utils.logger import get_logger


logger = get_logger(__name__)


def generate_synthetic_risk_data(rows: int = 5000) -> pd.DataFrame:
    """
    Generates a synthetic binary risk dataset.

    The target column 'risk_event' can represent:
    - Fraud event
    - Customer churn
    - Credit default
    - Suspicious user behavior

    Args:
        rows: Number of rows to generate.

    Returns:
        Pandas DataFrame containing synthetic risk data.
    """

    np.random.seed(42)

    customer_ids = [f"CUST_{str(i).zfill(6)}" for i in range(1, rows + 1)]

    account_age_days = np.random.randint(10, 3000, rows)
    transaction_count_30d = np.random.poisson(lam=25, size=rows)
    avg_transaction_amount = np.random.gamma(shape=2.0, scale=120.0, size=rows)
    failed_login_count_7d = np.random.poisson(lam=1.5, size=rows)
    support_tickets_90d = np.random.poisson(lam=2.0, size=rows)
    monthly_charges = np.random.normal(loc=1200, scale=400, size=rows)
    total_spend = monthly_charges * (account_age_days / 30) + np.random.normal(
        loc=0,
        scale=3000,
        size=rows,
    )
    credit_score = np.random.normal(loc=680, scale=80, size=rows)

    countries = np.random.choice(
        ["India", "USA", "Canada", "UK", "Australia"],
        size=rows,
        p=[0.45, 0.25, 0.12, 0.10, 0.08],
    )

    device_types = np.random.choice(
        ["Mobile", "Desktop", "Tablet"],
        size=rows,
        p=[0.65, 0.28, 0.07],
    )

    monthly_charges = np.maximum(monthly_charges, 100)
    total_spend = np.maximum(total_spend, 0)
    credit_score = np.clip(credit_score, 300, 900)

    risk_score = (
        0.002 * avg_transaction_amount
        + 0.35 * failed_login_count_7d
        + 0.20 * support_tickets_90d
        - 0.002 * credit_score
        - 0.0002 * account_age_days
        + np.where(device_types == "Mobile", 0.20, 0.0)
        + np.where(countries == "India", 0.10, 0.0)
    )

    probability = 1 / (1 + np.exp(-risk_score))

    adjusted_probability = probability * 0.35

    risk_event = np.random.binomial(1, adjusted_probability)

    data = pd.DataFrame(
        {
            "customer_id": customer_ids,
            "account_age_days": account_age_days,
            "transaction_count_30d": transaction_count_30d,
            "avg_transaction_amount": avg_transaction_amount.round(2),
            "failed_login_count_7d": failed_login_count_7d,
            "support_tickets_90d": support_tickets_90d,
            "monthly_charges": monthly_charges.round(2),
            "total_spend": total_spend.round(2),
            "credit_score": credit_score.round(0).astype(int),
            "country": countries,
            "device_type": device_types,
            "risk_event": risk_event,
        }
    )

    return data


def main() -> None:
    """
    Main function to generate and save synthetic raw data.
    """

    config = load_config("config/settings.yaml")
    root_dir = get_project_root()

    raw_path = root_dir / config["dataset"]["raw_path"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_synthetic_risk_data(rows=5000)

    data.to_csv(raw_path, index=False)

    logger.info("Synthetic dataset generated successfully.")
    logger.info("Dataset shape: %s", data.shape)
    logger.info("Saved raw data at: %s", raw_path)
    logger.info("Target distribution:\n%s", data["risk_event"].value_counts(normalize=True))


if __name__ == "__main__":
    main()
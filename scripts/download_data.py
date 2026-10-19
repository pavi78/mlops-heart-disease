
from pathlib import Path
import io
import zipfile

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_FILE = DATA_DIR / "heart_disease_raw.csv"

UCI_URL = (
    "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
)

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak", "slope",
    "ca", "thal", "num",
]


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(UCI_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        filename = "processed.cleveland.data"
        with archive.open(filename) as file:
            df = pd.read_csv(
                file,
                header=None,
                names=COLUMNS,
                na_values="?",
                skipinitialspace=True,
            )

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Dataset saved to: {OUTPUT_FILE}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()

import pandas as pd
from src.paths import RAW_DIR, PROCESSED_DIR

RAW_FILE = RAW_DIR / "bank-additional-full.csv"
OUT_FILE = PROCESSED_DIR / "dataset_clean.parquet"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading raw data...")
    df = pd.read_csv(RAW_FILE, sep=";")

    # ---- Basic cleaning ----
    df = df.drop_duplicates()

    # Convert target to binary
    df["y"] = df["y"].map({"yes": 1, "no": 0})

    # Remove unknown values
    df = df.replace("unknown", pd.NA).dropna()

    # Save processed dataset
    df.to_parquet(OUT_FILE, index=False)
    print(f"Processed dataset saved to: {OUT_FILE}")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")


if __name__ == "__main__":
    main()

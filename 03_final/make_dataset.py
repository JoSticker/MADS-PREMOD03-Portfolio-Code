from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "01_data" / "raw" / "bank-additional-full.csv"
OUT_FILE = ROOT / "01_data" / "processed" / "dataset_clean.parquet"



def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_FILE, sep=";")

    # Drop duplicates in Dataframe
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

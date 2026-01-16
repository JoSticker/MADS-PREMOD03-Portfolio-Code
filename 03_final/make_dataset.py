from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_FILE = ROOT / "01_data" / "raw" / "bank-additional-full.csv"
OUT_FILE = ROOT / "01_data" / "processed" / "dataset_clean.parquet"



def main():
    print("Loading raw data...")
    df = pd.read_csv(RAW_FILE, sep=";")

    # length of raw data
    rows_before = len(df)


    # Drop duplicates in Dataframe
    df = df.drop_duplicates()

    # Convert target to binary
    df["y"] = df["y"].map({"yes": 1, "no": 0}).astype("int64")

    # Remove unknown values
    df = df.replace("unknown", pd.NA).dropna()

    rows_after = len(df)
print(f"Rows before cleaning: {rows_before}")
print(f"Rows after cleaning:  {rows_after}")


    # Data quality checks
missing = int(df.isna().sum().sum())
duplicates = int(df.duplicated().sum())
unknown_present = bool((df.select_dtypes(include="object") == "unknown").any().any())
y_unique = sorted(df["y"].unique().tolist())

print(
    f"missing={missing}, "
    f"duplicates={duplicates}, "
    f"unknown_present={unknown_present}, "
    f"y_unique={y_unique}"
)

    # Drop leakage-prone features (post-event)
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])


    # Save processed dataset (make sure map exists)
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_FILE, index=False)

    print(f"Processed dataset saved to: {OUT_FILE}")
    print(f"Rows: {len(df)} | Columns: {df.shape[1]}")


if __name__ == "__main__":
    main()

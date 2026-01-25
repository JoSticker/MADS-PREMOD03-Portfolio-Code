import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "01_data" / "processed" / "dataset_clean.parquet"
MODEL_OUT = ROOT / "04_reports" / "model.joblib"

def main():
    df = pd.read_parquet(DATA)

    X = df.drop(columns=["y"])
    y = df["y"]

    cat_cols = X.select_dtypes(include="object").columns
    num_cols = X.select_dtypes(exclude="object").columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ])

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    model.fit(X_train_proc, y_train)

    joblib.dump((preprocessor, model), MODEL_OUT)

if __name__ == "__main__":
    main()
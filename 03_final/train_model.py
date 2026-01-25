import joblib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

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
    n_estimators=1000,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)


    model.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_test_proc, y_test)],
    verbose=False
    )

    y_pred_proba = model.predict_proba(X_test_proc)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    ap = average_precision_score(y_test, y_pred_proba)

    print(f"AUC (ROC): {auc:.3f}")
    print(f"Average Precision: {ap:.3f}")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    print(f"Recall at precision >= 0.35: {recall[precision >= 0.35].max():.3f}")

    joblib.dump((preprocessor, model), MODEL_OUT)

if __name__ == "__main__":
    main()
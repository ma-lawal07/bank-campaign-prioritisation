import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.evaluation import evaluate_ranking


ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "age", "job", "marital", "education",
    "default", "balance", "housing", "loan",
]

NUMERIC = ["age", "balance"]
CATEGORICAL = [name for name in FEATURES if name not in NUMERIC]


def make_pipeline(estimator):
    preprocessing = ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
    ])

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", estimator),
    ])


def main():
    data = pd.read_csv(
        ROOT / "data" / "raw" / "bank-full.csv",
        sep=";",
    )

    if len(data) != 45211:
        raise ValueError("Expected the original 45,211-record dataset.")

    X = data[FEATURES].copy()
    y = data["y"].map({"no": 0, "yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected target values.")

    train_end = int(len(data) * 0.60)
    valid_end = int(len(data) * 0.80)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_valid = X.iloc[train_end:valid_end]
    y_valid = y.iloc[train_end:valid_end]
    X_test, y_test = X.iloc[valid_end:], y.iloc[valid_end:]

    models = {
        "Logistic regression": make_pipeline(
            LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Random forest": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1,
            )
        ),
        "Gradient boosting": make_pipeline(
            GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=10,
                random_state=42,
            )
        ),
    }

    validation_results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        validation_results[name] = evaluate_ranking(
            y_valid,
            model.predict_proba(X_valid)[:, 1],
        )

        print(f"Trained: {name}")

    # Keep the model choice already made during notebook development.
    chosen_name = "Random forest"
    calibration_end = len(X_valid) // 2

    calibrated_model = CalibratedClassifierCV(
        estimator=FrozenEstimator(models[chosen_name]),
        method="sigmoid",
    )

    calibrated_model.fit(
        X_valid.iloc[:calibration_end],
        y_valid.iloc[:calibration_end],
    )

    # Preserve the original and calibrated candidates.
    # The calibrated candidate was selected during notebook development.
    final_model = calibrated_model
    final_model_name = "Calibrated random forest"

    reference_values = {
        feature: X_train[feature].median()
        for feature in NUMERIC
    }
    reference_values.update({
        feature: X_train[feature].mode().iloc[0]
        for feature in CATEGORICAL
    })

    test_models = {
        **models,
        final_model_name: final_model,
    }

    test_results = {
        name: evaluate_ranking(
            y_test,
            model.predict_proba(X_test)[:, 1],
        )
        for name, model in test_models.items()
    }

    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)

    joblib.dump(
        {
            "models": models,
            "final_model": final_model,
            "final_model_name": final_model_name,
            "features": FEATURES,
            "reference_values": reference_values,
        },
        artifact_dir / "campaign_models.joblib",
    )

    with (artifact_dir / "test_results.json").open("w") as file:
        json.dump(test_results, file, indent=2)

    with (artifact_dir / "validation_results.json").open("w") as file:
        json.dump(validation_results, file, indent=2)

    print(f"\nSaved artifacts to: {artifact_dir}")


if __name__ == "__main__":
    main()
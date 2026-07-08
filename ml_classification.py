"""
ml-classification

A simple machine learning classification template using
Scikit-learn and the Iris dataset.

Author: Verdiér
License: MIT
"""

import pandas as pd

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_data():
    """Load the Iris dataset."""

    iris = load_iris()

    X = pd.DataFrame(
        iris.data,
        columns=iris.feature_names,
    )

    y = pd.Series(
        iris.target,
        name="target",
    )

    return X, y, iris.target_names


def train_models(test_size=0.2, random_state=42):
    """Train Logistic Regression and Random Forest models."""

    X, y, target_names = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    logistic_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(max_iter=500),
            ),
        ]
    )

    logistic_pipeline.fit(X_train, y_train)

    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
    )

    rf_model.fit(X_train, y_train)

    logistic_predictions = logistic_pipeline.predict(X_test)
    rf_predictions = rf_model.predict(X_test)

    results = {
        "target_names": target_names,
        "logistic": {
            "model": logistic_pipeline,
            "accuracy": accuracy_score(
                y_test,
                logistic_predictions,
            ),
            "report": classification_report(
                y_test,
                logistic_predictions,
                target_names=target_names,
            ),
            "matrix": confusion_matrix(
                y_test,
                logistic_predictions,
            ),
        },
        "random_forest": {
            "model": rf_model,
            "accuracy": accuracy_score(
                y_test,
                rf_predictions,
            ),
            "report": classification_report(
                y_test,
                rf_predictions,
                target_names=target_names,
            ),
            "matrix": confusion_matrix(
                y_test,
                rf_predictions,
            ),
            "feature_importance": pd.Series(
                rf_model.feature_importances_,
                index=X.columns,
            ).sort_values(ascending=False),
        },
    }

    return results

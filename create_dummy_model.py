#!/usr/bin/env python3
"""Create and save a tiny dummy diabetes model for local testing.

This trains a simple LogisticRegression pipeline on synthetic data
and saves it to `diabetes_model.pkl` in the project root so `app.py`
can load it during development.
"""
import pickle
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def main():
    # Create synthetic binary classification data with 6 features
    X, y = make_classification(n_samples=1000, n_features=6, n_informative=4,
                               n_redundant=0, random_state=42)

    # Simple pipeline: scaling + logistic regression
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42))
    model.fit(X, y)

    # Save model to project root where app.py expects it
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print('Saved dummy model to diabetes_model.pkl')


if __name__ == '__main__':
    main()

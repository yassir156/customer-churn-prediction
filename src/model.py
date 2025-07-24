"""
Modeling module for the churn prediction project.

This module defines a :class:`ChurnModel` class that encapsulates the
definition, training and evaluation of machine learning models for churn
prediction. It currently supports logistic regression and random forest
classifiers. Additional models can be added following the same pattern.

Usage example::

    from model import ChurnModel
    model = ChurnModel(model_type='random_forest')
    model.train(X_train, y_train)
    metrics = model.evaluate(X_test, y_test)
    print(metrics)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@dataclass
class Metrics:
    """Container for classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float


class ChurnModel:
    """Machine learning model for churn prediction.

    Parameters
    ----------
    model_type : str, default='logistic_regression'
        Which model to use. Supported values are ``'logistic_regression'`` and
        ``'random_forest'``.
    model_params : dict, optional
        Dictionary of hyperparameters to pass to the underlying estimator.
    """

    def __init__(self, model_type: str = 'logistic_regression', model_params: Dict[str, Any] | None = None) -> None:
        self.model_type = model_type
        self.model_params = model_params or {}
        self.model = self._initialize_model()

    def _initialize_model(self) -> Any:
        """Instantiate the underlying scikit-learn estimator based on ``model_type``."""
        if self.model_type == 'logistic_regression':
            params = {'solver': 'lbfgs', 'max_iter': 1000, 'n_jobs': -1}
            params.update(self.model_params)
            return LogisticRegression(**params)
        elif self.model_type == 'random_forest':
            params = {'n_estimators': 200, 'random_state': 42, 'n_jobs': -1}
            params.update(self.model_params)
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """Fit the model to the training data."""
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Metrics:
        """Evaluate the trained model on the test set and compute metrics.

        Returns
        -------
        metrics : Metrics
            Container with accuracy, precision, recall and F1 scores.
        """
        y_pred = self.model.predict(X_test)
        metrics = Metrics(
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, zero_division=0),
            recall=recall_score(y_test, y_pred, zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
        )
        return metrics

    def get_model(self) -> Any:
        """Return the underlying scikit-learn estimator. Useful for inspection."""
        return self.model


__all__ = ["ChurnModel", "Metrics"]

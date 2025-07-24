"""
Preprocessing utilities for the churn project.

The :class:`Preprocessor` encapsulates all data transformation steps such as
splitting into train and test sets, encoding categorical variables and
scaling numerical features. By encapsulating transformations inside a class,
we keep the processing pipeline configurable and reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class SplitData:
    """Container for train/test splits.

    Attributes
    ----------
    X_train : np.ndarray
        Transformed training features.
    X_test : np.ndarray
        Transformed test features.
    y_train : np.ndarray
        Training target labels.
    y_test : np.ndarray
        Test target labels.
    preprocessor : ColumnTransformer
        Fitted transformer for later reuse on new data.
    feature_names : List[str]
        Names of generated features after one-hot encoding.
    """
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    preprocessor: ColumnTransformer
    feature_names: List[str]


class Preprocessor:
    """Preprocesses raw churn data for modelling.

    This class is responsible for splitting the dataset, identifying
    categorical and numerical columns, applying appropriate transformations
    (one-hot encoding for categoricals, scaling for numericals), and
    returning the transformed arrays alongside the fitted transformer.
    """

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def _build_transformer(self, X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str]]:
        """Build a column transformer for the given DataFrame.

        Parameters
        ----------
        X : pd.DataFrame
            Input features DataFrame.

        Returns
        -------
        transformer : ColumnTransformer
            A fitted transformer that applies one-hot encoding to categorical
            columns and scaling to numeric columns.
        feature_names : List[str]
            Names of the features after transformation (useful for model
            interpretation).
        """
        # Detect categorical and numerical columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

        # Pipeline for categorical features
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        # Pipeline for numerical features
        numeric_transformer = StandardScaler()

        # Create column transformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, categorical_cols),
                ('num', numeric_transformer, numeric_cols),
            ]
        )

        # Fit transformer to obtain feature names
        preprocessor.fit(X)
        # Extract feature names after transformation
        cat_features = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)
        feature_names = list(cat_features) + numeric_cols

        return preprocessor, feature_names

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> SplitData:
        """Split the dataset and apply transformations.

        Returns a :class:`SplitData` instance with transformed matrices and
        the fitted transformer for later use on new data.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        preprocessor, feature_names = self._build_transformer(X_train)
        X_train_processed = preprocessor.transform(X_train)
        X_test_processed = preprocessor.transform(X_test)
        return SplitData(
            X_train=X_train_processed,
            X_test=X_test_processed,
            y_train=y_train.to_numpy(),
            y_test=y_test.to_numpy(),
            preprocessor=preprocessor,
            feature_names=feature_names
        )


__all__ = ["Preprocessor", "SplitData"]

"""
Data loading module for the synthetic churn project.

This module exposes a ``DataLoader`` class which is responsible for reading
the CSV dataset into a pandas DataFrame and splitting it into feature and
target components. Keeping data ingestion in its own class makes it easy
to swap out different data sources or add additional processing at the
ingestion stage without affecting downstream code.

Example::

    from data_loader import DataLoader
    loader = DataLoader('data/churn_data.csv')
    X, y = loader.load_data()

"""

from __future__ import annotations

import pandas as pd
from pathlib import Path


class DataLoader:
    """Loads a churn dataset from a CSV file.

    Parameters
    ----------
    file_path : str | Path
        Path to the CSV file containing the dataset. The CSV must include
        a target column named ``churn`` along with the feature columns.
    """

    def __init__(self, file_path: str | Path) -> None:
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.file_path}")

    def load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load the dataset and return features and target.

        Returns
        -------
        X : pandas.DataFrame
            The input features.
        y : pandas.Series
            The target variable indicating churn (1) or not (0).

        Notes
        -----
        The loader does not perform any cleaning or preprocessing; it simply
        reads the file and splits the columns. Preprocessing should be
        handled by the :class:`Preprocessor` class.
        """
        df = pd.read_csv(self.file_path)
        if 'churn' not in df.columns:
            raise ValueError("Expected a target column named 'churn' in the dataset")
        y = df['churn']
        X = df.drop(columns=['churn'])
        return X, y


__all__ = ["DataLoader"]

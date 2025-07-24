"""
Training script for the churn prediction project.

This script ties together the data loading, preprocessing and model training
components. When run as a script, it loads the data from the ``data``
directory, splits the data into training and test sets, applies
transformations, trains a selected model and outputs evaluation metrics.

The script can be executed from the command line::

    python train.py --data-path ../data/churn_data.csv --model random_forest

Optionally, the trained model and preprocessor can be persisted to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import joblib  # type: ignore

from data_loader import DataLoader
from preprocessor import Preprocessor
from model import ChurnModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a churn prediction model.")
    parser.add_argument(
        '--data-path', type=str, default='../data/churn_data.csv',
        help='Path to the CSV data file relative to this script.',
    )
    parser.add_argument(
        '--model', type=str, default='random_forest', choices=['logistic_regression', 'random_forest'],
        help='Which model to train.',
    )
    parser.add_argument(
        '--save-model', action='store_true',
        help='If set, the trained model and preprocessor will be saved alongside metrics.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path)

    # Load data
    loader = DataLoader(data_path)
    X, y = loader.load_data()

    # Preprocess data
    preprocessor = Preprocessor()
    split_data = preprocessor.fit_transform(X, y)

    # Train model
    model = ChurnModel(model_type=args.model)
    model.train(split_data.X_train, split_data.y_train)

    # Evaluate model
    metrics = model.evaluate(split_data.X_test, split_data.y_test)
    print(f"Model type: {args.model}")
    print(f"Accuracy: {metrics.accuracy:.4f}")
    print(f"Precision: {metrics.precision:.4f}")
    print(f"Recall: {metrics.recall:.4f}")
    print(f"F1 score: {metrics.f1:.4f}")

    # Save model and preprocessor if requested
    if args.save_model:
        model_dir = Path(__file__).resolve().parent / 'artifacts'
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.get_model(), model_dir / 'model.joblib')
        joblib.dump(split_data.preprocessor, model_dir / 'preprocessor.joblib')
        print(f"Saved model and preprocessor to {model_dir}")


if __name__ == '__main__':
    main()

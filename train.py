import argparse

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)


def plot_dataset(X: np.ndarray, y: np.ndarray, filename: str) -> None:
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title('dataset')
    plt.savefig(filename)


def plot_decision_plane(X: np.ndarray, y: np.ndarray,
                        model: LogisticRegression, filename: str) -> None:
    b = model.intercept_[0]
    w1, w2 = model.coef_.T
    c = -b / w2
    m = -w1 / w2

    xmin, xmax = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
    ymin, ymax = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
    xd = np.array([xmin, xmax])
    yd = m * xd + c
    plt.plot(xd, yd, 'k', lw=1, ls='--')
    plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.2)
    plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.2)

    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.title('decision plane')
    plt.savefig(filename)


def save_metrics(X: np.ndarray, y: np.ndarray,
                 model: LogisticRegression, filename: str) -> None:
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)

    with open(filename, 'w') as file:
        file.write(f'accuracy: {accuracy}\n')
        file.write(f'precision: {precision}\n')
        file.write(f'recall: {recall}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--plot_dataset', type=str, default=None)
    parser.add_argument('--plot_decision_plane', type=str, default=None)
    parser.add_argument('--save_metrics', type=str, default=None)
    args = parser.parse_args()

    X, y = make_blobs(
            n_samples=1000,
            n_features=2,
            centers=2,
            random_state=args.random_seed,
        )

    X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=0.9, random_state=args.random_seed)

    if args.plot_dataset:
        plot_dataset(X_train, y_train, args.plot_dataset)

    model = LogisticRegression()
    model.fit(X, y)

    if args.plot_decision_plane:
        plot_decision_plane(X_train, y_train, model, args.plot_decision_plane)

    if args.save_metrics:
        save_metrics(X_test, y_test, model, args.save_metrics)


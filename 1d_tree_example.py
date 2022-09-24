import itertools
import math
import time
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from animate import rotanimate

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)

GIF_DIR = Path("gifs")
GIF_DIR.mkdir(exist_ok=True)


def remove_spines(ax) -> None:
    ax.spines.right.set_visible(False)

    ax.spines.top.set_visible(False)
    plt.tick_params(labelleft=False, left=False)
    plt.tick_params(labelbottom=False, bottom=False)


class GreedyTreeRegressor:
    def __init__(self):
        pass

    @staticmethod
    def split_node(feat_vals: np.ndarray, y: np.ndarray) -> float:
        assert feat_vals.size == y.size

        best_squared_error = np.inf
        best_split = feat_vals[0]

        for split_point in feat_vals:
            left_index_set = feat_vals <= split_point
            right_index_set = feat_vals > split_point

            y_left = y[left_index_set]
            y_right = y[right_index_set]

            if y_left.size == 0 or y_right.size == 0:
                continue

            y_pred_left = np.mean(y_left)
            y_pred_right = np.mean(y_right)

            squared_error = np.square(y_left - y_pred_left).sum() + np.square(y_right - y_pred_right).sum()

            if squared_error < best_squared_error:
                best_squared_error = squared_error
                best_split = split_point

        return best_split

    def fit(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        assert X.ndim == 2
        assert y.ndim == 1

        feat_vals = X.ravel()
        best_node_0_split = self.split_node(feat_vals, y)
        best_node_1_split = self.split_node(
            feat_vals[feat_vals <= best_node_0_split], y[feat_vals <= best_node_0_split]
        )

        self._node_0_split = best_node_0_split
        self._node_1_split = best_node_1_split

        self._y_pred_2 = y[feat_vals > best_node_0_split].mean()
        self._y_pred_3 = y[(feat_vals <= best_node_0_split) & (feat_vals <= best_node_1_split)].mean()
        self._y_pred_4 = y[(feat_vals <= best_node_0_split) & (feat_vals > best_node_1_split)].mean()

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat_vals = X.ravel()
        node_2_index_set = feat_vals > self._node_0_split
        node_3_index_set = (feat_vals <= self._node_0_split) & (feat_vals <= self._node_1_split)
        node_4_index_set = (feat_vals <= self._node_0_split) & (feat_vals > self._node_1_split)

        preds = np.zeros_like(feat_vals)
        preds[node_2_index_set] = self._y_pred_2
        preds[node_3_index_set] = self._y_pred_3
        preds[node_4_index_set] = self._y_pred_4

        return preds


class ExactTreeRegressor:
    def __init__(self):
        pass

    def fit(self, X: np.ndarray, y: np.ndarray):
        assert X.ndim == 2
        assert y.ndim == 1

        feat_vals = X.ravel()
        node_0_splits = []
        node_1_splits = []
        losses = []

        start = time.perf_counter()

        for i, (node_0_split, node_1_split) in enumerate(itertools.product(feat_vals, feat_vals.copy())):
            node_2_index_set = feat_vals > node_0_split
            node_3_index_set = (feat_vals <= node_0_split) & (feat_vals <= node_1_split)
            node_4_index_set = (feat_vals <= node_0_split) & (feat_vals > node_1_split)

            y_node_2 = y[node_2_index_set]
            y_node_3 = y[node_3_index_set]
            y_node_4 = y[node_4_index_set]

            if y_node_2.size == 0 or y_node_3.size == 0 or y_node_4.size == 0:
                continue

            y_pred_2 = y_node_2.mean()
            y_pred_3 = y_node_3.mean()
            y_pred_4 = y_node_4.mean()

            total_sq_error = (
                np.square(y_node_2 - y_pred_2).sum()
                + np.square(y_node_3 - y_pred_3).sum()
                + np.square(y_node_4 - y_pred_4).sum()
            )

            node_0_splits.append(node_0_split)
            node_1_splits.append(node_1_split)
            losses.append(total_sq_error)

        argmin_idx = np.argmin(losses)
        self._node_0_split = node_0_splits[argmin_idx]
        self._node_1_split = node_1_splits[argmin_idx]

        self._y_pred_2 = y[feat_vals > self._node_0_split].mean()
        self._y_pred_3 = y[(feat_vals <= self._node_0_split) & (feat_vals <= self._node_1_split)].mean()
        self._y_pred_4 = y[(feat_vals <= self._node_0_split) & (feat_vals > self._node_1_split)].mean()

        return node_0_splits, node_1_splits, losses

    def predict(self, X: np.ndarray) -> np.ndarray:
        feat_vals = X.ravel()
        node_2_index_set = feat_vals > self._node_0_split
        node_3_index_set = (feat_vals <= self._node_0_split) & (feat_vals <= self._node_1_split)
        node_4_index_set = (feat_vals <= self._node_0_split) & (feat_vals > self._node_1_split)

        preds = np.zeros_like(feat_vals)
        preds[node_2_index_set] = self._y_pred_2
        preds[node_3_index_set] = self._y_pred_3
        preds[node_4_index_set] = self._y_pred_4

        return preds


def make_1d_data(n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.random.rand(n_samples, 1)
    y = 1 / np.pi * (np.sin(math.tau * x) - 1 / 3 * np.sin(math.tau * 3 * x))
    return x, y.ravel()


def plot_exact_tree_fit_loss_surface(x: np.ndarray, y: np.ndarray) -> None:
    model_1 = ExactTreeRegressor()

    x1, x2, x3 = model_1.fit(x, y)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_trisurf(x1, x2, x3, cmap=cm.coolwarm, antialiased=False, alpha=0.9)
    plt.tight_layout()
    angles = np.linspace(0, 360, 120)[:-1]

    # create an animated gif (5ms between frames)
    rotanimate(ax, angles, GIF_DIR / "tree_loss_surface.gif", delay=7)


def plot_exact_versus_greedy_fit(x: np.ndarray, y: np.ndarray) -> None:
    model_1 = ExactTreeRegressor()
    model_1.fit(x, y)

    model_2 = GreedyTreeRegressor()
    model_2.fit(x, y)

    x_vals = np.linspace(0, 1, 100)
    plt.figure()
    plt.scatter(x, y, label="Train data")
    plt.plot(x_vals, model_1.predict(x_vals), label="ExactTree", color="tab:orange")
    plt.plot(x_vals, model_2.predict(x_vals), label="GreedyTree", color="tab:purple")

    plt.xlabel(r"$x$", fontsize=14, color="w")
    plt.ylabel(r"$y$", fontsize=14, color="w")
    ax = plt.gca()
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    remove_spines(plt.gca())

    leg = plt.legend(frameon=False)
    for text in leg.get_texts():
        text.set_color("w")
    plt.savefig(IMAGE_DIR / "brute_force_fit.png", transparent=True)

    exact_mse = np.mean(np.square(y - model_1.predict(x)))
    greedy_mse = np.mean(np.square(y - model_2.predict(x)))
    print(f"Exact tree mse: {exact_mse:.4}")
    print(f"Greedy tree mse {greedy_mse:.4}")
    print(f"Exact pct improvement over greedy: {(greedy_mse /  exact_mse - 1)*100:.4}%")


def main() -> None:
    n_samples = 150
    np.random.seed(31415)
    x, y = make_1d_data(n_samples)

    plot_exact_tree_fit_loss_surface(x, y)
    # plot_exact_versus_greedy_fit(x, y)


if __name__ == "__main__":
    main()

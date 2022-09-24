from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import remove_all_axes, remove_spines

IMAGE_DIR = Path("images")
IMAGE_DIR.mkdir(exist_ok=True)


def make_data(n: int) -> np.ndarray:
    r = 3 * np.random.rand(n)
    theta = 2 * np.pi * np.random.rand(n)

    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.sin(r) / r

    return np.stack([x, y], axis=1), z


def quantise(X, n_bins) -> Tuple[np.ndarray, np.ndarray]:
    n, d = X.shape
    X_q = np.zeros((n, d), dtype=np.int32)
    quantisation_arr = np.zeros((n_bins - 1, d), dtype=np.float32)

    for j in range(d):
        x_feat = X[:, j]
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.quantile(x_feat, quantiles[1:-1])
        quantisation_arr[:, j] = bin_edges
        X_q[:, j] = np.digitize(x_feat, bin_edges)

    return X_q, quantisation_arr


def depict_quantisation_algo(X, n_bins=8):
    n, d = X.shape
    X_q, bin_edges = quantise(X, n_bins)

    plt.figure()
    for j in range(d):
        x_feat = X[:, j]
        plt.step(np.sort(x_feat), 1 / n * np.cumsum(np.ones(n)), label="Unquantised")
        x_vals = [np.min(x_feat)] + list(bin_edges[:, j]) + [np.max(x_feat)]
        y_vals = []
        for x in x_vals:
            y_vals.append(1 / n * np.sum(x_feat <= x))

        text_color = "w"
        plt.step(x_vals, y_vals, alpha=0.7, label="Quantised")
        ax = plt.gca()
        ax.spines["left"].set_color("white")
        ax.spines["bottom"].set_color("white")
        remove_spines(ax)

        plt.xlabel("Feature value", color=text_color)
        plt.ylabel("Empirical CDF", color=text_color)
        leg = plt.legend(frameon=False)
        for text in leg.get_texts():
            text.set_color(text_color)
        plt.tight_layout()
        plt.savefig(IMAGE_DIR / f"quantisation_level_{n_bins}.png", transparent=True)
        break


def simulate_gradients_and_hessians(n_bins: int, node_id: int, feat_id: int) -> Tuple[np.ndarray, np.ndarray]:
    np.random.seed(int(f"{node_id}{feat_id}") + 3141)
    bin_ids = np.arange(n_bins)
    mu = np.zeros(n_bins)
    sigma = 15
    grad_covariance = np.exp(-((bin_ids - bin_ids[:, np.newaxis]) ** 2) / sigma)

    gradients = np.random.multivariate_normal(mu, grad_covariance)

    sigma = 5
    mu = np.ones(n_bins)
    hess_covariance = np.exp(-((bin_ids - bin_ids[:, np.newaxis]) ** 2) / sigma)
    hessians = np.abs(np.random.multivariate_normal(mu, hess_covariance))

    return gradients, hessians


def compute_split_quality_scores(gradients: np.ndarray, hessians: np.ndarray) -> np.ndarray:
    cumulative_gradients = np.cumsum(gradients)
    cumulative_hessians = np.cumsum(hessians)

    lam = 0
    g_total = cumulative_gradients[-1]
    h_total = cumulative_hessians[-1]

    # n bins means n - 1 split points
    g_left = cumulative_gradients[:-1]
    h_left = cumulative_hessians[:-1]

    g_right = g_total - g_left
    h_right = h_total - h_left

    q_left = g_left**2 / (h_left + lam)
    q_right = g_right**2 / (h_right + lam)

    q_total = g_total**2 / (h_total + lam)
    q_scores = q_left + q_right - q_total

    return q_scores


def depict_cumulative_gradients_and_hessians(gradients: np.ndarray, hessians: np.ndarray, node_id: int, feat_id: int):
    cumulative_gradients = np.cumsum(gradients)
    cumulative_hessians = np.cumsum(hessians)

    suffix = f"{feat_id}_{node_id}"
    stem_kwargs = {"basefmt": "gray", "linefmt": "w", "markerfmt": "wo"}
    text_color = "w"

    plt.figure()
    plt.stem(gradients, **stem_kwargs)
    plt.xlabel("Bin Id", fontsize=14, color="w")
    plt.title("grad", fontsize=14, family="monospace", color=text_color)
    remove_all_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"grad_node_{suffix}.png", transparent=True)

    plt.figure()
    plt.stem(cumulative_gradients, **stem_kwargs)
    plt.xlabel("Bin Id", fontsize=14, color=text_color)
    plt.title(r"$G_L = \sum_{k=0}^i$grad[k]", fontsize=14, family="monospace", color=text_color)
    remove_all_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"cumulative_grad_node_{suffix}.png", transparent=True)

    plt.figure()
    plt.stem(hessians, **stem_kwargs)
    plt.xlabel("Bin Id", fontsize=14, color=text_color)
    plt.title("hess", fontsize=14, family="monospace", color=text_color)
    remove_all_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"hess_node_{suffix}.png", transparent=True)

    plt.figure()
    plt.stem(cumulative_hessians, **stem_kwargs)
    plt.xlabel("Bin Id", fontsize=14, color=text_color)
    plt.title(r"$H_L = \sum_{k=0}^i$hess[k]", fontsize=14, family="monospace", color=text_color)
    remove_all_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"cumulative_hess_node_{suffix}.png", transparent=True)


def depict_quality_scores(q_scores: np.ndarray, node_id: int, feat_id: int) -> None:
    suffix = f"{feat_id}_{node_id}"
    text_color = "w"
    stem_kwargs = {"basefmt": "gray", "linefmt": "w", "markerfmt": "wo"}

    plt.figure()
    markers, stemlines, baseline = plt.stem(q_scores, **stem_kwargs)
    plt.xlabel("Bin Id", fontsize=14, color=text_color)
    plt.ylabel("Quality Score", fontsize=14, color=text_color)
    remove_all_axes(plt.gca())
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"q_score_node_{suffix}.png", transparent=True)

    plt.figure()
    i_max = np.argmax(q_scores)

    n_bins = len(q_scores)
    non_max_q_scores = np.append(q_scores[:i_max], q_scores[i_max + 1 :])
    markers, stemlines, baseline = plt.stem([i for i in range(n_bins) if i != i_max], non_max_q_scores, **stem_kwargs)
    plt.setp(markers, color="white", linewidth=0.5, alpha=0.5)
    plt.setp(stemlines, linestyle="-", color="white", linewidth=0.5, alpha=0.3)

    plt.xlabel("Bin Id", fontsize=14, color=text_color)
    plt.ylabel("Quality Score", fontsize=14, color=text_color)
    remove_all_axes(plt.gca())
    plt.stem([i_max], [q_scores[i_max]], **stem_kwargs)
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / f"q_score_best_{suffix}.png", transparent=True)


def main():
    n = 100
    d = 2
    X, y = make_data(n)
    for i in [4, 8, 16, 32]:
        depict_quantisation_algo(X, i)

    # feat_id = 0
    # for node_id in range(3, 7):
    #     gradients, hessians = simulate_gradients_and_hessians(32, node_id, feat_id)
    #     depict_cumulative_gradients_and_hessians(gradients, hessians, node_id=node_id, feat_id=feat_id)
    #     q_scores = compute_split_quality_scores(gradients, hessians)
    #     depict_quality_scores(q_scores, node_id, feat_id)


if __name__ == "__main__":
    main()

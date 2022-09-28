import matplotlib.pyplot as plt

from tree_algo import IMAGE_DIR


def main():
    plt.axhline(y=0.5, color="k", linestyle="-", linewidth=4, xmin=0, xmax=1)
    plt.axvline(x=0.25, color="k", linestyle="-", linewidth=4, ymin=0, ymax=0.75)
    plt.fill_between([-1, 1], y1=0.5, y2=1, color="tab:blue", alpha=0.7)
    plt.fill_between([-1, 0.25], y1=-1, y2=0.5, color="turquoise", alpha=0.7)
    plt.fill_between([0.25, 1], y1=-1, y2=0.5, color="dodgerblue", alpha=0.7)

    plt.xlim([-1, 1])
    plt.ylim([-1, 1])

    ax = plt.gca()
    ax.spines["top"].set_color("none")
    ax.spines["left"].set_position("zero")
    ax.spines["right"].set_color("none")
    ax.spines["bottom"].set_position("zero")
    ax.spines["left"].set_color("white")
    ax.spines["bottom"].set_color("white")
    # ax.set_aspect("equal", adjustable="box")

    plt.tick_params(labelleft=False, left=True, color="white")
    plt.tick_params(labelbottom=False, bottom=True, direction="inout")
    plt.tight_layout()
    # plt.grid(
    plt.grid(linestyle="--", color="lightgray", alpha=0.5)
    plt.savefig(IMAGE_DIR / "regions.png", transparent=True)


if __name__ == "__main__":
    main()

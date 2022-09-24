import matplotlib.pyplot as plt


def remove_spines(ax) -> None:
    ax.spines.right.set_visible(False)

    ax.spines.top.set_visible(False)
    plt.tick_params(labelleft=False, left=False)
    plt.tick_params(labelbottom=False, bottom=False)


def remove_all_axes(ax):
    ax.spines.right.set_visible(False)
    ax.spines.left.set_visible(False)

    ax.spines.top.set_visible(False)
    ax.spines.bottom.set_visible(False)
    plt.tick_params(labelleft=False, left=False)
    plt.tick_params(labelbottom=False, bottom=False)

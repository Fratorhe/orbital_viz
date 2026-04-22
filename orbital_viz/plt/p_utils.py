from matplotlib.lines import Line2D


def add_legend_entry(
    ax,
    label,
    color="C0",
    linewidth=2,
    linestyle="-",
):
    """
    Add a single legend entry to the axis, avoiding duplicates.

    This uses a persistent storage on the axis so multiple calls
    accumulate entries cleanly.
    """

    if label is None:
        return

    # Initialize storage if needed
    if not hasattr(ax, "_orbit_legend_handles"):
        ax._orbit_legend_handles = []
        ax._orbit_legend_labels = []

    # Avoid duplicates
    if label in ax._orbit_legend_labels:
        return

    handle = Line2D(
        [0],
        [0],
        color=color,
        lw=linewidth,
        linestyle=linestyle,
        label=label,
    )

    ax._orbit_legend_handles.append(handle)
    ax._orbit_legend_labels.append(label)

    ax.legend(ax._orbit_legend_handles, ax._orbit_legend_labels)

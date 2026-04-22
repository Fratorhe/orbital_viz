import matplotlib.pyplot as plt


def setup_axis(view="3D", figsize=(7, 7), lim=None):
    """
    Create a 3D axis. If view='2D', use a top-down camera view.

    Parameters
    ----------
    view : str
        "3D" or "2D"
    figsize : tuple
        Figure size
    lim : float or None
        Symmetric axis limit in km

    Returns
    -------
    fig, ax
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")
    ax.set_zlabel("z [km]")

    if lim is not None:
        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_zlim([-lim, lim])

    ax.set_box_aspect([1, 1, 1])
    ax.grid(True)

    set_view(ax, view=view)

    return fig, ax


def set_view(ax, view="3D", hide_z_in_2d=True):
    """
    Set the camera view of an existing 3D axis.
    """
    view = view.upper()

    if view == "3D":
        ax.view_init(elev=25, azim=45)

        # Ensure z-axis is visible again
        ax.zaxis.set_visible(True)

    elif view == "2D":
        ax.view_init(elev=90, azim=-90)

        if hide_z_in_2d:
            ax.zaxis.set_visible(False)
            ax.set_proj_type("ortho")

            # Optional cleanup for cleaner 2D look
            ax.set_zticks([])
            ax.set_zlabel("")
            ax.zaxis.line.set_lw(0)

    else:
        raise ValueError("view must be '3D' or '2D'")

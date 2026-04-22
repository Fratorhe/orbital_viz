import numpy as np

BODY_COLORS = {
    "sun": "gold",
    "mercury": "darkgray",
    "venus": "goldenrod",
    "earth": "royalblue",
    "moon": "lightgray",
    "mars": "orangered",
    "jupiter": "tan",
    "saturn": "khaki",
    "uranus": "paleturquoise",
    "neptune": "deepskyblue",
}

BODY_RADII_KM = {
    "sun": 696340.0,
    "mercury": 2439.7,
    "venus": 6051.8,
    "earth": 6378.137,
    "moon": 1737.4,
    "mars": 3389.5,
    "jupiter": 71492.0,
    "saturn": 60268.0,
    "uranus": 25559.0,
    "neptune": 24764.0,
}


def _estimate_axis_scale(ax):
    """
    Estimate characteristic length scale from axis limits.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    zlim = ax.get_zlim()

    dx = abs(xlim[1] - xlim[0])
    dy = abs(ylim[1] - ylim[0])
    dz = abs(zlim[1] - zlim[0])

    return max(dx, dy, dz)


def plot_central_body(
    ax,
    body=None,
    radius=None,
    color=None,
    alpha=0.6,
    scale_factor=8000,  # controls visual size
    min_size=50,
    max_size=5000,
    center=(0.0, 0.0, 0.0),
    **kwargs,
):
    """
    Plot central body as a marker with size scaled to orbit size.

    Parameters
    ----------
    ax : 3D axis
    body : str or None
    radius : float or None (km)
    color : str or None
    alpha : float
    scale_factor : float
        Controls overall visual scaling
    min_size, max_size : float
        Clamp marker size
    center : tuple
    """

    # --- Resolve body ---
    body_key = body.strip().lower() if body is not None else None

    if radius is None:
        if body_key in BODY_RADII_KM:
            radius = BODY_RADII_KM[body_key]
        else:
            raise ValueError("Provide radius if body is unknown")

    if color is None:
        color = BODY_COLORS.get(body_key, "gray")

    # --- Estimate axis scale ---
    L = _estimate_axis_scale(ax)

    # Avoid division by zero
    if L <= 0:
        L = radius * 10

    # --- Scale marker size ---
    ratio = radius / L
    size = scale_factor * (ratio**2)

    # Clamp for visibility
    size = np.clip(size, min_size, max_size)

    # --- Plot ---
    x, y, z = center

    point = ax.scatter(
        [x],
        [y],
        [z],
        s=size,
        color=color,
        alpha=alpha,
        edgecolors="k",
        linewidths=0.5,
        **kwargs,
    )

    return point

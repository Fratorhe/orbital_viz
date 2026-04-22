import numpy as np
import plotly.graph_objects as go

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


def plot_central_body(
    fig,
    body=None,
    radius=None,
    color=None,
    alpha=0.6,
    scene_scale=None,
    scale_factor=120,
    min_size=8,
    max_size=40,
    center=(0.0, 0.0, 0.0),
    name=None,
    showlegend=False,
    **kwargs,
):
    """
    Plot central body as a marker in Plotly.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
    body : str or None
    radius : float or None
        Physical radius in km.
    color : str or None
    alpha : float
    scene_scale : float or None
        Characteristic scale of the scene in km, typically the orbit size
        or plot limit. Required for sensible marker scaling.
    scale_factor : float
        Controls visual scaling of the marker.
    min_size, max_size : float
        Clamp marker size.
    center : tuple
        Center position.
    name : str or None
        Trace name.
    showlegend : bool
        Whether to show in legend.
    """

    body_key = body.strip().lower() if body is not None else None

    if radius is None:
        if body_key in BODY_RADII_KM:
            radius = BODY_RADII_KM[body_key]
        else:
            raise ValueError("Provide radius if body is unknown")

    if color is None:
        color = BODY_COLORS.get(body_key, "gray")

    if scene_scale is None or scene_scale <= 0:
        scene_scale = radius * 10.0

    ratio = radius / scene_scale

    # Plotly marker size is in pixels, so use linear-ish scaling
    size = scale_factor * np.sqrt(ratio)
    size = float(np.clip(size, min_size, max_size))

    x, y, z = center

    trace = go.Scatter3d(
        x=[x],
        y=[y],
        z=[z],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            opacity=alpha,
            line=dict(color="black", width=1),
        ),
        name=name if name is not None else (body.capitalize() if body else "Body"),
        showlegend=showlegend,
        hoverinfo="skip",
        **kwargs,
    )

    fig.add_trace(trace)
    return trace

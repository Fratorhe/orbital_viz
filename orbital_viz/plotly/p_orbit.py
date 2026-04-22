import numpy as np
import plotly.graph_objects as go

_MPL_TO_PLOTLY_DASH = {
    "-": "solid",
    "--": "dash",
    "-.": "dashdot",
    ":": "dot",
}


def plot_orbit(
    fig,
    orbit_state,
    n_points=500,
    theta_range=None,
    alpha=0.7,
    ls="--",
    show_apses=False,
    show_direction=True,
    **kwargs,
):
    """
    Plot an orbit in 3D using Plotly.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure where traces will be added.
    orbit_state : OrbitalState
        Must contain at least a, e, i, Omega, omega.
    n_points : int
        Number of sampled points along the orbit.
    theta_range : tuple(float, float) or None
        Range of true anomaly to plot.
    alpha : float
        Opacity of orbit-related traces.
    ls : str
        Matplotlib-like linestyle: "-", "--", "-.", or ":"
    show_apses : bool
        Whether to plot the line of apses.
    show_direction : bool
        Whether to plot a short direction indicator.
    **kwargs
        Supported keys:
            color
            linewidth

    Returns
    -------
    dict
        Dictionary of Plotly trace objects added to the figure.
    """

    required = ["a", "e", "i", "Omega", "omega"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        raise ValueError(f"Missing orbital elements for plotting: {missing}")

    e = orbit_state.e
    a = orbit_state.a
    assert e is not None
    assert a is not None

    if theta_range is None:
        if e < 1:
            theta_vals = np.linspace(0.0, 2.0 * np.pi, n_points)
        else:
            theta_lim = np.arccos(-1.0 / e) - 1e-3
            theta_vals = np.linspace(-theta_lim, theta_lim, n_points)
    else:
        theta_vals = np.linspace(theta_range[0], theta_range[1], n_points)

    r_points = []
    for theta in theta_vals:
        temp_state = orbit_state.copy()
        temp_state.theta = theta
        temp_state.compute_state_vectors()
        r_points.append(temp_state.r_vec)

    r_points = np.asarray(r_points)

    x = r_points[:, 0]
    y = r_points[:, 1]
    z = r_points[:, 2]

    orbit_color = kwargs.get("color", None)
    orbit_lw = kwargs.get("linewidth", 2)
    dash_style = _MPL_TO_PLOTLY_DASH.get(ls, "solid")

    artists = {}

    # --- Orbit ---
    orbit_trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode="lines",
        line=dict(
            color=orbit_color,
            width=orbit_lw,
            dash=dash_style,
        ),
        opacity=alpha,
        showlegend=False,
        hoverinfo="skip",
    )
    fig.add_trace(orbit_trace)
    artists["orbit"] = orbit_trace

    # --- Line of apses ---
    if show_apses:
        apses_points = orbit_state.get_apses_line_points()
        print(apses_points)

        if apses_points is not None:
            p1, p2 = apses_points

            apses_trace = go.Scatter3d(
                x=[p1[0], p2[0]],
                y=[p1[1], p2[1]],
                z=[p1[2], p2[2]],
                mode="lines",
                line=dict(
                    color=orbit_color,
                    width=orbit_lw,
                    dash="dashdot",
                ),
                opacity=alpha * 0.7,
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_trace(apses_trace)
            artists["apses"] = apses_trace

    # --- Direction indicator ---
    if show_direction and n_points >= 2:
        idx = int(0.6 * (n_points - 2))

        p0 = r_points[idx]
        p1 = r_points[idx + 1]

        d = p1 - p0
        d_norm = np.linalg.norm(d)

        if d_norm > 1e-12:
            d_hat = d / d_norm
            orbit_scale = np.max(np.linalg.norm(r_points, axis=1))
            arrow_length = 0.1 * orbit_scale
            p_tip = p0 + arrow_length * d_hat

            # short line
            dir_line = go.Scatter3d(
                x=[p0[0], p_tip[0]],
                y=[p0[1], p_tip[1]],
                z=[p0[2], p_tip[2]],
                mode="lines",
                line=dict(
                    color=orbit_color,
                    width=orbit_lw + 1,
                ),
                opacity=min(1.0, alpha + 0.1),
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_trace(dir_line)

            # tip marker
            dir_tip = go.Scatter3d(
                x=[p_tip[0]],
                y=[p_tip[1]],
                z=[p_tip[2]],
                mode="markers",
                marker=dict(
                    size=4,
                    color=orbit_color,
                    opacity=min(1.0, alpha + 0.1),
                ),
                showlegend=False,
                hoverinfo="skip",
            )
            fig.add_trace(dir_tip)

            artists["direction"] = (dir_line, dir_tip)

    return artists

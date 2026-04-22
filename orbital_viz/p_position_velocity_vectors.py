import numpy as np

from orbital_viz.orbital_state import OrbitalState


def plot_position(ax, orbit_state, color="red", size=40, label=None, **kwargs):
    """
    Plot the current spacecraft position if theta is available.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis where the position will be plotted.
    orbit_state : OrbitalState
        Must contain a, e, i, Omega, omega, theta
    color : str, optional
        Marker color.
    size : float, optional
        Marker size.
    label : str or None, optional
        Optional label for legend.
    **kwargs
        Extra keyword arguments passed to ax.scatter().

    Returns
    -------
    point
        Output from ax.scatter(...)
    """
    required = ["a", "e", "i", "Omega", "omega", "theta"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        return None

    orbit_state.compute_state_vectors()
    r_vec = np.asarray(orbit_state.r_vec, dtype=float)

    point = ax.scatter(
        [r_vec[0]],
        [r_vec[1]],
        [r_vec[2]],
        s=size,
        color=color,
        label=label,
        **kwargs,
    )

    return point


def plot_position_vector(
    ax,
    orbit_state,
    color="C1",
    linewidth=2,
    label=None,
    **kwargs,
):
    """
    Plot the position vector from the origin to the current spacecraft position.
    """
    required = ["a", "e", "i", "Omega", "omega", "theta"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        return None

    orbit_state.compute_state_vectors()
    r_vec = np.asarray(orbit_state.r_vec, dtype=float)

    line = ax.plot(
        [0.0, r_vec[0]],
        [0.0, r_vec[1]],
        [0.0, r_vec[2]],
        color=color,
        linewidth=linewidth,
        label=label,
        **kwargs,
    )

    return line


def plot_velocity_vector(
    ax,
    orbit_state,
    scale=500.0,
    color="C2",
    linewidth=2,
    normalize=False,
    label=None,
    **kwargs,
):
    """
    Plot the velocity vector at the current spacecraft position.

    Parameters
    ----------
    scale : float
        Multiplies the velocity vector for visualization.
    normalize : bool
        If True, only the direction is used and the vector is scaled to `scale`.
    """
    required = ["a", "e", "i", "Omega", "omega", "theta"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        return None

    orbit_state.compute_state_vectors()
    r_vec = np.asarray(orbit_state.r_vec, dtype=float)
    v_vec = np.asarray(orbit_state.v_vec, dtype=float)

    if normalize:
        v_norm = np.linalg.norm(v_vec)
        if v_norm > 0:
            v_plot = scale * v_vec / v_norm
        else:
            v_plot = np.zeros(3)
    else:
        v_plot = scale * v_vec

    q = ax.quiver(
        r_vec[0],
        r_vec[1],
        r_vec[2],
        v_plot[0],
        v_plot[1],
        v_plot[2],
        color=color,
        linewidth=linewidth,
        label=label,
        **kwargs,
    )

    return q


def plot_current_state(
    ax,
    orbit_state,
    show_position=True,
    show_r=True,
    show_v=True,
    position_kwargs=None,
    r_kwargs=None,
    v_kwargs=None,
):
    """
    Plot current spacecraft marker and optional position/velocity vectors.
    """
    position_kwargs = {} if position_kwargs is None else position_kwargs
    r_kwargs = {} if r_kwargs is None else r_kwargs
    v_kwargs = {} if v_kwargs is None else v_kwargs

    artists = {}

    if show_position:
        artists["position"] = plot_position(ax, orbit_state, **position_kwargs)

    if show_r:
        artists["r_vec"] = plot_position_vector(ax, orbit_state, **r_kwargs)

    if show_v:
        artists["v_vec"] = plot_velocity_vector(ax, orbit_state, **v_kwargs)

    return artists


def plot_velocity_components(
    ax,
    orbit_state,
    scale=500.0,
    color_vr="C0",
    color_vt="C1",
    ls_vr=":",
    ls_vt=":",
    as_sum=False,
    linewidth=2,
    **kwargs,
):
    """
    Plot radial and transverse velocity components using line segments.
    """
    orbit_state.ensure_state_vectors()

    r_vec = np.asarray(orbit_state.r_vec, dtype=float)

    vr_plot = scale * np.asarray(orbit_state.v_r_vec, dtype=float)
    vt_plot = scale * np.asarray(orbit_state.v_t_vec, dtype=float)

    artists = {}

    # --- Radial component ---
    p0 = r_vec
    p1 = r_vec + vr_plot

    artists["v_r"] = ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        [p0[2], p1[2]],
        color=color_vr,
        linestyle=ls_vr,
        linewidth=linewidth,
        **kwargs,
    )

    # --- Transverse component ---
    if as_sum:
        p2 = p1
        p3 = p1 + vt_plot
    else:
        p2 = r_vec
        p3 = r_vec + vt_plot

    artists["v_t"] = ax.plot(
        [p2[0], p3[0]],
        [p2[1], p3[1]],
        [p2[2], p3[2]],
        color=color_vt,
        linestyle=ls_vt,
        linewidth=linewidth,
        **kwargs,
    )

    return artists

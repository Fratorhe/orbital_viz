import numpy as np
import plotly.graph_objects as go

from orbital_viz.plotly.p_utils import (
    _MPL_TO_PLOTLY_DASH,
    add_legend_entry,
    plot_vector,
)


def plot_position(fig, orbit_state, color="red", size=6, label=None, **kwargs):
    """
    Plot the current spacecraft position if theta is available.
    """
    required = ["a", "e", "i", "Omega", "omega", "theta"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        return None

    orbit_state.ensure_state_vectors()
    r_vec = np.asarray(orbit_state.r_vec, dtype=float)
    ref_length = orbit_state.r_a if orbit_state.r_a is not None else orbit_state.a
    point = go.Scatter3d(
        x=[r_vec[0]],
        y=[r_vec[1]],
        z=[r_vec[2]],
        mode="markers",
        marker=dict(
            size=ref_length * 5e-4,
            color=color,
            line=dict(color="black", width=1),
        ),
        name=label,
        showlegend=False,
        hoverinfo="skip",
        **kwargs,
    )
    fig.add_trace(point)

    if label is not None:
        add_legend_entry(
            fig,
            label=label,
            color=color,
            linewidth=2,
            linestyle="-",
        )

    return point


def plot_position_vector(
    fig,
    orbit_state,
    color="C1",
    linewidth=4,
    alpha=0.9,
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

    orbit_state.ensure_state_vectors()
    r_vec = np.asarray(orbit_state.r_vec, dtype=float)
    ref_length = orbit_state.r_a if orbit_state.r_a is not None else orbit_state.a

    artist = plot_vector(
        fig,
        origin=np.zeros(3),
        vec=r_vec,
        color=color,
        scale=1.0,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
        ref_length=ref_length,
        **kwargs,
    )

    return artist


def plot_velocity_vector(
    fig,
    orbit_state,
    scale=500.0,
    color="C2",
    linewidth=4,
    normalize=False,
    alpha=0.9,
    label=None,
    **kwargs,
):
    """
    Plot the velocity vector at the current spacecraft position.
    """
    required = ["a", "e", "i", "Omega", "omega", "theta"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        return None

    orbit_state.ensure_state_vectors()
    r_vec = np.asarray(orbit_state.r_vec, dtype=float)
    v_vec = np.asarray(orbit_state.v_vec, dtype=float)

    if normalize:
        v_norm = np.linalg.norm(v_vec)
        if v_norm > 0:
            v_plot = v_vec / v_norm
            vec_scale = scale
        else:
            v_plot = np.zeros(3)
            vec_scale = 0.0
    else:
        v_plot = v_vec
        vec_scale = scale
    ref_length = orbit_state.r_a if orbit_state.r_a is not None else orbit_state.a

    artist = plot_vector(
        fig,
        origin=r_vec,
        vec=v_plot,
        color=color,
        scale=vec_scale,
        alpha=alpha,
        linewidth=linewidth,
        label=label,
        ref_length=ref_length,
        **kwargs,
    )

    return artist


def plot_current_state(
    fig,
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
        artists["position"] = plot_position(fig, orbit_state, **position_kwargs)

    if show_r:
        artists["r_vec"] = plot_position_vector(fig, orbit_state, **r_kwargs)

    if show_v:
        artists["v_vec"] = plot_velocity_vector(fig, orbit_state, **v_kwargs)

    return artists


def plot_velocity_components(
    fig,
    orbit_state,
    scale=500.0,
    color_vr="gray",
    color_vt="gray",
    ls_vr="--",
    ls_vt=":",
    as_sum=False,
    linewidth=2,
    alpha=0.6,
    label_vr=None,
    label_vt=None,
    **kwargs,
):
    """
    Plot radial and transverse velocity components using plot_vector.

    Default style:
        - gray color
        - dashed/dotted
        - lower alpha (secondary visual importance)
    """

    orbit_state.ensure_state_vectors()

    r_vec = np.asarray(orbit_state.r_vec, dtype=float)

    vr_plot = np.asarray(orbit_state.v_r_vec, dtype=float)
    vt_plot = np.asarray(orbit_state.v_t_vec, dtype=float)

    # --- Reference length for consistent cone size ---
    ref_length = orbit_state.r_a if orbit_state.r_a is not None else orbit_state.a

    artists = {}

    dash_vr = _MPL_TO_PLOTLY_DASH.get(ls_vr, "dash")
    dash_vt = _MPL_TO_PLOTLY_DASH.get(ls_vt, "dot")

    # --- Radial component ---
    artists["v_r"] = plot_vector(
        fig,
        origin=r_vec,
        vec=vr_plot,
        color=color_vr,
        scale=scale,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=dash_vr,
        label=label_vr,
        ref_length=ref_length,
        cone_fraction=0.005,
        **kwargs,
    )

    # --- Transverse component ---
    if as_sum:
        p2 = r_vec + scale * vr_plot
    else:
        p2 = r_vec

    artists["v_t"] = plot_vector(
        fig,
        origin=p2,
        vec=vt_plot,
        color=color_vt,
        scale=scale,
        alpha=alpha,
        linewidth=linewidth,
        linestyle=dash_vt,
        label=label_vt,
        ref_length=ref_length,
        cone_fraction=0.005,
        **kwargs,
    )

    return artists

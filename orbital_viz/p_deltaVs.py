import numpy as np

from orbital_viz.p_utils import add_legend_entry  # your helper


def plot_delta_v_hohmann(
    ax,
    orbit_state,
    delta_v,
    color="limegreen",
    scale=300.0,
    linewidth=3,
    alpha=0.9,
    linestyle="-",
    label=None,
    marker_size=8,
):
    """
    Plot a delta-v vector with an arrowhead at the tip.

    The arrow is aligned with the velocity direction (tangential).
    """

    orbit_state.ensure_state_vectors()

    r_vec = np.asarray(orbit_state.r_vec, dtype=float)
    v_vec = np.asarray(orbit_state.v_vec, dtype=float)

    v_norm = np.linalg.norm(v_vec)
    if v_norm < 1e-12:
        return None

    v_hat = v_vec / v_norm
    dv_vec = scale * delta_v * v_hat

    tip = r_vec + dv_vec

    # --- Main line ---
    ax.plot(
        [r_vec[0], tip[0]],
        [r_vec[1], tip[1]],
        [r_vec[2], tip[2]],
        color=color,
        linewidth=linewidth,
        alpha=alpha,
        linestyle=linestyle,
    )

    # --- Arrowhead (marker at tip) ---
    ax.plot(
        [tip[0]],
        [tip[1]],
        [tip[2]],
        marker="d",
        markersize=marker_size,
        color=color,
        alpha=alpha,
    )

    # --- Legend ---
    add_legend_entry(
        ax,
        label=label,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
    )

    return ax

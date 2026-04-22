from matplotlib.lines import Line2D

from code.p_central_body import plot_central_body
from code.p_orbit import plot_orbit
from code.p_plot_plane import plot_orbital_plane
from code.p_position_velocity_vectors import (
    plot_position,
    plot_position_vector,
    plot_velocity_components,
    plot_velocity_vector,
)
from code.p_utils import add_legend_entry


def plot_orbit_scene(
    ax,
    orbit_state,
    body="Earth",
    color="C0",
    label=None,
    show_plane=True,
    show_position=True,
    show_vectors=True,
    show_direction=True,
    scale_vectors=500,
):
    """
    High-level wrapper to plot a full orbital scene in a clean, consistent style.
    """

    # --- Ensure state is ready ---
    orbit_state.ensure_state_vectors()
    orbit_state.ensure_elements()

    # --- Central body ---
    plot_central_body(ax, body)

    # --- Orbital plane (very light) ---
    if show_plane:
        plot_orbital_plane(
            ax,
            orbit_state,
            color=color,
            alpha=0.08,  # very subtle
        )

    # --- Orbit ---
    plot_orbit(
        ax,
        orbit_state,
        color=color,
        linewidth=2,
        alpha=0.9,
        ls="-",
        show_apses=True,
        show_direction=show_direction,
    )

    # --- Position ---
    if show_position:
        plot_position(
            ax,
            orbit_state,
            color=color,
            size=40,
        )

        plot_position_vector(
            ax,
            orbit_state,
            color=color,
            linewidth=2,
            alpha=0.8,
        )

    # --- Velocity ---
    if show_vectors:
        plot_velocity_vector(
            ax,
            orbit_state,
            color=color,
            linewidth=2,
            alpha=0.6,
            linestyle="-",
            scale=scale_vectors,
        )

        # Components: lighter + different style
        plot_velocity_components(
            ax,
            orbit_state,
            color_vr=color,
            color_vt=color,
            ls_vr="--",
            ls_vt=":",
            alpha=0.6,
            as_sum=True,
        )

    # --- Legend storage on axis ---
    add_legend_entry(
        ax,
        label=label,
        color=color,
        linewidth=2,
        linestyle="-",
    )

    return ax

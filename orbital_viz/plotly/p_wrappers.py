from orbital_viz.plotly.p_central_body import plot_central_body
from orbital_viz.plotly.p_orbit import plot_orbit
from orbital_viz.plotly.p_plot_plane import plot_orbital_plane
from orbital_viz.plotly.p_position_velocity_vectors import (
    plot_position,
    plot_position_vector,
    plot_velocity_components,
    plot_velocity_vector,
)
from orbital_viz.plotly.p_utils import add_legend_entry


def plot_orbit_scene(
    fig,
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
    High-level Plotly wrapper to plot a full orbital scene.
    """

    # --- Ensure state is ready ---
    orbit_state.ensure_state_vectors()
    orbit_state.ensure_elements()

    # --- Reference length for consistent vector/cone scaling ---
    ref_length = orbit_state.r_a if orbit_state.r_a is not None else orbit_state.a

    # --- Central body ---
    plot_central_body(
        fig,
        body=body,
        scene_scale=ref_length,
    )

    # --- Orbital plane ---
    if show_plane:
        plot_orbital_plane(
            fig,
            orbit_state,
            color=color,
            alpha=0.08,
        )

    # --- Orbit ---
    plot_orbit(
        fig,
        orbit_state,
        color=color,
        linewidth=5,
        alpha=0.9,
        ls="-",
        show_apses=True,
        show_direction=show_direction,
    )

    # --- Position ---
    if show_position:
        plot_position(
            fig,
            orbit_state,
            color=color,
            ref_length=ref_length * 1e-1,
        )

        plot_position_vector(
            fig,
            orbit_state,
            color=color,
            linewidth=3,
            alpha=0.6,
            cone_fraction=0.01,
        )

    # --- Velocity ---
    if show_vectors:
        plot_velocity_vector(
            fig,
            orbit_state,
            color=color,
            linewidth=4,
            alpha=0.7,
            linestyle="-",
            scale=scale_vectors,
            cone_fraction=0.02,
        )

        plot_velocity_components(
            fig,
            orbit_state,
            color_vr="gray",
            color_vt="gray",
            ls_vr="--",
            ls_vt=":",
            alpha=0.5,
            as_sum=True,
            scale=scale_vectors,
        )

    # --- Legend entry ---
    add_legend_entry(
        fig,
        label=label,
        color=color,
        linewidth=5,
        linestyle="-",
    )

    return fig

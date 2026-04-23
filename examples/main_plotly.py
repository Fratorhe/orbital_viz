import plotly.graph_objects as go

from orbital_viz.constants import deg2rad, mu_earth
from orbital_viz.orbital_state import OrbitalState
from orbital_viz.plotly.p_central_body import plot_central_body
from orbital_viz.plotly.p_orbit import plot_orbit
from orbital_viz.plotly.p_plot_plane import plot_orbital_plane
from orbital_viz.plotly.p_position_velocity_vectors import (
    plot_current_state,
    plot_velocity_components,
)
from orbital_viz.plotly.p_utils import show_figure

orbital_object = OrbitalState(
    a=13000,
    e=0.7,
    i=5 * deg2rad,
    Omega=0 * deg2rad,
    omega=0 * deg2rad,
    theta=20 * deg2rad,
    mu=mu_earth,
)

fig = go.Figure()

plot_central_body(fig, "Mars")


plot_orbit(
    fig,
    orbital_object,
    color="red",
    linewidth=6,
    ls="--",
    show_apses=True,
    show_direction=True,
)
plot_orbital_plane(fig, orbital_object, color="red")


orbital_object = OrbitalState(
    a=18000,
    e=0.4,
    i=45 * deg2rad,
    Omega=0 * deg2rad,
    omega=152 * deg2rad,
    theta=20 * deg2rad,
    mu=mu_earth,
)

plot_orbit(
    fig,
    orbital_object,
    color="blue",
    linewidth=6,
    ls="--",
    show_apses=True,
    show_direction=True,
)

plot_orbital_plane(fig, orbital_object, color="blue")

plot_current_state(
    fig,
    orbital_object,
)

plot_velocity_components(fig, orbital_object)

show_figure(fig)
plot_velocity_components(fig, orbital_object)

show_figure(fig)

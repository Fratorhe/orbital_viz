from matplotlib import pyplot as plt

from orbital_viz.constants import deg2rad, mu_earth
from orbital_viz.orbital_state import OrbitalState
from orbital_viz.plt.p_central_body import plot_central_body
from orbital_viz.plt.p_orbit import plot_orbit
from orbital_viz.plt.p_plot_plane import plot_orbital_plane
from orbital_viz.plt.p_position_velocity_vectors import (
    plot_position,
    plot_position_vector,
    plot_velocity_components,
    plot_velocity_vector,
)
from orbital_viz.plt.plotter import setup_axis

fig, ax = setup_axis(view="2D", lim=20000)
orbital_object = OrbitalState(
    a=13000,
    e=0.7,
    i=5 * deg2rad,
    Omega=0 * deg2rad,
    omega=0 * deg2rad,
    theta=20 * deg2rad,
    mu=mu_earth,
)

plot_central_body(ax, "Mars")

plot_orbit(
    ax,
    orbital_object,
    color="red",
    show_apses=True,
    linewidth=2,
)

plot_position(
    ax,
    orbital_object,
    color="red",
)

plot_position_vector(
    ax,
    orbital_object,
    color="red",
)

plot_velocity_vector(
    ax,
    orbital_object,
    color="red",
)

plot_velocity_components(ax, orbital_object, color_vr="red", color_vt="red", alpha=0.7)


plot_orbital_plane(ax, orbital_object, color="red")

plt.show()

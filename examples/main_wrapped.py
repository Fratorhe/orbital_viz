from matplotlib import pyplot as plt

from orbital_viz.constants import deg2rad, mu_earth
from orbital_viz.orbital_state import OrbitalState
from orbital_viz.plt.p_wrappers import plot_orbit_scene
from orbital_viz.plt.plotter import setup_axis

fig, ax = setup_axis(view="3D", lim=20000)

orbit = OrbitalState(
    a=13000,
    e=0.7,
    i=5 * deg2rad,
    Omega=0 * deg2rad,
    omega=0 * deg2rad,
    theta=20 * deg2rad,
    mu=mu_earth,
)

plot_orbit_scene(ax, orbit, body="Mars", color="red", label="Object1")

orbit2 = OrbitalState(
    a=16000,
    e=0.9,
    i=25 * deg2rad,
    Omega=0 * deg2rad,
    omega=0 * deg2rad,
    theta=170 * deg2rad,
    mu=mu_earth,
)

plot_orbit_scene(ax, orbit2, body="Mars", color="blue", label="Object2")


plt.show()

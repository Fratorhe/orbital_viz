from matplotlib import pyplot as plt

from orbital_viz.c_hohmann_tr import compute_hohmann_transfer
from orbital_viz.constants import deg2rad, mu_earth
from orbital_viz.orbital_state import OrbitalState
from orbital_viz.plt.p_deltaVs import plot_delta_v_hohmann
from orbital_viz.plt.p_wrappers import plot_orbit_scene
from orbital_viz.plt.plotter import setup_axis

initial = OrbitalState(
    a=13000,
    e=0.7,
    i=5 * deg2rad,
    Omega=0 * deg2rad,
    omega=0 * deg2rad,
    theta=0 * deg2rad,
    mu=mu_earth,
)


final = OrbitalState(
    a=10000,
    e=0.5,
    i=5 * deg2rad,
    Omega=0 * deg2rad,
    omega=0 * deg2rad,
    theta=180 * deg2rad,
    mu=mu_earth,
)

tr_orbit, dv1, dv2, tof = compute_hohmann_transfer(initial, final)


fig, ax = setup_axis(view="2D", lim=20000)

plot_orbit_scene(ax, initial, body="Earth", color="blue", label="Initial")
plot_orbit_scene(ax, final, body="Earth", color="red", label="Final")
plot_orbit_scene(ax, tr_orbit, body="Earth", color="orange", label="Transfer")
plot_delta_v_hohmann(ax, tr_orbit, dv1, color="green", label="deltaV1", scale=10000)

tr_orbit.theta = 180 * deg2rad
plot_orbit_scene(ax, tr_orbit, body="Earth", color="orange", label="Transfer")
plot_delta_v_hohmann(ax, tr_orbit, dv2, color="green", label="deltaV2", scale=10000)

print(f"Delta v are: {dv1}, {dv2} km/s")

plt.show()

import plotly.graph_objects as go

from orbital_viz.constants import deg2rad, mu_earth
from orbital_viz.orbital_state import OrbitalState
from orbital_viz.plotly.p_central_body import plot_central_body
from orbital_viz.plotly.p_orbit import plot_orbit

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
    show_direction=False,
)

orbital_object = OrbitalState(
    a=18000,
    e=0.4,
    i=45 * deg2rad,
    Omega=0 * deg2rad,
    omega=35 * deg2rad,
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
    show_direction=False,
)

fig.update_layout(
    scene=dict(
        xaxis_title="x [km]",
        yaxis_title="y [km]",
        zaxis_title="z [km]",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, b=0, t=0),
)

fig.show()

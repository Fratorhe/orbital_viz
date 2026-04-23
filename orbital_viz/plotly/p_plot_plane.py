import numpy as np
import plotly.graph_objects as go


def plot_orbital_plane(
    fig,
    orbit_state,
    size=None,
    color="royalblue",
    alpha=0.15,
    **kwargs,
):
    """
    Plot the orbital plane as a transparent square centered at the origin.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Figure where the plane will be added.
    orbit_state : OrbitalState
        Must contain at least a, e, i, Omega.
    size : float or None, optional
        Half-size of the square in km. The full side length is 2*size.
        If None, a reasonable value is estimated from the orbit.
    color : str, optional
        Plane color.
    alpha : float, optional
        Transparency of the plane.
    **kwargs
        Extra keyword arguments passed to go.Mesh3d().

    Returns
    -------
    mesh : go.Mesh3d
        Plotly mesh trace added to the figure.
    """
    required = ["a", "e", "i", "Omega"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        raise ValueError(f"Missing orbital elements for plane plotting: {missing}")

    a = orbit_state.a
    e = orbit_state.e
    i = orbit_state.i
    Omega = orbit_state.Omega

    assert a is not None
    assert e is not None
    assert i is not None
    assert Omega is not None

    if size is None:
        if e < 1:
            size = 1.1 * a * (1 + e)
        else:
            size = 2.0 * abs(a)

    # First in-plane direction: along line of nodes
    p_hat = np.array([np.cos(Omega), np.sin(Omega), 0.0])

    # Plane normal
    h_hat = np.array(
        [
            np.sin(i) * np.sin(Omega),
            -np.sin(i) * np.cos(Omega),
            np.cos(i),
        ]
    )
    h_hat = h_hat / np.linalg.norm(h_hat)

    # Second in-plane direction
    q_hat = np.cross(h_hat, p_hat)
    q_hat = q_hat / np.linalg.norm(q_hat)

    # Four corners of the square
    corners = np.array(
        [
            -size * p_hat - size * q_hat,
            +size * p_hat - size * q_hat,
            +size * p_hat + size * q_hat,
            -size * p_hat + size * q_hat,
        ]
    )

    x = corners[:, 0]
    y = corners[:, 1]
    z = corners[:, 2]

    # Two triangles: (0,1,2) and (0,2,3)
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        color=color,
        opacity=alpha,
        flatshading=True,
        hoverinfo="skip",
        showlegend=False,
        **kwargs,
    )

    fig.add_trace(mesh)
    return mesh

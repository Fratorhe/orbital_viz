import numpy as np


def plot_orbital_plane(
    ax,
    orbit_state,
    size=None,
    color="C0",
    alpha=0.15,
    **kwargs,
):
    """
    Plot the orbital plane as a transparent square centered at the origin.

    Parameters
    ----------
    ax : matplotlib 3D axis
        Axis where the plane will be plotted.
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
        Extra keyword arguments passed to ax.plot_surface().

    Returns
    -------
    surface
        Output from ax.plot_surface(...)
    """
    required = ["a", "e", "i", "Omega"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        raise ValueError(f"Missing orbital elements for plane plotting: {missing}")

    a = orbit_state.a
    e = orbit_state.e
    i = orbit_state.i
    Omega = orbit_state.Omega

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

    # Square coordinates in local plane basis
    u = np.array([-size, size])
    v = np.array([-size, size])
    U, V = np.meshgrid(u, v)

    X = U * p_hat[0] + V * q_hat[0]
    Y = U * p_hat[1] + V * q_hat[1]
    Z = U * p_hat[2] + V * q_hat[2]

    surface = ax.plot_surface(
        X,
        Y,
        Z,
        color=color,
        alpha=alpha,
        linewidth=0,
        shade=False,
        **kwargs,
    )

    return surface

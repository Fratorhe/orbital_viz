import numpy as np

from orbital_viz.orbital_state import OrbitalState


def plot_orbit(
    ax,
    orbit_state,
    n_points=500,
    theta_range=None,
    alpha=0.7,
    ls="--",
    show_apses=False,
    show_direction=True,
    **kwargs,
):
    """
    Plot an orbit in 3D using the orbital elements stored in orbit_state.
    Optionally plot the line of apses.

    Returns
    -------
    dict with:
        "orbit": line artist
        "apses": line artist (if requested)
    """

    required = ["a", "e", "i", "Omega", "omega"]
    missing = [name for name in required if getattr(orbit_state, name) is None]
    if missing:
        raise ValueError(f"Missing orbital elements for plotting: {missing}")

    e = orbit_state.e
    a = orbit_state.a

    if theta_range is None:
        if e < 1:
            theta_vals = np.linspace(0.0, 2.0 * np.pi, n_points)
        else:
            theta_lim = np.arccos(-1.0 / e) - 1e-3
            theta_vals = np.linspace(-theta_lim, theta_lim, n_points)
    else:
        theta_vals = np.linspace(theta_range[0], theta_range[1], n_points)

    r_points = []

    for theta in theta_vals:
        temp_state = orbit_state.copy()
        temp_state.theta = theta

        temp_state.compute_state_vectors()
        r_points.append(temp_state.r_vec)

    r_points = np.asarray(r_points)

    x = r_points[:, 0]
    y = r_points[:, 1]
    z = r_points[:, 2]

    artists = {}

    artists["orbit"] = ax.plot(x, y, z, alpha=alpha, ls=ls, **kwargs)
    orbit_color = kwargs.get("color", None)
    orbit_lw = kwargs.get("linewidth", 2)

    # --- Line of apses ---
    if show_apses:
        apses_points = orbit_state.get_apses_line_points()

        if apses_points is not None:
            p1, p2 = apses_points

            orbit_color = kwargs.get("color", None)

            artists["apses"] = ax.plot(
                [p1[0], p2[0]],
                [p1[1], p2[1]],
                [p1[2], p2[2]],
                color=orbit_color,
                linestyle="-.",
                alpha=alpha * 0.7,
                linewidth=orbit_lw,
            )

    # --- Direction arrows ---
    if show_direction and n_points >= 2:
        idx = int(0.6 * (n_points - 2))  # good visual location

        p0 = r_points[idx]
        p1 = r_points[idx + 1]

        d = p1 - p0
        d_norm = np.linalg.norm(d)

        if d_norm > 1e-12:
            d_hat = d / d_norm

            # Scale arrow ~10% of orbit size
            orbit_scale = np.max(np.linalg.norm(r_points, axis=1))
            arrow_length = 0.1 * orbit_scale

            d_plot = arrow_length * d_hat

            artists["direction"] = ax.quiver(
                p0[0],
                p0[1],
                p0[2],
                d_plot[0],
                d_plot[1],
                d_plot[2],
                color=orbit_color,
                alpha=min(1.0, alpha + 0.1),
                linewidth=orbit_lw,
            )

    return artists

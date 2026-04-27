import numpy as np
from scipy.optimize import fsolve

from orbital_viz.orbital_state import OrbitalState

deg2rad = np.pi / 180


def compute_time_from_periapsis(orbital_object):
    e = orbital_object.e
    period = orbital_object.period
    theta = orbital_object.theta

    theta = np.mod(theta, 2 * np.pi)

    E = 2 * np.arctan2(
        np.sqrt(1 - e) * np.sin(theta / 2), np.sqrt(1 + e) * np.cos(theta / 2)
    )
    E = np.mod(E, 2 * np.pi)

    M = E - e * np.sin(E)
    M = np.mod(M, 2 * np.pi)

    t_from_periapsis = M * period / (2 * np.pi)
    return t_from_periapsis


def compute_theta_from_periapsis(orbital_object, time):
    e = orbital_object.e
    period = orbital_object.period

    M = time * 2 * np.pi / period
    M = np.mod(M, 2 * np.pi)

    # better initial guess than always 1
    E0 = M if e < 0.8 else np.pi

    E_sol = fsolve(lambda E: E - e * np.sin(E) - M, E0)[0]
    E_sol = np.mod(E_sol, 2 * np.pi)

    theta = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E_sol / 2), np.sqrt(1 - e) * np.cos(E_sol / 2)
    )
    theta = np.mod(theta, 2 * np.pi)

    return theta


if __name__ == "__main__":
    theta = 75.0 * deg2rad  # type: ignore
    print(theta)
    e = 0.348837
    period = 11092  # seconds
    orbiting_object = OrbitalState(a=10750, e=0.348837, theta=theta)
    orbiting_object.period = period

    time_from_periapsis = compute_time_from_periapsis(orbiting_object)
    print(time_from_periapsis)

    theta = compute_theta_from_periapsis(orbiting_object, time_from_periapsis)
    print(theta)

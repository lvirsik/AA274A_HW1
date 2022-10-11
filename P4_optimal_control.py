import typing as T

import numpy as np
import matplotlib.pyplot as plt  # type: ignore
from scipy.optimize import minimize, Bounds  # type: ignore
from utils import save_dict, maybe_makedirs

N = 20  # Number of time discretization nodes (0, 1, ... N).
s_dim = 3  # State dimension; 3 for (x, y, th).
u_dim = 2  # Control dimension; 2 for (V, om).
v_max = 0.5  # Maximum linear velocity.
om_max = 1.0  # Maximum angular velocity.

s_0 = np.array([0, 0, -np.pi / 2])  # Initial state.
s_f = np.array([5, 5, -np.pi / 2])  # Final state.


def pack_decision_variables(t_f: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Packs decision variables (final time, states, controls) into a 1D vector.

    Args:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).

    Returns:
        An array `z` of shape (1 + (N + 1) * s_dim + N * u_dim,).
    """
    return np.concatenate([[t_f], s.ravel(), u.ravel()])


def unpack_decision_variables(z: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Unpacks a 1D vector into decision variables (final time, states, controls).

    Args:
        z: An array of shape (1 + (N + 1) * s_dim + N * u_dim,).

    Returns:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).
    """
    t_f = float(z[0])
    s = z[1:1 + (N + 1) * s_dim].reshape(N + 1, s_dim)
    u = z[-N * u_dim:].reshape(N, u_dim)
    return t_f, s, u




def optimize_trajectory(
    time_weight: float = 1.0,
    verbose: bool = True
) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Computes the optimal trajectory as a function of `time_weight`.

    Args:
        time_weight: \lambda in the HW writeup.

    Returns:
        t_f_opt: Optimal final time, a scalar.
        s_opt: Optimal states, an array of shape (N + 1, s_dim).
        u_opt: Optimal controls, an array of shape (N, u_dim).
    """

    # NOTE: When using `minimize`, you may find the utilities
    # `pack_decision_variables` and `unpack_decision_variables` useful.

    # WRITE YOUR CODE BELOW ###################################################

    t_f_guess = 0.5

    states_Guess = np.zeros((N,3))
    controls_Guess = np.zeros((N,2))

    for i in range(N):
        states_Guess[i,0] = (s_f[0] - s_0[0]) * i / N
        states_Guess[i,1] = (s_f[1] - s_0[1]) * i / N
        states_Guess[i,2] = (s_f[2] - s_0[2]) * i / N
        controls_Guess[i,0] = 3
        controls_Guess[i,0] = 3

    def f(z) -> float:
        t_f, s, c = unpack_decision_variables(z)
        result = 0
        for i in range(len(s)):
            result = result + (time_weight + [i - 1][0] ** 2 + c[i - 1][1] ** 2) * t_f / N
        return result

    def dynamics(t_f, s, c):
        x = s[0]
        y = s[1]
        th = s[2]
        V = c[0]
        om = c[1]
        xd = V * np.cos(th)
        yd = V * np.sin(th)
        thd = om
        result = np.array([xd,yd,thd])
        return result

    xbnd = []
    ybnd = []
    thbnd = []
    Vbnd = []
    ombnd = []
    for i in range(N):
        xbnd.append((None, None))
        ybnd.append((None, None))
        thbnd.append((None, None))
        Vbnd.append((-0.5,0.5))
        ombnd.append((-1,1))
    bnds = [(None, None)] + xbnd + ybnd + thbnd + Vbnd + ombnd

    def constraints(z):
        t_f, s, c = unpack_decision_variables(z)
        constraint_list = [s[0] - s_0, s[-1] - s_f]
        for i in range(N):
            constraint_list.append(s[i + 1] - (s[i] + (t_f/N) * dynamics(t_f, s[i], c[i])))
        return np.concatenate(constraint_list)

    packed_Guess = pack_decision_variables(t_f_guess, states_Guess, controls_Guess)
    minimum = minimize(f, packed_Guess, constraints= {'type': 'eq', 'fun': constraints}, bounds=bnds)
    t_f_opt, s_opt, c_opt = unpack_decision_variables(minimum.x)
    return t_f_opt, s_opt, c_opt


    ###########################################################################


if __name__ == '__main__':
    for time_weight in (1.0, 0.2):
        t_f, s, u = optimize_trajectory(time_weight)
        V = u[:, 0]
        om = u[:, 1]
        t = np.linspace(0, t_f, N + 1)[:-1]
        x = s[:, 0]
        y = s[:, 1]
        th = s[:, 2]
        data = {'t_f': t_f, 's': s, 'u': u}
        save_dict(data, f'data/optimal_control_{time_weight}.pkl')
        maybe_makedirs('plots')

        # plotting
        # plt.rc('font', weight='bold', size=16)
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, y, 'k-', linewidth=2)
        plt.quiver(x, y, np.cos(th), np.sin(th))
        plt.grid(True)
        plt.plot(0, 0, 'go', markerfacecolor='green', markersize=15)
        plt.plot(5, 5, 'ro', markerfacecolor='red', markersize=15)
        plt.xlabel('X [m]')
        plt.ylabel('Y [m]')
        plt.axis([-1, 6, -1, 6])
        plt.title(f'Optimal Control Trajectory (lambda = {time_weight})')

        plt.subplot(1, 2, 2)
        plt.plot(t, V, linewidth=2)
        plt.plot(t, om, linewidth=2)
        plt.grid(True)
        plt.xlabel('Time [s]')
        plt.legend(['V [m/s]', '$\omega$ [rad/s]'], loc='best')
        plt.title(f'Optimal control sequence (lambda = {time_weight})')
        plt.tight_layout()
        plt.savefig(f'plots/optimal_control_{time_weight}.png')
        plt.show()

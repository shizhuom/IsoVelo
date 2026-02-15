"""Simulation utilities for a two-isoform system.

Equations:
	du/dt  = alpha - (beta1 + beta2) * u
	ds1/dt = beta1 * u - gamma1 * s1
	ds2/dt = beta2 * u - gamma2 * s2
"""

from __future__ import annotations

from typing import Iterable, Tuple
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np


def time_grid(t0: float, t1: float, dt: float) -> np.ndarray:
	"""Create an inclusive time grid [t0, t1] with step dt."""
	if dt <= 0:
		raise ValueError("dt must be positive")
	if t1 < t0:
		raise ValueError("t1 must be >= t0")
	n_steps = int(np.floor((t1 - t0) / dt))
	return t0 + dt * np.arange(n_steps + 1, dtype=float)


def simulate_two_isoforms(
	t: Iterable[float] | np.ndarray,
	alpha: float,
	beta1: float,
	beta2: float,
	gamma1: float,
	gamma2: float,
	u0: float = 0.0,
	s10: float = 0.0,
	s20: float = 0.0,
	method: str = "analytic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	"""Simulate the two-isoform system.

	Args:
		t: Time points (1D, increasing).
		alpha, beta1, beta2, gamma1, gamma2: Model parameters.
		u0, s10, s20: Initial values at t[0].
		method: "analytic" (closed-form) or "rk4" (numeric).

	Returns:
		(u, s1, s2) arrays with the same length as t.
	"""
	t = np.asarray(t, dtype=float)
	if t.ndim != 1 or t.size == 0:
		raise ValueError("t must be a non-empty 1D array")
	if np.any(np.diff(t) <= 0):
		raise ValueError("t must be strictly increasing")

	method = method.lower().strip()
	if method == "analytic":
		return _simulate_analytic(t, alpha, beta1, beta2, gamma1, gamma2, u0, s10, s20)
	if method == "rk4":
		return _simulate_rk4(t, alpha, beta1, beta2, gamma1, gamma2, u0, s10, s20)
	raise ValueError("method must be 'analytic' or 'rk4'")


def _simulate_analytic(
	t: np.ndarray,
	alpha: float,
	beta1: float,
	beta2: float,
	gamma1: float,
	gamma2: float,
	u0: float,
	s10: float,
	s20: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	k = beta1 + beta2
	if k <= 0:
		raise ValueError("beta1 + beta2 must be positive")

	u_ss = alpha / k
	exp_k = np.exp(-k * (t - t[0]))
	u = u_ss + (u0 - u_ss) * exp_k

	s1 = _s_analytic(t, u0, s10, beta1, gamma1, u_ss, k)
	s2 = _s_analytic(t, u0, s20, beta2, gamma2, u_ss, k)
	return u, s1, s2


def _s_analytic(
	t: np.ndarray,
	u0: float,
	s0: float,
	beta: float,
	gamma: float,
	u_ss: float,
	k: float,
) -> np.ndarray:
	t0 = t[0]
	tau = t - t0
	exp_g = np.exp(-gamma * tau)

	if np.isclose(k, gamma):
		# Special case to avoid division by zero.
		s = (
			s0 * exp_g
			+ beta * (u_ss * (1.0 - exp_g) / gamma + (u0 - u_ss) * tau * exp_g)
		)
		return s

	exp_kg = np.exp(-(k - gamma) * tau)
	term_const = u_ss * (1.0 - exp_g) / gamma
	term_trans = (u0 - u_ss) * exp_g * (1.0 - exp_kg) / (k - gamma)
	return s0 * exp_g + beta * (term_const + term_trans)


def _simulate_rk4(
	t: np.ndarray,
	alpha: float,
	beta1: float,
	beta2: float,
	gamma1: float,
	gamma2: float,
	u0: float,
	s10: float,
	s20: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	
	def f(state: np.ndarray) -> np.ndarray:
		u, s1, s2 = state
		du = alpha - (beta1 + beta2) * u
		ds1 = beta1 * u - gamma1 * s1
		ds2 = beta2 * u - gamma2 * s2
		return np.array([du, ds1, ds2], dtype=float)

	n = t.size
	y = np.zeros((n, 3), dtype=float)
	y[0] = np.array([u0, s10, s20], dtype=float)

	for i in range(n - 1):
		dt = t[i + 1] - t[i]
		k1 = f(y[i])
		k2 = f(y[i] + 0.5 * dt * k1)
		k3 = f(y[i] + 0.5 * dt * k2)
		k4 = f(y[i] + dt * k3)
		y[i + 1] = y[i] + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

	return y[:, 0], y[:, 1], y[:, 2]


def steady_state_time(
	proportion: float,
	alpha: float,
	beta1: float,
	beta2: float,
	gamma1: float,
	gamma2: float,
) -> float:
	"""Estimate time to reach a given proportion of steady state.

	This uses the slowest rate among k=beta1+beta2, gamma1, gamma2.
	For a proportion p, remaining fraction is exp(-rate * t).
	"""
	if proportion <= 0.0 or proportion >= 1.0:
		raise ValueError("proportion must be between 0 and 1 (exclusive)")

	k = beta1 + beta2
	if k <= 0 or gamma1 <= 0 or gamma2 <= 0:
		raise ValueError("beta1+beta2, gamma1, and gamma2 must be positive")

	rate = min(k, gamma1, gamma2)
	return -np.log(1.0 - proportion) / rate

__all__ = [
	"time_grid",
	"simulate_two_isoforms",
	"steady_state_time",
]

dt = 0.1
steady_t1 = steady_state_time(
    proportion=0.99,
    alpha=2.0, beta1=1.6, beta2=1.4, gamma1=1, gamma2=1,
)
# Phase 1: t = 0..steady_t1
t1 = time_grid(0.0, steady_t1, dt)
u1, s1_1, s2_1 = simulate_two_isoforms(
    t1,
    alpha=2.0, beta1=1.6, beta2=1.4, gamma1=1, gamma2=1,
    u0=0.0, s10=0.0, s20=0.0,
    method="analytic",
)
# estimation (relative to gamma1 = gamma2 = 1)
beta1 = s1_1[-1] / u1[-1]
beta2 = s2_1[-1] / u1[-1]
alpha = s1_1[-1] + s2_1[-1]
print(f"Estimated parameters at t={steady_t1:.1f}: alpha={alpha:.2f}, beta1={beta1:.2f}, beta2={beta2:.2f}")

# Phase 2: t = steady_t1..steady_t2
steady_t2 = steady_state_time(
    proportion=0.99,
    alpha=2.0, beta1=2.4, beta2=0.8, gamma1=1, gamma2=1,
)
t2 = time_grid(steady_t1, steady_t1 + steady_t2, dt)
u2, s1_2, s2_2 = simulate_two_isoforms(
    t2,
    alpha=2.0, beta1=2.4, beta2=0.8, gamma1=1, gamma2=1,
    u0=u1[-1], s10=s1_1[-1], s20=s2_1[-1],
    method="analytic",
)
# estimation (relative to gamma1 = gamma2 = 1)
beta1 = s1_2[-1] / u2[-1]
beta2 = s2_2[-1] / u2[-1]
alpha = s1_2[-1] + s2_2[-1]
print(f"Estimated parameters at t={steady_t1 + steady_t2:.1f}: alpha={alpha:.2f}, beta1={beta1:.2f}, beta2={beta2:.2f}")

# Concatenate, dropping the duplicated time point at t=10 in phase 2
t = np.concatenate([t1, t2[1:]])
u = np.concatenate([u1, u2[1:]])
s1 = np.concatenate([s1_1, s1_2[1:]])
s2 = np.concatenate([s2_1, s2_2[1:]])

# 3D static plot with Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u, s1, s2, linewidth=2.0)
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

# 3D interactive plot with Plotly
fig = go.Figure(
    data=[
        go.Scatter3d(
            x=u,
            y=s1,
            z=s2,
            mode="lines",
            line=dict(width=4),
        )
    ]
)
fig.update_layout(
    scene=dict(xaxis_title="u", yaxis_title="s1", zaxis_title="s2"),
    margin=dict(l=0, r=0, b=0, t=0),
)
fig.show()
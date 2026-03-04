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
import math
import warnings

# Simulation 1 functions

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

# Simulation 2 functions

def simulate_two_phase_trajectory(
	dt: float,
	proportion: float,
	alpha1: float,
	beta1_1: float,
	beta2_1: float,
	gamma1_1: float,
	gamma2_1: float,
	alpha2: float,
	beta1_2: float,
	beta2_2: float,
	gamma1_2: float,
	gamma2_2: float,
	method: str = "analytic",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
	"""Simulate a two-phase trajectory and return concatenated states.

	Returns:
		t, u, s1, s2, steady_t1_center, steady_t2_center
		where steady_t1_center and steady_t2_center are absolute times.
	"""
	steady_t1 = steady_state_time(
		proportion=proportion,
		alpha=alpha1,
		beta1=beta1_1,
		beta2=beta2_1,
		gamma1=gamma1_1,
		gamma2=gamma2_1,
	)
	t1 = time_grid(0.0, steady_t1, dt)
	u1, s1_1, s2_1 = simulate_two_isoforms(
		t1,
		alpha=alpha1,
		beta1=beta1_1,
		beta2=beta2_1,
		gamma1=gamma1_1,
		gamma2=gamma2_1,
		u0=0.0,
		s10=0.0,
		s20=0.0,
		method=method,
	)

	steady_t2_duration = steady_state_time(
		proportion=proportion,
		alpha=alpha2,
		beta1=beta1_2,
		beta2=beta2_2,
		gamma1=gamma1_2,
		gamma2=gamma2_2,
	)
	steady_t2 = steady_t1 + steady_t2_duration
	t2 = time_grid(steady_t1, steady_t2, dt)
	u2, s1_2, s2_2 = simulate_two_isoforms(
		t2,
		alpha=alpha2,
		beta1=beta1_2,
		beta2=beta2_2,
		gamma1=gamma1_2,
		gamma2=gamma2_2,
		u0=u1[-1],
		s10=s1_1[-1],
		s20=s2_1[-1],
		method=method,
	)

	t = np.concatenate([t1, t2[1:]])
	u = np.concatenate([u1, u2[1:]])
	s1 = np.concatenate([s1_1, s1_2[1:]])
	s2 = np.concatenate([s2_1, s2_2[1:]])
	return t, u, s1, s2, steady_t1, steady_t2

def sample_real_case_cells(
	t: np.ndarray,
	u: np.ndarray,
	s1: np.ndarray,
	s2: np.ndarray,
	n_cells: int,
	p1: float,
	p2: float,
	steady_t1: float,
	steady_t2: float,
	steady_window1: float,
	steady_window2: float,
	seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	"""Sample unlabeled snapshot cells from mixed steady/transition states.

	Mixture:
		- p1 of cells sampled near steady_t1 (phase 1 steady neighborhood)
		- p2 of cells sampled near steady_t2 (phase 2 steady neighborhood)
		- 1-p1-p2 sampled uniformly in [steady_t1, steady_t2] (transition)

	Returns:
		u_obs, s1_obs, s2_obs, latent_t, group_label
		group_label: 1=phase1 steady, 2=phase2 steady, 0=transition
	"""
	if n_cells <= 0:
		raise ValueError("n_cells must be positive")
	if p1 < 0.0 or p2 < 0.0 or (p1 + p2) > 1.0:
		raise ValueError("p1 and p2 must satisfy p1>=0, p2>=0, and p1+p2<=1")
	if steady_t2 <= steady_t1:
		raise ValueError("steady_t2 must be greater than steady_t1")
	if steady_window1 < 0.0 or steady_window2 < 0.0:
		raise ValueError("steady windows must be non-negative")

	t = np.asarray(t, dtype=float)
	u = np.asarray(u, dtype=float)
	s1 = np.asarray(s1, dtype=float)
	s2 = np.asarray(s2, dtype=float)
	if t.ndim != 1 or u.ndim != 1 or s1.ndim != 1 or s2.ndim != 1:
		raise ValueError("t, u, s1, s2 must be 1D arrays")
	if not (t.size == u.size == s1.size == s2.size):
		raise ValueError("t, u, s1, and s2 must have the same length")
	if np.any(np.diff(t) <= 0):
		raise ValueError("t must be strictly increasing")

	raw_counts = np.array([p1, p2, 1.0 - p1 - p2], dtype=float) * n_cells
	counts = np.floor(raw_counts).astype(int)
	remainder = n_cells - int(counts.sum())
	if remainder > 0:
		order = np.argsort(raw_counts - counts)[::-1]
		counts[order[:remainder]] += 1
	n1, n2, n_transition = counts.tolist()

	rng = np.random.default_rng(seed)

	def sample_interval(n: int, start: float, end: float) -> np.ndarray:
		if n <= 0:
			return np.empty(0, dtype=float)
		lo, hi = (start, end) if start <= end else (end, start)
		if np.isclose(lo, hi):
			return np.full(n, lo, dtype=float)
		return rng.uniform(lo, hi, size=n)

	t_min, t_max = float(t[0]), float(t[-1])
	w1_lo = max(t_min, steady_t1 - 0.5 * steady_window1)
	w1_hi = min(t_max, steady_t1 + 0.5 * steady_window1)
	w2_lo = max(t_min, steady_t2 - 0.5 * steady_window2)
	w2_hi = min(t_max, steady_t2 + 0.5 * steady_window2)

	t_phase1 = sample_interval(n1, w1_lo, w1_hi)
	t_phase2 = sample_interval(n2, w2_lo, w2_hi)
	t_transition = sample_interval(n_transition, steady_t1, steady_t2)

	latent_t = np.concatenate([t_phase1, t_phase2, t_transition])
	group_label = np.concatenate(
		[
			np.full(n1, 1, dtype=int),
			np.full(n2, 2, dtype=int),
			np.full(n_transition, 0, dtype=int),
		]
	)
	perm = rng.permutation(n_cells)
	latent_t = latent_t[perm]
	group_label = group_label[perm]

	u_obs = np.interp(latent_t, t, u)
	s1_obs = np.interp(latent_t, t, s1)
	s2_obs = np.interp(latent_t, t, s2)

	return u_obs, s1_obs, s2_obs, latent_t, group_label

def _kmeans_1d(x: np.ndarray, k: int = 2, n_iter: int = 50, seed: int = 0):
    """Simple 1D k-means. Returns (labels, centers)."""
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)

    # Initialize centers via quantiles for stability.
    qs = np.linspace(0.2, 0.8, k)
    centers = np.quantile(x, qs)

    for _ in range(n_iter):
        d = np.abs(x[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = np.array(
            [x[labels == j].mean() if np.any(labels == j) else centers[j] for j in range(k)],
            dtype=float,
        )
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    return labels, centers

def _knn_density_scores(points: np.ndarray, k: int = 20) -> np.ndarray:
    """
    Brute-force kNN average distance (lower = denser).
    points: (n, d)
    """
    pts = np.asarray(points, float)
    n = pts.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=float)

    # Squared distance matrix: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    sq = np.sum(pts * pts, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * pts.dot(pts.T)
    dist2[dist2 < 0] = 0.0

    k_eff = min(k, n - 1)
    # Pick (k_eff+1) smallest per row (including self), using argpartition for speed.
    idx = np.argpartition(dist2, kth=k_eff, axis=1)[:, : k_eff + 1]
    dsel = np.take_along_axis(dist2, idx, axis=1)
    dsel.sort(axis=1)

    # Exclude the first element (self-distance = 0).
    mean_dist = np.mean(np.sqrt(dsel[:, 1 : k_eff + 1]), axis=1)
    return mean_dist

def _weighted_linreg(x: np.ndarray, y: np.ndarray, w: np.ndarray, fit_intercept: bool):
    """Weighted least squares fit y ~ m*x + b. Returns (m, b)."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)

    if not fit_intercept:
        denom = np.sum(w * x * x)
        if denom <= 0:
            return 0.0, 0.0
        m = np.sum(w * x * y) / denom
        return float(m), 0.0

    W = np.sum(w)
    if W <= 0:
        return 0.0, 0.0

    xw = np.sum(w * x)
    yw = np.sum(w * y)
    xxw = np.sum(w * x * x)
    xyw = np.sum(w * x * y)

    denom = W * xxw - xw * xw
    if np.isclose(denom, 0.0):
        # Fallback
        m, _ = _weighted_linreg(x, y, w, fit_intercept=False)
        b = (yw - m * xw) / W
        return float(m), float(b)

    m = (W * xyw - xw * yw) / denom
    b = (yw - m * xw) / W
    return float(m), float(b)

def _velocyto_diag_extreme_weights(u: np.ndarray, s: np.ndarray, maxmin_perc=(2, 98)):
    """
    Mimic velocyto 'maxmin_diag' extrema selection for one variable pair:
      X = s/den_s + u/den_u, then select bottom/top percentiles of X.
    """
    u = np.asarray(u, float)
    s = np.asarray(s, float)

    den_s = np.percentile(s, 99.9)
    den_u = np.percentile(u, 99.9)
    den_s = max(den_s, max(np.max(s), 1e-3))
    den_u = max(den_u, max(np.max(u), 1e-3))

    X = (s / den_s) + (u / den_u)
    down, up = np.percentile(X, maxmin_perc)
    w = ((X <= down) | (X >= up)).astype(float)
    return w

def _fit_equilibrium_slope_velocyto_like(u: np.ndarray, s: np.ndarray, maxmin_perc=(2, 98), fit_offset=True):
    """Fit u ~ rho*s + q using diagonal extreme weights (velocyto-inspired)."""
    w = _velocyto_diag_extreme_weights(u, s, maxmin_perc=maxmin_perc)
    if w.sum() < 5:
        w = np.ones_like(w)
    rho, q = _weighted_linreg(s, u, w, fit_intercept=fit_offset)
    # Constrain to non-negative slope (velocyto enforces non-negative gammas).
    rho = max(rho, 1e-8)
    q = max(q, 0.0) if fit_offset else 0.0
    return rho, q

def fit_isoform_switching_velocyto(
    u_obs: np.ndarray,
    s1_obs: np.ndarray,
    s2_obs: np.ndarray,
    *,
    state: str = "late",          # "early" (low s1 fraction) or "late" (high s1 fraction)
    steady_frac: float = 0.10,    # fraction per regime used as "steady representatives"
    knn_k: int = 20,              # k for kNN-density selection
    assume_equal_gamma: bool = True,
    gamma_equal_value: float = 1.0,
    # If assume_equal_gamma is False, solve gamma1/gamma2 from alpha invariance and apply scale:
    gamma_scale_mode: str = "geometric_mean_1",  # or "gamma2_to_1"
    # Velocyto-like regression diagnostics:
    maxmin_perc=(2, 98),
    fit_offset: bool = True,
    seed: int = 0,
    return_debug: bool = False,
):
    """
    Estimate (alpha, beta1, beta2, gamma1, gamma2) for a 2-isoform gene with a switch.

    Model:
      du/dt  = alpha - (beta1 + beta2) * u
      ds1/dt = beta1 * u - gamma1 * s1
      ds2/dt = beta2 * u - gamma2 * s2

    Returns:
      alpha, beta1, beta2, gamma1, gamma2   (for requested `state`)
    """
    u = np.asarray(u_obs, float)
    s1 = np.asarray(s1_obs, float)
    s2 = np.asarray(s2_obs, float)
    if u.shape != s1.shape or u.shape != s2.shape:
        raise ValueError("u_obs, s1_obs, s2_obs must have the same shape.")
    if u.size < 50:
        raise ValueError("Need at least ~50 cells for stable estimation.")

    eps = 1e-8
    f = s1 / (s1 + s2 + eps)  # isoform fraction coordinate

    # 1) Find two regimes via 1D k-means on isoform fraction.
    labels_raw, centers = _kmeans_1d(f, k=2, seed=seed)
    order = np.argsort(centers)
    remap = {int(order[i]): i for i in range(2)}  # 0=lower f, 1=higher f
    labels = np.vectorize(remap.get)(labels_raw)
    centers = centers[order]

    # 2) Select steady representatives per regime using density in (u, s1, s2).
    regimes = []
    for reg in [0, 1]:
        mask = labels == reg
        pts = np.stack([u[mask], s1[mask], s2[mask]], axis=1)
        dens = _knn_density_scores(pts, k=knn_k)  # lower is denser
        k_sel = max(5, int(np.ceil(steady_frac * pts.shape[0])))
        idx_local = np.argsort(dens)[:k_sel]
        idx = np.where(mask)[0][idx_local]

        u_ss = float(np.median(u[idx]))
        s1_ss = float(np.median(s1[idx]))
        s2_ss = float(np.median(s2[idx]))

        # Velocyto-style equilibrium slope diagnostics (may be ill-conditioned if u ~ const in a regime).
        rho1, q1 = _fit_equilibrium_slope_velocyto_like(u[mask], s1[mask], maxmin_perc=maxmin_perc, fit_offset=fit_offset)
        rho2, q2 = _fit_equilibrium_slope_velocyto_like(u[mask], s2[mask], maxmin_perc=maxmin_perc, fit_offset=fit_offset)

        regimes.append(
            dict(
                reg=reg,
                f_center=float(centers[reg]),
                u_ss=u_ss,
                s1_ss=s1_ss,
                s2_ss=s2_ss,
                rho1=rho1,
                q1=q1,
                rho2=rho2,
                q2=q2,
                n_cells=int(mask.sum()),
                n_steady=int(k_sel),
            )
        )

    # 3) Choose gamma scale (needed because absolute rates are not identifiable from snapshots).
    if assume_equal_gamma:
        gamma1 = gamma2 = float(gamma_equal_value)
    else:
        # Enforce alpha invariance across the two regimes:
        # alpha = gamma1*s1_ss + gamma2*s2_ss  (derived from steady-state equations)
        A, B = regimes[0], regimes[1]
        denom = A["s1_ss"] - B["s1_ss"]
        if np.isclose(denom, 0.0):
            gamma_ratio = 1.0
        else:
            gamma_ratio = (B["s2_ss"] - A["s2_ss"]) / denom  # gamma1/gamma2
        gamma_ratio = float(gamma_ratio)
        if (not np.isfinite(gamma_ratio)) or gamma_ratio <= 0:
            gamma_ratio = 1.0

        if gamma_scale_mode == "geometric_mean_1":
            gamma1 = math.sqrt(gamma_ratio)
            gamma2 = 1.0 / gamma1
        elif gamma_scale_mode == "gamma2_to_1":
            gamma2 = 1.0
            gamma1 = gamma_ratio
        else:
            raise ValueError("Unknown gamma_scale_mode.")

    # 4) Compute beta and alpha per regime from steady-state algebra:
    #    beta1 = gamma1*s1_ss/u_ss, beta2 = gamma2*s2_ss/u_ss, alpha = (beta1+beta2)*u_ss
    for R in regimes:
        u_ss = R["u_ss"]
        R["gamma1"] = gamma1
        R["gamma2"] = gamma2
        R["beta1"] = gamma1 * R["s1_ss"] / max(u_ss, eps)
        R["beta2"] = gamma2 * R["s2_ss"] / max(u_ss, eps)
        R["alpha"] = (R["beta1"] + R["beta2"]) * u_ss

    # 5) Return early or late regime.
    state = state.lower().strip()
    if state not in {"early", "late"}:
        raise ValueError("state must be 'early' or 'late'.")
    chosen = regimes[0] if state == "early" else regimes[1]

    out = (float(chosen["alpha"]), float(chosen["beta1"]), float(chosen["beta2"]), float(gamma1), float(gamma2))
    if return_debug:
        return out, regimes
    return out

def isoform_velocity(u, s1, s2, alpha, beta1, beta2, gamma1, gamma2):
    du  = alpha - (beta1 + beta2) * u
    ds1 = beta1 * u - gamma1 * s1
    ds2 = beta2 * u - gamma2 * s2
    return du, ds1, ds2

def extrapolate_state(u, s1, s2, du, ds1, ds2, dt=1.0, clip_nonneg=True):
    u2  = u  + dt * du
    s12 = s1 + dt * ds1
    s22 = s2 + dt * ds2
    if clip_nonneg:
        u2  = np.maximum(u2, 0.0)
        s12 = np.maximum(s12, 0.0)
        s22 = np.maximum(s22, 0.0)
    return u2, s12, s22

# Simulation 3 functions
def _normalize_sigma(
    sigma: float | Tuple[float, float, float],
) -> np.ndarray:
    """Normalize sigma input to a length-3 non-negative array."""
    if np.isscalar(sigma):
        sig = np.array([float(sigma)] * 3, dtype=float)
    else:
        sig = np.asarray(sigma, dtype=float)
        if sig.shape != (3,):
            raise ValueError("sigma must be a scalar or a length-3 tuple")

    if np.any(~np.isfinite(sig)) or np.any(sig < 0.0):
        raise ValueError("all sigma values must be finite and non-negative")
    return sig

def _gaussian_from_mean(
    mean: np.ndarray,
    sigma: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample Gaussian noise around `mean` with std `sigma`."""
    mean = np.asarray(mean, dtype=float)
    if mean.ndim != 1:
        raise ValueError("mean must be a 1D array")

    noise = rng.normal(loc=0.0, scale=float(sigma), size=mean.size)
    return mean + noise

def add_gaussian_noise(
    u: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    sigma: float | Tuple[float, float, float],
    seed: int | None = None,
    clip_nonneg: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Add Gaussian noise to (u, s1, s2) using inputs as means."""
    u = np.asarray(u, dtype=float)
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)

    if u.ndim != 1 or s1.ndim != 1 or s2.ndim != 1:
        raise ValueError("u, s1, and s2 must be 1D arrays")
    if not (u.size == s1.size == s2.size):
        raise ValueError("u, s1, and s2 must have the same length")

    sig = _normalize_sigma(sigma)
    rng = np.random.default_rng(seed)

    u_g = _gaussian_from_mean(u, float(sig[0]), rng)
    s1_g = _gaussian_from_mean(s1, float(sig[1]), rng)
    s2_g = _gaussian_from_mean(s2, float(sig[2]), rng)

    if clip_nonneg:
        u_g = np.maximum(u_g, 0.0)
        s1_g = np.maximum(s1_g, 0.0)
        s2_g = np.maximum(s2_g, 0.0)

    return u_g, s1_g, s2_g

# Simulation 4 functions

def add_random_dropout(
    u: np.ndarray,
    s1: np.ndarray,
    s2: np.ndarray,
    target_zero_proportions: Tuple[float, float, float],
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Randomly set non-zero entries to zero to reach target zero proportions.

    Args:
        u, s1, s2: Input 1D arrays.
        target_zero_proportions: Length-3 tuple of target zero fractions for
            (u, s1, s2), each in [0, 1].
        seed: Random seed for reproducibility.

    Returns:
        (u_out, s1_out, s2_out) after random drop-out.
    """
    u = np.asarray(u).copy()
    s1 = np.asarray(s1).copy()
    s2 = np.asarray(s2).copy()

    if u.ndim != 1 or s1.ndim != 1 or s2.ndim != 1:
        raise ValueError("u, s1, and s2 must be 1D arrays")
    if not (u.size == s1.size == s2.size):
        raise ValueError("u, s1, and s2 must have the same length")

    target = np.asarray(target_zero_proportions, dtype=float)
    if target.shape != (3,):
        raise ValueError("target_zero_proportions must be a length-3 tuple")
    if np.any(~np.isfinite(target)) or np.any(target < 0.0) or np.any(target > 1.0):
        raise ValueError("target_zero_proportions values must be finite and in [0, 1]")

    rng = np.random.default_rng(seed)

    def _apply(arr: np.ndarray, name: str, target_prop: float) -> np.ndarray:
        arr_out = arr.copy()
        n = arr_out.size
        is_zero = arr_out == 0
        current_zero_count = int(np.sum(is_zero))
        current_zero_prop = current_zero_count / float(n)
        target_zero_count = int(np.round(target_prop * n))

        if current_zero_count >= target_zero_count:
            warnings.warn(
                f"{name}: current zero proportion is {current_zero_prop:.3f}, "
                f"which already satisfies target {target_prop:.3f}.",
                stacklevel=2,
            )
            return arr_out

        non_zero_idx = np.flatnonzero(~is_zero)
        n_to_drop = target_zero_count - current_zero_count
        n_to_drop = min(n_to_drop, non_zero_idx.size)
        if n_to_drop > 0:
            drop_idx = rng.choice(non_zero_idx, size=n_to_drop, replace=False)
            arr_out[drop_idx] = 0
        return arr_out

    u_out = _apply(u, "u", float(target[0]))
    s1_out = _apply(s1, "s1", float(target[1]))
    s2_out = _apply(s2, "s2", float(target[2]))
    return u_out, s1_out, s2_out

# Simulation 5 functions

def _sample_counts_from_mean(
    mu: np.ndarray,
    rng: np.random.Generator,
    *,
    dist: str = "nb",
    theta: float = 20.0,
) -> np.ndarray:
    """
    Sample integer counts given mean `mu` (>=0).

    dist:
      - "poisson": Poisson(mu)
      - "nb": Negative binomial via Gamma-Poisson mixture with dispersion theta
              Var = mu + mu^2/theta.  Larger theta -> closer to Poisson.

    Returns:
      int array with same shape as mu.
    """
    mu = np.asarray(mu, dtype=float)
    mu = np.clip(mu, 0.0, None)

    dist = dist.lower().strip()
    if dist == "poisson":
        return rng.poisson(mu).astype(int)

    if dist == "nb":
        th = float(theta)
        if not np.isfinite(th) or th <= 0.0:
            raise ValueError("theta must be finite and > 0 for NB sampling")

        # Gamma-Poisson mixture: lambda ~ Gamma(shape=theta, scale=mu/theta), y ~ Poisson(lambda)
        # Handle mu==0 safely.
        lam = rng.gamma(shape=th, scale=np.where(mu > 0.0, mu / th, 0.0))
        return rng.poisson(lam).astype(int)

    raise ValueError("dist must be 'poisson' or 'nb'")


def simulate_two_phase_snapshot_counts(
    *,
    dt: float,
    proportion: float,
    # Phase 1 params
    alpha1: float,
    beta1_1: float,
    beta2_1: float,
    gamma1_1: float,
    gamma2_1: float,
    # Phase 2 params
    alpha2: float,
    beta1_2: float,
    beta2_2: float,
    gamma1_2: float,
    gamma2_2: float,
    # Snapshot sampling params
    n_cells: int,
    p1: float,
    p2: float,
    steady_window1: float,
    steady_window2: float,
    method: str = "analytic",
    # Count sampling params
    count_dist: str = "nb",              # "nb" or "poisson"
    theta: float | Tuple[float, float, float] = 20.0,  # NB dispersion (scalar or per (u,s1,s2))
    size_factor_mean: float | None = None,  # if set, sample per-cell lognormal size factors (mean approx = this)
    size_factor_cv: float = 0.25,           # CV of size factors (lognormal)
    seed: int | None = None,
    return_means: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float] | Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, Tuple[np.ndarray, np.ndarray, np.ndarray]
]:
    """
    Use your existing 2-phase deterministic trajectory + snapshot sampling logic,
    but output *discrete* counts (u, s1, s2).

    Returns:
      u_ct, s1_ct, s2_ct, latent_t, group_label, steady_t1, steady_t2
    If return_means=True, also returns (u_mean, s1_mean, s2_mean) used for count sampling.
    """
    # 1) Deterministic trajectory (means over time)
    t, u, s1, s2, steady_t1, steady_t2 = simulate_two_phase_trajectory(
        dt=dt,
        proportion=proportion,
        alpha1=alpha1,
        beta1_1=beta1_1,
        beta2_1=beta2_1,
        gamma1_1=gamma1_1,
        gamma2_1=gamma2_1,
        alpha2=alpha2,
        beta1_2=beta1_2,
        beta2_2=beta2_2,
        gamma1_2=gamma1_2,
        gamma2_2=gamma2_2,
        method=method,
    )

    # 2) Sample snapshot cells along that trajectory (continuous means per cell)
    # Use a separate seed stream for time sampling vs count sampling for reproducibility.
    seed_times = seed
    seed_counts = None if seed is None else int(seed) + 1

    u_mean, s1_mean, s2_mean, latent_t, group_label = sample_real_case_cells(
        t=t,
        u=u,
        s1=s1,
        s2=s2,
        n_cells=n_cells,
        p1=p1,
        p2=p2,
        steady_t1=steady_t1,
        steady_t2=steady_t2,
        steady_window1=steady_window1,
        steady_window2=steady_window2,
        seed=seed_times,
    )

    # 3) Optional per-cell size factors (library sizes)
    rng = np.random.default_rng(seed_counts)
    if size_factor_mean is None:
        sf = np.ones(n_cells, dtype=float)
    else:
        if size_factor_mean <= 0.0:
            raise ValueError("size_factor_mean must be > 0 if provided")
        if size_factor_cv < 0.0:
            raise ValueError("size_factor_cv must be >= 0")

        # Lognormal parameterization so the mean is approximately size_factor_mean.
        # mean = exp(mu + 0.5*sigma^2), CV^2 = exp(sigma^2)-1
        sigma2 = math.log(1.0 + float(size_factor_cv) ** 2)
        sigma = math.sqrt(sigma2)
        mu = math.log(float(size_factor_mean)) - 0.5 * sigma2
        sf = rng.lognormal(mean=mu, sigma=sigma, size=n_cells).astype(float)

    u_mean = np.clip(u_mean * sf, 0.0, None)
    s1_mean = np.clip(s1_mean * sf, 0.0, None)
    s2_mean = np.clip(s2_mean * sf, 0.0, None)

    # 4) Count sampling
    if isinstance(theta, (tuple, list, np.ndarray)):
        if len(theta) != 3:
            raise ValueError("theta must be a scalar or a length-3 tuple for (u, s1, s2)")
        th_u, th_s1, th_s2 = (float(theta[0]), float(theta[1]), float(theta[2]))
    else:
        th_u = th_s1 = th_s2 = float(theta)

    u_ct = _sample_counts_from_mean(u_mean, rng, dist=count_dist, theta=th_u)
    s1_ct = _sample_counts_from_mean(s1_mean, rng, dist=count_dist, theta=th_s1)
    s2_ct = _sample_counts_from_mean(s2_mean, rng, dist=count_dist, theta=th_s2)

    if return_means:
        return u_ct, s1_ct, s2_ct, latent_t, group_label, float(steady_t1), float(steady_t2), (u_mean, s1_mean, s2_mean)
    return u_ct, s1_ct, s2_ct, latent_t, group_label, float(steady_t1), float(steady_t2)


def _nonzero_median(x: np.ndarray, min_n: int = 5) -> float:
    """Median over x[x>0] if enough nonzeros exist; otherwise fallback to overall median."""
    x = np.asarray(x, dtype=float)
    nz = x[x > 0.0]
    if nz.size >= int(min_n):
        return float(np.median(nz))
    if x.size == 0:
        return 0.0
    return float(np.median(x))


def fit_isoform_switching_velocyto_robust(
    u_obs: np.ndarray,
    s1_obs: np.ndarray,
    s2_obs: np.ndarray,
    *,
    state: str = "late",           # "early" (low s1 fraction) or "late" (high s1 fraction)
    steady_frac: float = 0.10,     # fraction per regime used as "steady representatives"
    knn_k: int = 20,
    assume_equal_gamma: bool = True,
    gamma_equal_value: float = 1.0,
    gamma_scale_mode: str = "geometric_mean_1",  # only used when assume_equal_gamma=False
    # Robustness knobs (new)
    pseudo_count: float = 1.0,                 # stabilizes fraction under zeros
    coord_clip_q: Tuple[float, float] = (0.02, 0.98),  # winsorize isoform fraction for robust k-means centers
    coord_require_both_isoforms: bool = True,  # fit regime centers on cells with s1>0 and s2>0 if possible
    min_nonzero_dims: int = 2,                 # candidate steady cells must have >= this many nonzero dims among (u,s1,s2)
    zero_penalty: float = 2.0,                 # penalize (u,s1,s2) zero-rich cells during density selection
    use_log1p_density: bool = True,            # compute density in log1p-space for stability
    seed: int = 0,
    return_debug: bool = False,
):
    """
    Dropout-robust variant of the original fitter.

    Major changes vs your current version:
      - Regime discovery uses a pseudocount-smoothed isoform fraction and clipped centers to reduce dropout domination.
      - "Steady representatives" are selected by kNN density in log1p space with an explicit penalty
        for cells with many zeros (prevents the origin/axes dropout pile from being labeled 'steady').
      - Steady-state (u_ss, s1_ss, s2_ss) are computed as medians over nonzero values when possible.

    Returns:
      alpha, beta1, beta2, gamma1, gamma2   (for requested `state`)
    If return_debug=True, returns ((params), regimes_debug)
    """
    u = np.asarray(u_obs, dtype=float)
    s1 = np.asarray(s1_obs, dtype=float)
    s2 = np.asarray(s2_obs, dtype=float)
    if u.shape != s1.shape or u.shape != s2.shape:
        raise ValueError("u_obs, s1_obs, s2_obs must have the same shape.")
    if u.size < 50:
        raise ValueError("Need at least ~50 cells for stable estimation.")

    # Clip negatives defensively (shouldn't appear for counts, but safe)
    u = np.maximum(u, 0.0)
    s1 = np.maximum(s1, 0.0)
    s2 = np.maximum(s2, 0.0)

    eps = 1e-8
    pc = float(pseudo_count)
    if pc <= 0.0 or not np.isfinite(pc):
        raise ValueError("pseudo_count must be finite and > 0")

    # --- 1) Robust regime coordinate: smoothed isoform fraction in [0,1]
    f_all = (s1 + pc) / (s1 + s2 + 2.0 * pc)

    # Use a subset to fit robust centers (avoid dropout driven 0/1 extremes dominating kmeans).
    mask_fit = (s1 + s2) > 0.0
    if coord_require_both_isoforms:
        mask_fit = mask_fit & (s1 > 0.0) & (s2 > 0.0)

    f_fit = f_all[mask_fit]
    if f_fit.size < 20:
        # Relax if too few usable points
        f_fit = f_all[(s1 + s2) > 0.0]
    if f_fit.size < 20:
        f_fit = f_all  # last resort

    qlo, qhi = coord_clip_q
    if not (0.0 <= qlo < qhi <= 1.0):
        raise ValueError("coord_clip_q must satisfy 0 <= qlo < qhi <= 1")
    lo = float(np.quantile(f_fit, qlo))
    hi = float(np.quantile(f_fit, qhi))
    f_fit_clip = np.clip(f_fit, lo, hi)

    # k-means on clipped coordinate (your existing helper)
    labels_raw, centers = _kmeans_1d(f_fit_clip, k=2, seed=seed)
    order = np.argsort(centers)
    centers = centers[order]  # centers[0] = early (low fraction), centers[1]=late

    # Assign all cells to nearest center
    d = np.abs(f_all[:, None] - centers[None, :])
    labels = d.argmin(axis=1).astype(int)

    # --- 2) Select steady representatives per regime robustly
    regimes = []
    for reg in [0, 1]:
        mask = labels == reg
        if int(mask.sum()) < 10:
            # fallback: if clustering degenerates, use all points (should be rare)
            mask = np.ones_like(labels, dtype=bool)

        idx_reg = np.where(mask)[0]

        # Candidate steady cells must not be "too empty" (dropout pile)
        nz_dims = (
            (u[idx_reg] > 0.0).astype(int)
            + (s1[idx_reg] > 0.0).astype(int)
            + (s2[idx_reg] > 0.0).astype(int)
        )
        cand = nz_dims >= int(min_nonzero_dims)
        if int(cand.sum()) < max(10, int(0.2 * idx_reg.size)):
            # Too strict: relax to "at least one molecule in any layer"
            cand = (u[idx_reg] + s1[idx_reg] + s2[idx_reg]) > 0.0

        idx_cand = idx_reg[cand]
        if idx_cand.size == 0:
            idx_cand = idx_reg

        # Density in (u,s1,s2) but in log1p space for stability, plus zero-penalty
        if idx_cand.size <= 10:
            idx_sel = idx_cand
        else:
            if use_log1p_density:
                pts = np.stack(
                    [np.log1p(u[idx_cand]), np.log1p(s1[idx_cand]), np.log1p(s2[idx_cand])],
                    axis=1,
                )
            else:
                pts = np.stack([u[idx_cand], s1[idx_cand], s2[idx_cand]], axis=1)

            dens = _knn_density_scores(pts, k=knn_k)  # lower = denser

            nz = (
                (u[idx_cand] > 0.0).astype(int)
                + (s1[idx_cand] > 0.0).astype(int)
                + (s2[idx_cand] > 0.0).astype(int)
            )
            dens = dens + float(zero_penalty) * (3 - nz)

            k_sel = max(10, int(np.ceil(float(steady_frac) * idx_reg.size)))
            k_sel = min(k_sel, int(idx_cand.size))
            idx_sel = idx_cand[np.argsort(dens)[:k_sel]]

        # Robust steady-state medians (prefer nonzero medians)
        u_ss = _nonzero_median(u[idx_sel], min_n=5)
        s1_ss = _nonzero_median(s1[idx_sel], min_n=5)
        s2_ss = _nonzero_median(s2[idx_sel], min_n=5)

        regimes.append(
            dict(
                reg=int(reg),
                f_center=float(centers[reg]),
                u_ss=float(u_ss),
                s1_ss=float(s1_ss),
                s2_ss=float(s2_ss),
                n_cells=int(mask.sum()),
                n_steady=int(idx_sel.size),
            )
        )

    # --- 3) Gamma scale (same as your original logic)
    if assume_equal_gamma:
        gamma1 = gamma2 = float(gamma_equal_value)
    else:
        # Enforce alpha invariance across regimes:
        # alpha = gamma1*s1_ss + gamma2*s2_ss  (steady-state identity)
        A, B = regimes[0], regimes[1]
        denom = A["s1_ss"] - B["s1_ss"]
        if np.isclose(denom, 0.0):
            gamma_ratio = 1.0
        else:
            gamma_ratio = (B["s2_ss"] - A["s2_ss"]) / denom  # gamma1/gamma2
        gamma_ratio = float(gamma_ratio)
        if (not np.isfinite(gamma_ratio)) or gamma_ratio <= 0.0:
            gamma_ratio = 1.0

        if gamma_scale_mode == "geometric_mean_1":
            gamma1 = math.sqrt(gamma_ratio)
            gamma2 = 1.0 / gamma1
        elif gamma_scale_mode == "gamma2_to_1":
            gamma2 = 1.0
            gamma1 = gamma_ratio
        else:
            raise ValueError("Unknown gamma_scale_mode.")

    # --- 4) Compute (alpha, beta1, beta2) per regime using steady-state algebra
    for R in regimes:
        u_ss = max(float(R["u_ss"]), eps)
        s1_ss = float(R["s1_ss"])
        s2_ss = float(R["s2_ss"])

        R["gamma1"] = float(gamma1)
        R["gamma2"] = float(gamma2)

        # Steady-state identities:
        # beta1 = gamma1*s1_ss/u_ss, beta2 = gamma2*s2_ss/u_ss
        # alpha = gamma1*s1_ss + gamma2*s2_ss
        R["beta1"] = float(gamma1) * s1_ss / u_ss
        R["beta2"] = float(gamma2) * s2_ss / u_ss
        R["alpha"] = float(gamma1) * s1_ss + float(gamma2) * s2_ss

    state = state.lower().strip()
    if state not in {"early", "late"}:
        raise ValueError("state must be 'early' or 'late'.")
    chosen = regimes[0] if state == "early" else regimes[1]

    out = (float(chosen["alpha"]), float(chosen["beta1"]), float(chosen["beta2"]), float(gamma1), float(gamma2))
    if return_debug:
        return out, regimes
    return out

# Simulation 1: Ideally case
dt = 0.1
steady_t1 = steady_state_time(
    proportion=0.99,
    alpha=2.0, beta1=1.6, beta2=1.4, gamma1=1, gamma2=1,
)
## Phase 1: t = 0..steady_t1
t1 = time_grid(0.0, steady_t1, dt)
u1, s1_1, s2_1 = simulate_two_isoforms(
    t1,
    alpha=2.0, beta1=1.6, beta2=1.4, gamma1=1, gamma2=1,
    u0=0.0, s10=0.0, s20=0.0,
    method="analytic",
)
### estimation (relative to gamma1 = gamma2 = 1)
beta1 = s1_1[-1] / u1[-1]
beta2 = s2_1[-1] / u1[-1]
alpha = s1_1[-1] + s2_1[-1]
print(f"Estimated parameters at t={steady_t1:.1f}: alpha={alpha:.2f}, beta1={beta1:.2f}, beta2={beta2:.2f}")

## Phase 2: t = steady_t1..steady_t2
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
### estimation (relative to gamma1 = gamma2 = 1)
beta1 = s1_2[-1] / u2[-1]
beta2 = s2_2[-1] / u2[-1]
alpha = s1_2[-1] + s2_2[-1]
print(f"Estimated parameters at t={steady_t1 + steady_t2:.1f}: alpha={alpha:.2f}, beta1={beta1:.2f}, beta2={beta2:.2f}")

### Concatenate, dropping the duplicated time point at t=10 in phase 2
t = np.concatenate([t1, t2[1:]])
u = np.concatenate([u1, u2[1:]])
s1 = np.concatenate([s1_1, s1_2[1:]])
s2 = np.concatenate([s2_1, s2_2[1:]])

### 3D static plot with Matplotlib
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u, s1, s2, linewidth=2.0)
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

### 3D interactive plot with Plotly
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

# Simulation 2: Real case
N = 1500
p1 = 0.45
p2 = 0.4
steady_window1 = 1.0
steady_window2 = 1.0

alpha1_true = 2
beta1_1_true = 1.6
beta2_1_true = 1.4
gamma1_1_true = 1
gamma2_1_true = 1

alpha2_true = 2
beta1_2_true = 2.4
beta2_2_true = 0.8
gamma1_2_true = 1
gamma2_2_true = 1

real_t, real_u, real_s1, real_s2, real_steady_t1, real_steady_t2 = simulate_two_phase_trajectory(
	dt=0.1,
	proportion=0.99,
	alpha1=alpha1_true,
	beta1_1=beta1_1_true,
	beta2_1=beta2_1_true,
	gamma1_1=gamma1_1_true,
	gamma2_1=gamma2_1_true,
	alpha2=alpha2_true,
	beta1_2=beta1_2_true,
	beta2_2=beta2_2_true,
	gamma1_2=gamma1_2_true,
	gamma2_2=gamma2_2_true,
	method="analytic",
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(real_u, real_s1, real_s2, linewidth=2.0)
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

u_obs, s1_obs, s2_obs, latent_t, group_label = sample_real_case_cells(
	t=real_t,
	u=real_u,
	s1=real_s1,
	s2=real_s2,
	n_cells=N,
	p1=p1,
	p2=p2,
	steady_t1=real_steady_t1,
	steady_t2=real_steady_t2,
	steady_window1=steady_window1,
	steady_window2=steady_window2,
	seed=7,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u_obs, s1_obs, s2_obs, 'o')
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

alpha_early, beta1_early, beta2_early, gamma1_early, gamma2_early = fit_isoform_switching_velocyto(
	u_obs=u_obs, 
	s1_obs=s1_obs, 
	s2_obs=s2_obs, 
	state="early",
    assume_equal_gamma=False,  # just to see what it does in the ideal case
	)
alpha_late, beta1_late, beta2_late, gamma1_late, gamma2_late = fit_isoform_switching_velocyto(
	u_obs=u_obs, 
	s1_obs=s1_obs, 
	s2_obs=s2_obs, 
	state="late",
    assume_equal_gamma=False,  # just to see what it does in the ideal case
	)
print(
	"Estimated parameters for Phase 1: "
	f"alpha={alpha_early:.2f}, beta1={beta1_early:.2f}, beta2={beta2_early:.2f}, gamma1={gamma1_early:.2f}, gamma2={gamma2_early:.2f}"
)
print(
	"True parameters for Phase 1: "
	f"alpha={alpha1_true:.2f}, beta1={beta1_1_true:.2f}, beta2={beta2_1_true:.2f}, gamma1={gamma1_1_true:.2f}, gamma2={gamma2_1_true:.2f}"
)
print(
	"Estimated parameters for Phase 2: "
	f"alpha={alpha_late:.2f}, beta1={beta1_late:.2f}, beta2={beta2_late:.2f}, gamma1={gamma1_late:.2f}, gamma2={gamma2_late:.2f}"
)
print(
	"True parameters for Phase 2: "
	f"alpha={alpha2_true:.2f}, beta1={beta1_2_true:.2f}, beta2={beta2_2_true:.2f}, gamma1={gamma1_2_true:.2f}, gamma2={gamma2_2_true:.2f}"
)

# Simulation 3: Real case + Noise
N = 1500
p1 = 0.45
p2 = 0.4
steady_window1 = 1.0
steady_window2 = 1.0

alpha1_true = 2
beta1_1_true = 1.6
beta2_1_true = 1.4
gamma1_1_true = 1
gamma2_1_true = 1

alpha2_true = 2
beta1_2_true = 2.4
beta2_2_true = 0.8
gamma1_2_true = 1
gamma2_2_true = 1

real_t, real_u, real_s1, real_s2, real_steady_t1, real_steady_t2 = simulate_two_phase_trajectory(
	dt=0.1,
	proportion=0.99,
	alpha1=alpha1_true,
	beta1_1=beta1_1_true,
	beta2_1=beta2_1_true,
	gamma1_1=gamma1_1_true,
	gamma2_1=gamma2_1_true,
	alpha2=alpha2_true,
	beta1_2=beta1_2_true,
	beta2_2=beta2_2_true,
	gamma1_2=gamma1_2_true,
	gamma2_2=gamma2_2_true,
	method="analytic",
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(real_u, real_s1, real_s2, linewidth=2.0)
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

u_obs, s1_obs, s2_obs, latent_t, group_label = sample_real_case_cells(
	t=real_t,
	u=real_u,
	s1=real_s1,
	s2=real_s2,
	n_cells=N,
	p1=p1,
	p2=p2,
	steady_t1=real_steady_t1,
	steady_t2=real_steady_t2,
	steady_window1=steady_window1,
	steady_window2=steady_window2,
	seed=7,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u_obs, s1_obs, s2_obs, 'o')
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

sigma = (0.05, 0.25, 0.25) ### smaller dispersion means larger variance

u_g, s1_g, s2_g = add_gaussian_noise(
		u=u_obs,
		s1=s1_obs,
		s2=s2_obs,
		sigma=sigma,
		seed=7,
	)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u_g, s1_g, s2_g, 'o')
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

alpha_early, beta1_early, beta2_early, gamma1_early, gamma2_early = fit_isoform_switching_velocyto_robust(
	u_obs=u_g, 
	s1_obs=s1_g, 
	s2_obs=s2_g, 
	state="early",
    assume_equal_gamma=False,  # just to see what it does in the ideal case
	)
alpha_late, beta1_late, beta2_late, gamma1_late, gamma2_late = fit_isoform_switching_velocyto_robust(
	u_obs=u_g, 
	s1_obs=s1_g, 
	s2_obs=s2_g, 
	state="late",
    assume_equal_gamma=False,  # just to see what it does in the ideal case
	)
print(
	"Estimated parameters for Phase 1: "
	f"alpha={alpha_early:.2f}, beta1={beta1_early:.2f}, beta2={beta2_early:.2f}, gamma1={gamma1_early:.2f}, gamma2={gamma2_early:.2f}"
)
print(
	"True parameters for Phase 1: "
	f"alpha={alpha1_true:.2f}, beta1={beta1_1_true:.2f}, beta2={beta2_1_true:.2f}, gamma1={gamma1_1_true:.2f}, gamma2={gamma2_1_true:.2f}"
)
print(
	"Estimated parameters for Phase 2: "
	f"alpha={alpha_late:.2f}, beta1={beta1_late:.2f}, beta2={beta2_late:.2f}, gamma1={gamma1_late:.2f}, gamma2={gamma2_late:.2f}"
)
print(
	"True parameters for Phase 2: "
	f"alpha={alpha2_true:.2f}, beta1={beta1_2_true:.2f}, beta2={beta2_2_true:.2f}, gamma1={gamma1_2_true:.2f}, gamma2={gamma2_2_true:.2f}"
)

# Simulation 4: Real case + Noise + Random Dropout
N = 1500
p1 = 0.45
p2 = 0.4
steady_window1 = 1.0
steady_window2 = 1.0

alpha1_true = 2
beta1_1_true = 1.6
beta2_1_true = 1.4
gamma1_1_true = 1
gamma2_1_true = 1

alpha2_true = 2
beta1_2_true = 2.4
beta2_2_true = 0.8
gamma1_2_true = 1
gamma2_2_true = 1

sigma = (0.05, 0.25, 0.25) ### smaller dispersion means larger variance

real_t, real_u, real_s1, real_s2, real_steady_t1, real_steady_t2 = simulate_two_phase_trajectory(
	dt=0.1,
	proportion=0.99,
	alpha1=alpha1_true,
	beta1_1=beta1_1_true,
	beta2_1=beta2_1_true,
	gamma1_1=gamma1_1_true,
	gamma2_1=gamma2_1_true,
	alpha2=alpha2_true,
	beta1_2=beta1_2_true,
	beta2_2=beta2_2_true,
	gamma1_2=gamma1_2_true,
	gamma2_2=gamma2_2_true,
	method="analytic",
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(real_u, real_s1, real_s2, linewidth=2.0)
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

u_obs, s1_obs, s2_obs, latent_t, group_label = sample_real_case_cells(
	t=real_t,
	u=real_u,
	s1=real_s1,
	s2=real_s2,
	n_cells=N,
	p1=p1,
	p2=p2,
	steady_t1=real_steady_t1,
	steady_t2=real_steady_t2,
	steady_window1=steady_window1,
	steady_window2=steady_window2,
	seed=7,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u_obs, s1_obs, s2_obs, 'o')
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

u_g, s1_g, s2_g = add_gaussian_noise(
		u=u_obs,
		s1=s1_obs,
		s2=s2_obs,
		sigma=sigma,
		seed=7,
	)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u_g, s1_g, s2_g, 'o')
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

u_do, s1_do, s2_do = add_random_dropout(
    u_g, s1_g, s2_g,
    target_zero_proportions=(0.2, 0.2, 0.2),
    seed=42,
)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(u_do, s1_do, s2_do, 'o')
ax.set_xlabel("u")
ax.set_ylabel("s1")
ax.set_zlabel("s2")
plt.show()

alpha_early, beta1_early, beta2_early, gamma1_early, gamma2_early = fit_isoform_switching_velocyto_robust(
	u_obs=u_do, 
	s1_obs=s1_do, 
	s2_obs=s2_do, 
	state="early",
    assume_equal_gamma=False,  # just to see what it does in the ideal case
	)
alpha_late, beta1_late, beta2_late, gamma1_late, gamma2_late = fit_isoform_switching_velocyto_robust(
	u_obs=u_do, 
	s1_obs=s1_do, 
	s2_obs=s2_do, 
	state="late",
    assume_equal_gamma=False,  # just to see what it does in the ideal case
	)
print(
	"Estimated parameters for Phase 1: "
	f"alpha={alpha_early:.2f}, beta1={beta1_early:.2f}, beta2={beta2_early:.2f}, gamma1={gamma1_early:.2f}, gamma2={gamma2_early:.2f}"
)
print(
	"True parameters for Phase 1: "
	f"alpha={alpha1_true:.2f}, beta1={beta1_1_true:.2f}, beta2={beta2_1_true:.2f}, gamma1={gamma1_1_true:.2f}, gamma2={gamma2_1_true:.2f}"
)
print(
	"Estimated parameters for Phase 2: "
	f"alpha={alpha_late:.2f}, beta1={beta1_late:.2f}, beta2={beta2_late:.2f}, gamma1={gamma1_late:.2f}, gamma2={gamma2_late:.2f}"
)
print(
	"True parameters for Phase 2: "
	f"alpha={alpha2_true:.2f}, beta1={beta1_2_true:.2f}, beta2={beta2_2_true:.2f}, gamma1={gamma1_2_true:.2f}, gamma2={gamma2_2_true:.2f}"
)

# Simulation 5: Real case (count matrix) + Noise + Random Dropout + Robust velocyto
N = 1500
p1 = 0.45
p2 = 0.40
steady_window1 = 1.0
steady_window2 = 1.0

alpha1_true = 2.0
beta1_1_true = 1.6
beta2_1_true = 1.4
gamma1_1_true = 1.0
gamma2_1_true = 1.0

alpha2_true = 2.0
beta1_2_true = 2.4
beta2_2_true = 0.8
gamma1_2_true = 1.0
gamma2_2_true = 1.0

u_ct, s1_ct, s2_ct, latent_t, group_label, steady_t1, steady_t2 = simulate_two_phase_snapshot_counts(
        dt=0.1,
        proportion=0.99,
        alpha1=alpha1_true,
        beta1_1=beta1_1_true,
        beta2_1=beta2_1_true,
        gamma1_1=gamma1_1_true,
        gamma2_1=gamma2_1_true,
        alpha2=alpha2_true,
        beta1_2=beta1_2_true,
        beta2_2=beta2_2_true,
        gamma1_2=gamma1_2_true,
        gamma2_2=gamma2_2_true,
        n_cells=N,
        p1=p1,
        p2=p2,
        steady_window1=steady_window1,
        steady_window2=steady_window2,
        method="analytic",
        count_dist=count_dist,   # "nb" or "poisson"
        theta=theta,             # NB dispersion (larger -> closer to Poisson)
        size_factor_mean=None,   # set e.g. 5.0 to add library size variability
        size_factor_cv=0.25,
        seed=seed,
        return_means=False,
    )
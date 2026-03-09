
import numpy as np
from scipy.special import gammaln, digamma, polygamma, logsumexp


def _regularized_proportions(s, n, pseudo=0.5):
    denom = n[:, None] + pseudo * s.shape[1]
    p = (s + pseudo) / np.where(denom == 0, 1.0, denom)
    zero = n == 0
    if np.any(zero):
        p[zero] = 1.0 / s.shape[1]
    return p


def _squared_distance_to_centers(P, centers):
    P_norm = np.sum(P * P, axis=1, keepdims=True)
    C_norm = np.sum(centers * centers, axis=1)[None, :]
    d = P_norm + C_norm - 2.0 * P @ centers.T
    np.maximum(d, 0.0, out=d)
    return d


def _kmeans_plus_plus(P, R, rng):
    n = P.shape[0]
    centers = np.empty((R, P.shape[1]), dtype=float)
    centers[0] = P[rng.integers(n)]
    closest_dist_sq = np.sum((P - centers[0]) ** 2, axis=1)

    for c in range(1, R):
        total = closest_dist_sq.sum()
        if (not np.isfinite(total)) or total <= 0:
            idx = rng.integers(n)
        else:
            idx = rng.choice(n, p=closest_dist_sq / total)
        centers[c] = P[idx]
        closest_dist_sq = np.minimum(
            closest_dist_sq,
            np.sum((P - centers[c]) ** 2, axis=1),
        )
    return centers


def _kmeans_init(P, R, rng, max_init_cells=20000, kmeans_iter=15):
    n = P.shape[0]
    if n <= R:
        idx = np.arange(n)
        if n < R:
            idx = np.concatenate([idx, rng.choice(n, size=R - n, replace=True)])
        centers = P[idx].copy()
        labels = np.argmin(_squared_distance_to_centers(P, centers), axis=1)
        return centers, labels

    P_sub = P[rng.choice(n, size=max_init_cells, replace=False)] if n > max_init_cells else P
    centers = _kmeans_plus_plus(P_sub, R, rng)

    for _ in range(kmeans_iter):
        lab_sub = np.argmin(_squared_distance_to_centers(P_sub, centers), axis=1)
        new_centers = centers.copy()
        for r in range(R):
            mask = lab_sub == r
            if np.any(mask):
                new_centers[r] = P_sub[mask].mean(axis=0)
            else:
                new_centers[r] = P_sub[rng.integers(P_sub.shape[0])]
        if np.allclose(new_centers, centers):
            centers = new_centers
            break
        centers = new_centers

    labels = np.argmin(_squared_distance_to_centers(P, centers), axis=1)
    return centers, labels


def _init_kappa_from_props(P, n, pi, default=20.0, kappa_min=1e-2, kappa_max=1e4):
    if P.shape[0] <= 1:
        return float(default)

    mu = np.clip(pi, 1e-12, 1 - 1e-12)
    denom = np.sum(mu * (1.0 - mu))
    if denom <= 0:
        return float(default)

    var = P.var(axis=0, ddof=1)
    c_hat = np.sum(var) / denom

    n_pos = n[n > 0]
    if n_pos.size == 0:
        return float(default)

    a = np.mean(1.0 / n_pos)
    eps = 1e-8
    c_hat = np.clip(c_hat, a + eps, 1.0 - eps)
    denom2 = c_hat - a
    if denom2 <= eps:
        return float(kappa_max)

    kappa = (1.0 - c_hat) / denom2
    return float(np.clip(kappa, kappa_min, kappa_max))


def _initialize_parameters(s, R, rng, pseudo=0.5):
    N, K = s.shape
    totals = s.sum(axis=1).astype(float)
    informative = totals > 0

    if not np.any(informative):
        tau = np.full(R, 1.0 / R)
        alpha = np.full((R, K), 20.0 / K)
        return tau, alpha

    P = _regularized_proportions(s[informative].astype(float), totals[informative], pseudo=pseudo)
    centers, labels = _kmeans_init(P, R, rng)

    tau = np.empty(R, dtype=float)
    pi = np.empty((R, K), dtype=float)
    kappa = np.empty(R, dtype=float)

    global_pi = (s[informative].sum(axis=0) + pseudo) / (totals[informative].sum() + pseudo * K)
    global_kappa = _init_kappa_from_props(
        P,
        totals[informative],
        global_pi,
        default=max(float(np.median(totals[informative])), 20.0),
    )

    informative_idx = np.flatnonzero(informative)
    for r in range(R):
        idx = np.where(labels == r)[0]
        if idx.size == 0:
            pi[r] = centers[r] / centers[r].sum()
            tau[r] = 1.0 / R
            kappa[r] = global_kappa
            continue

        cells = informative_idx[idx]
        counts_r = s[cells].sum(axis=0).astype(float)
        n_r = totals[cells].sum()
        pi[r] = (counts_r + pseudo) / (n_r + pseudo * K)
        tau[r] = idx.size / informative.sum()
        kappa[r] = _init_kappa_from_props(P[idx], totals[cells], pi[r], default=global_kappa)

    tau = np.clip(tau, 1e-8, None)
    tau /= tau.sum()
    alpha = np.clip(kappa[:, None] * pi, 1e-8, None)
    return tau, alpha


def _iter_batches(N, batch_size):
    for start in range(0, N, batch_size):
        yield start, min(start + batch_size, N)


def _dm_logpmf_batch(s_batch, n_batch, alpha):
    a0 = np.sum(alpha)
    return (
        gammaln(n_batch + 1.0)
        - np.sum(gammaln(s_batch + 1.0), axis=1)
        + gammaln(a0)
        - gammaln(n_batch + a0)
        + np.sum(gammaln(s_batch + alpha[None, :]) - gammaln(alpha)[None, :], axis=1)
    )


def _loglikelihood(s, n, tau, alpha, batch_size=8192):
    N, R = s.shape[0], alpha.shape[0]
    log_tau = np.log(np.clip(tau, 1e-300, None))
    total = 0.0

    for start, stop in _iter_batches(N, batch_size):
        sb = s[start:stop]
        nb = n[start:stop]
        log_comp = np.empty((stop - start, R), dtype=float)
        for r in range(R):
            log_comp[:, r] = log_tau[r] + _dm_logpmf_batch(sb, nb, alpha[r])
        total += np.sum(logsumexp(log_comp, axis=1))

    return float(total)


def _e_step(s, n, tau, alpha, batch_size=8192):
    N, R = s.shape[0], alpha.shape[0]
    log_tau = np.log(np.clip(tau, 1e-300, None))
    resp = np.empty((N, R), dtype=float)
    ll = 0.0

    for start, stop in _iter_batches(N, batch_size):
        sb = s[start:stop]
        nb = n[start:stop]
        log_comp = np.empty((stop - start, R), dtype=float)
        for r in range(R):
            log_comp[:, r] = log_tau[r] + _dm_logpmf_batch(sb, nb, alpha[r])
        log_norm = logsumexp(log_comp, axis=1, keepdims=True)
        resp[start:stop] = np.exp(log_comp - log_norm)
        ll += np.sum(log_norm)

    return resp, float(ll)


def _Q_alpha(s, n, w, alpha, batch_size=8192):
    N = s.shape[0]
    a0 = alpha.sum()
    const = gammaln(a0)
    log_gamma_alpha = gammaln(alpha)
    total = 0.0

    for start, stop in _iter_batches(N, batch_size):
        sb = s[start:stop]
        nb = n[start:stop]
        wb = w[start:stop]
        term = const - gammaln(nb + a0) + np.sum(
            gammaln(sb + alpha[None, :]) - log_gamma_alpha[None, :],
            axis=1,
        )
        total += np.dot(wb, term)

    return float(total)


def _alpha_grad_hess_parts(s, n, w, alpha, batch_size=8192):
    N, K = s.shape
    a0 = alpha.sum()
    common_g = 0.0
    common_h = 0.0
    g = np.zeros(K, dtype=float)
    b = np.zeros(K, dtype=float)

    dig_alpha = digamma(alpha)
    tri_alpha = polygamma(1, alpha)
    dig_a0 = digamma(a0)
    tri_a0 = polygamma(1, a0)

    for start, stop in _iter_batches(N, batch_size):
        sb = s[start:stop]
        nb = n[start:stop]
        wb = w[start:stop]

        tmp_g = dig_a0 - digamma(nb + a0)
        tmp_h = tri_a0 - polygamma(1, nb + a0)
        common_g += np.dot(wb, tmp_g)
        common_h += np.dot(wb, tmp_h)

        g += np.sum(wb[:, None] * (digamma(sb + alpha[None, :]) - dig_alpha[None, :]), axis=0)
        b += np.sum(wb[:, None] * (polygamma(1, sb + alpha[None, :]) - tri_alpha[None, :]), axis=0)

    g += common_g
    b = np.minimum(b, -1e-12)
    return g, b, common_h


def _solve_hessian_system(b, c, g):
    inv_b = 1.0 / b
    s1 = np.sum(inv_b)
    s2 = np.sum(inv_b * g)
    denom = 1.0 + c * s1
    return inv_b * g - (c / denom) * inv_b * s2


def _maximize_alpha_newton(s, n, w, alpha0, max_newton=50, tol=1e-8, min_alpha=1e-8, batch_size=8192):
    alpha = np.clip(alpha0.astype(float, copy=True), min_alpha, None)
    if np.sum(w) <= 1e-12:
        return alpha

    q_old = _Q_alpha(s, n, w, alpha, batch_size=batch_size)

    for _ in range(max_newton):
        g, b, c = _alpha_grad_hess_parts(s, n, w, alpha, batch_size=batch_size)
        if np.max(np.abs(g)) < tol:
            break

        step = _solve_hessian_system(b, c, g)
        t = 1.0
        accepted = False
        while t >= 1e-8:
            cand = alpha - t * step
            if np.all(cand > min_alpha):
                q_new = _Q_alpha(s, n, w, cand, batch_size=batch_size)
                if np.isfinite(q_new) and q_new >= q_old - 1e-12:
                    alpha = cand
                    accepted = True
                    if abs(q_new - q_old) <= tol * max(1.0, abs(q_old)):
                        q_old = q_new
                        t = 0.0
                    else:
                        q_old = q_new
                    break
            t *= 0.5
        if not accepted or t == 0.0:
            break

    return np.clip(alpha, min_alpha, None)


def fit_dm_mixture_em(s, R, max_iteration=200, convg_threshold=1e-6, random_state=None, batch_size=8192, verbose=True):
    s = np.asarray(s)
    if s.ndim != 2:
        raise ValueError("s must be a 2D numpy array of shape (n_cells, n_isoforms).")
    if np.any(s < 0):
        raise ValueError("s must contain nonnegative counts.")
    if R < 1:
        raise ValueError("R must be at least 1.")
    if max_iteration < 1:
        raise ValueError("max_iteration must be at least 1.")
    if convg_threshold <= 0:
        raise ValueError("convg_threshold must be positive.")

    if not np.issubdtype(s.dtype, np.integer):
        s = np.rint(s).astype(np.int64)
    else:
        s = s.astype(np.int64, copy=False)

    N, K = s.shape
    n = s.sum(axis=1).astype(float)
    rng = np.random.default_rng(random_state)

    tau, alpha = _initialize_parameters(s, R, rng)
    prev_ll = _loglikelihood(s, n, tau, alpha, batch_size=batch_size)
    if verbose:
        print(f"Initialization log-likelihood: {prev_ll:.10f}")

    for it in range(1, max_iteration + 1):
        resp, _ = _e_step(s, n, tau, alpha, batch_size=batch_size)
        tau_new = np.clip(resp.mean(axis=0), 1e-12, None)
        tau_new /= tau_new.sum()

        alpha_new = np.empty_like(alpha)
        for r in range(R):
            alpha_new[r] = _maximize_alpha_newton(
                s,
                n,
                resp[:, r],
                alpha[r],
                max_newton=50,
                tol=1e-8,
                min_alpha=1e-8,
                batch_size=batch_size,
            )

        ll_new = _loglikelihood(s, n, tau_new, alpha_new, batch_size=batch_size)
        rel_diff = abs(ll_new - prev_ll) / max(1.0, abs(prev_ll))
        if verbose:
            print(f"Iteration {it:3d} | relative likelihood difference = {rel_diff:.6e}")

        tau, alpha, prev_ll = tau_new, alpha_new, ll_new
        if rel_diff < convg_threshold:
            break

    pi = alpha / alpha.sum(axis=1, keepdims=True)
    kappa = alpha.sum(axis=1)
    n_params = R * K + (R - 1)
    bic = -2.0 * prev_ll + n_params * np.log(max(N, 2))
    return {
        "tau_gr": tau,
        "pi_gr": pi,
        "kappa_gr": kappa,
        "data_likelihood": prev_ll,
        "BIC": bic,
        "alpha_gr": alpha,
    }

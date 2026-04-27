import os
import csv
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Tuple, List
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# SECTION 1. GENERATIVE SYSTEM
# ============================================================

@dataclass
class SimParams:
    gamma: float = 1.0
    alpha: float = 1.5
    delta0: float = 0.0
    sigma: float = 0.4
    n_freq: int = 2**14
    w_max: float = 25.0
    z: float = 1.0
    kappa: float = 2.0
    memory_decay: float = 0.15
    memory_gain: float = 1.0
    te_delta_feedback: float = 0.25
    te_kappa_feedback: float = 0.25
    observer_sigma: float = 1.0


ESTIMATORS = [
    "phase", "centroid", "peak", 
    "opt", "opt_sum1", "opt_sum1_nn", "opt_tikhonov", 
    "opt_local", "opt_local_sum1", "opt_local_sum1_nn", "opt_local_tikhonov"
]

PAIRWISE = [
    ("phase", "centroid"), ("phase", "peak"), ("centroid", "peak"),
    ("phase", "opt"), ("centroid", "opt"), ("peak", "opt"),
    ("phase", "opt_sum1"), ("centroid", "opt_sum1"), ("peak", "opt_sum1"),
    ("phase", "opt_sum1_nn"), ("centroid", "opt_sum1_nn"), ("peak", "opt_sum1_nn"),
    ("phase", "opt_tikhonov"), ("centroid", "opt_tikhonov"), ("peak", "opt_tikhonov"),
    ("opt", "opt_sum1"), ("opt", "opt_sum1_nn"), ("opt_sum1", "opt_sum1_nn"),
    ("opt", "opt_tikhonov")
]


def lorentzian_susceptibility(w: np.ndarray, gamma: float, alpha: float) -> np.ndarray:
    return alpha / (w + 1j * gamma)


def gaussian_spectrum(w: np.ndarray, delta0: float, sigma: float) -> np.ndarray:
    return np.exp(-0.5 * ((w - delta0) / sigma) ** 2)


def transmission_function(w: np.ndarray, gamma: float, alpha: float, z: float) -> np.ndarray:
    chi = lorentzian_susceptibility(w, gamma, alpha)
    return np.exp(1j * chi * z)


def time_grid_from_freq(w: np.ndarray) -> np.ndarray:
    dw = w[1] - w[0]
    n = len(w)
    dt = 2 * np.pi / (n * dw)
    return (np.arange(n) - n // 2) * dt


def ifft_shifted(spec: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(spec)))


def gaussian_kernel_sigma_samples(sigma_samples: float, radius_factor: float = 4.0) -> np.ndarray:
    sigma_samples = max(float(sigma_samples), 1e-9)
    radius = max(1, int(np.ceil(radius_factor * sigma_samples)))
    x = np.arange(-radius, radius + 1)
    k = np.exp(-0.5 * (x / sigma_samples) ** 2)
    k /= np.sum(k)
    return k


def observer_filter(signal: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 1e-9:
        return signal
    kernel = gaussian_kernel_sigma_samples(sigma_samples)
    return np.convolve(signal, kernel, mode="same")


def compute_phase_estimator(w: np.ndarray, H: np.ndarray, delta0: float) -> float:
    phase = np.unwrap(np.angle(H))
    dphi_dw = np.gradient(phase, w)
    idx = np.argmin(np.abs(w - delta0))
    return float(dphi_dw[idx])


def compute_centroid_estimator(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray, observer_sigma: float) -> float:
    i_in = observer_filter(np.abs(e_in_t) ** 2, observer_sigma)
    i_out = observer_filter(np.abs(e_out_t) ** 2, observer_sigma)
    in_norm = np.trapezoid(i_in, t)
    out_norm = np.trapezoid(i_out, t)
    if in_norm <= 1e-15 or out_norm <= 1e-15:
        return 0.0
    t_in = np.trapezoid(t * i_in, t) / in_norm
    t_out = np.trapezoid(t * i_out, t) / out_norm
    return float(t_out - t_in)


def compute_peak_estimator(t: np.ndarray, e_in_t: np.ndarray, e_out_t: np.ndarray, observer_sigma: float) -> float:
    i_in = observer_filter(np.abs(e_in_t) ** 2, observer_sigma)
    i_out = observer_filter(np.abs(e_out_t) ** 2, observer_sigma)
    return float(t[np.argmax(i_out)] - t[np.argmax(i_in)])


def solve_atomic_excitation(
    t: np.ndarray,
    e_in_t: np.ndarray,
    gamma: float,
    delta0: float,
    kappa: float,
    memory_decay: float,
    memory_gain: float,
    te_delta_feedback: float,
    te_kappa_feedback: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    dt = t[1] - t[0]
    a = np.zeros_like(e_in_t, dtype=np.complex128)
    te_current = np.zeros_like(t, dtype=np.float64)
    te_memory = np.zeros_like(t, dtype=np.float64)

    for i in range(len(t) - 1):
        eff_delta = delta0 + te_delta_feedback * te_memory[i]
        eff_kappa = kappa * (1.0 + te_kappa_feedback * te_memory[i])
        lam = gamma + 1j * eff_delta

        decay = np.exp(-lam * dt)
        drive = (eff_kappa / lam) * (1.0 - decay) * e_in_t[i]
        a[i + 1] = a[i] * decay + drive

        te_current[i] = abs(e_in_t[i] - a[i])
        te_memory[i + 1] = (1.0 - memory_decay) * te_memory[i] + memory_gain * te_current[i] * abs(dt)

    te_current[-1] = abs(e_in_t[-1] - a[-1])
    excitation = np.abs(a) ** 2
    return excitation, te_current, te_memory


def support_window_mask(weight: np.ndarray, frac: float = 0.01) -> np.ndarray:
    peak = float(np.max(weight)) if weight.size else 0.0
    if peak <= 1e-15:
        return np.zeros_like(weight, dtype=bool)
    return weight >= (frac * peak)


def excitation_reference_estimator(t: np.ndarray, excitation_t: np.ndarray, support_frac: float = 0.01) -> float:
    weight = np.maximum(excitation_t, 0.0)
    mask = support_window_mask(weight, support_frac)
    if not np.any(mask):
        return 0.0
    t_win = t[mask]
    w_win = weight[mask]
    norm = np.trapezoid(w_win, t_win)
    if norm <= 1e-15:
        return 0.0
    return float(np.trapezoid(t_win * w_win, t_win) / norm)


def weighted_width(t: np.ndarray, weight: np.ndarray, support_frac: float = 0.01) -> float:
    w = np.maximum(weight, 0.0)
    mask = support_window_mask(w, support_frac)
    if not np.any(mask):
        return 0.0
    t_win = t[mask]
    w_win = w[mask]
    norm = np.trapezoid(w_win, t_win)
    if norm <= 1e-15:
        return 0.0
    mu = np.trapezoid(t_win * w_win, t_win) / norm
    var = np.trapezoid(((t_win - mu) ** 2) * w_win, t_win) / norm
    return float(np.sqrt(max(var, 0.0)))


def weighted_skew(t: np.ndarray, weight: np.ndarray, support_frac: float = 0.01) -> float:
    w = np.maximum(weight, 0.0)
    mask = support_window_mask(w, support_frac)
    if not np.any(mask):
        return 0.0
    t_win = t[mask]
    w_win = w[mask]
    norm = np.trapezoid(w_win, t_win)
    if norm <= 1e-15:
        return 0.0
    mu = np.trapezoid(t_win * w_win, t_win) / norm
    var = np.trapezoid(((t_win - mu) ** 2) * w_win, t_win) / norm
    sigma = np.sqrt(max(var, 0.0))
    if sigma <= 1e-15:
        return 0.0
    m3 = np.trapezoid(((t_win - mu) ** 3) * w_win, t_win) / norm
    return float(m3 / (sigma ** 3))


def run_single_panel(params: SimParams) -> Dict[str, float]:
    w = np.linspace(-params.w_max, params.w_max, params.n_freq)
    t = time_grid_from_freq(w)

    e_in_w = gaussian_spectrum(w, params.delta0, params.sigma)
    e_in_t = ifft_shifted(e_in_w)

    H = transmission_function(w, params.gamma, params.alpha, params.z)
    e_out_w = e_in_w * H
    e_out_t = ifft_shifted(e_out_w)

    phase_est = compute_phase_estimator(w, H, params.delta0)
    centroid_est = compute_centroid_estimator(t, e_in_t, e_out_t, params.observer_sigma)
    peak_est = compute_peak_estimator(t, e_in_t, e_out_t, params.observer_sigma)

    exc_t, te_current, te_memory = solve_atomic_excitation(
        t,
        e_in_t,
        params.gamma,
        params.delta0,
        params.kappa,
        params.memory_decay,
        params.memory_gain,
        params.te_delta_feedback,
        params.te_kappa_feedback,
    )

    exc_abs = excitation_reference_estimator(t, exc_t)
    i_in = np.abs(e_in_t) ** 2
    in_norm = np.trapezoid(i_in, t)
    t_in = np.trapezoid(t * i_in, t) / in_norm if in_norm > 1e-15 else 0.0
    exc_ref = exc_abs - t_in
    ref_scale = max(abs(exc_ref), 1e-12)

    row = {
        "delta0": params.delta0,
        "memory_decay": params.memory_decay,
        "memory_gain": params.memory_gain,
        "reference_excitation": exc_ref,
        "reference_scale": ref_scale,
        "estimator_phase": phase_est,
        "estimator_centroid": centroid_est,
        "estimator_peak": peak_est,
        "reference_width": weighted_width(t, exc_t),
        "reference_skew": weighted_skew(t, exc_t),
        "output_width": weighted_width(t, np.abs(e_out_t) ** 2),
        "te_current_peak": float(np.max(te_current)),
        "te_memory_peak": float(np.max(te_memory)),
    }
    for name, est in [("phase", phase_est), ("centroid", centroid_est), ("peak", peak_est)]:
        err = est - exc_ref
        row[f"error_{name}"] = err
        row[f"abs_error_{name}"] = abs(err)
        row[f"squared_error_{name}"] = err ** 2
        row[f"normalized_error_{name}"] = err / ref_scale
        row[f"ratio_abs_{name}"] = abs(est / exc_ref) if abs(exc_ref) > 1e-12 else np.nan
    return row


# ============================================================
# SECTION 2. STATISTICAL LAYER
# ============================================================

def mean_var_mse(x: np.ndarray) -> Tuple[float, float, float, float, float]:
    mean_x = float(np.mean(x))
    var_x = float(np.var(x))
    mse_x = float(np.mean(x ** 2))
    rmse_x = float(np.sqrt(mse_x))
    mae_x = float(np.mean(np.abs(x)))
    return mean_x, var_x, mse_x, rmse_x, mae_x


def summarize_estimator(rows: List[Dict[str, float]], estimator_name: str, resonance_delta: float = 1.0) -> Dict[str, float]:
    err = np.array([r[f"error_{estimator_name}"] for r in rows], dtype=float)
    nerr = np.array([r[f"normalized_error_{estimator_name}"] for r in rows], dtype=float)
    est = np.array([r[f"estimator_{estimator_name}"] for r in rows], dtype=float)
    ref = np.array([r["reference_excitation"] for r in rows], dtype=float)
    delta = np.array([r["delta0"] for r in rows], dtype=float)

    bias, var, mse, rmse, mae = mean_var_mse(err)
    nbias, nvar, nmse, nrmse, nmae = mean_var_mse(nerr)

    mask_res = np.abs(delta) <= resonance_delta
    mask_off = np.abs(delta) > resonance_delta
    mse_res = float(np.mean(err[mask_res] ** 2)) if np.any(mask_res) else np.nan
    mse_off = float(np.mean(err[mask_off] ** 2)) if np.any(mask_off) else np.nan
    conc = float(mse_res / mse_off) if np.isfinite(mse_off) and mse_off > 1e-15 else np.nan

    return {
        "estimator": estimator_name,
        "sample_size": int(len(rows)),
        "mean_estimator": float(np.mean(est)),
        "median_estimator": float(np.median(est)),
        "mean_reference": float(np.mean(ref)),
        "bias": bias,
        "variance": var,
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "mean_absolute_error": mae,
        "normalized_bias": nbias,
        "normalized_variance": nvar,
        "normalized_mse": nmse,
        "normalized_rmse": nrmse,
        "normalized_mae": nmae,
        "resonance_mse": mse_res,
        "offresonance_mse": mse_off,
        "resonance_concentration_ratio": conc,
    }


def summarize_pairwise(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for a, b in PAIRWISE:
        diff = np.array([r[f"estimator_{a}"] - r[f"estimator_{b}"] for r in rows], dtype=float)
        mean_d, var_d, mse_d, rmse_d, mae_d = mean_var_mse(diff)
        out.append({
            "estimator_a": a,
            "estimator_b": b,
            "mean_difference": mean_d,
            "difference_variance": var_d,
            "difference_mse": mse_d,
            "difference_rmse": rmse_d,
            "mean_absolute_difference": mae_d,
        })
    return out


def best_estimator_name(summary_by_estimator: Dict[str, Dict[str, float]]) -> str:
    return min(summary_by_estimator.keys(), key=lambda k: summary_by_estimator[k]["mean_squared_error"])


def compute_weight_roughness(delta: np.ndarray, W: np.ndarray) -> float:
    W = np.asarray(W, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    if len(delta) < 2 or W.shape[0] < 2:
        return 0.0
    rough = 0.0
    for j in range(W.shape[1]):
        dw = np.gradient(W[:, j], delta)
        rough += float(np.mean(dw ** 2))
    return rough / max(W.shape[1], 1)


def compute_family_tension(rows: List[Dict[str, float]], estimator_name: str) -> float:
    est = np.array([r[f"estimator_{estimator_name}"] for r in rows], dtype=float)
    phase = np.array([r["estimator_phase"] for r in rows], dtype=float)
    centroid = np.array([r["estimator_centroid"] for r in rows], dtype=float)
    peak = np.array([r["estimator_peak"] for r in rows], dtype=float)
    return float(np.mean(((est - phase) ** 2 + (est - centroid) ** 2 + (est - peak) ** 2) / 3.0))


def compute_pointwise_family_tension(rows: List[Dict[str, float]], estimator_name: str) -> np.ndarray:
    est = np.array([r[f"estimator_{estimator_name}"] for r in rows], dtype=float)
    phase = np.array([r["estimator_phase"] for r in rows], dtype=float)
    centroid = np.array([r["estimator_centroid"] for r in rows], dtype=float)
    peak = np.array([r["estimator_peak"] for r in rows], dtype=float)
    return ((est - phase) ** 2 + (est - centroid) ** 2 + (est - peak) ** 2) / 3.0


def compute_pointwise_weight_roughness(delta: np.ndarray, weights: np.ndarray) -> np.ndarray:
    W = np.asarray(weights, dtype=float)
    delta = np.asarray(delta, dtype=float)
    if W.ndim == 1:
        W = W.reshape(-1, 1)
    if len(delta) < 2 or W.shape[0] < 2:
        return np.zeros(len(delta), dtype=float)
    pointwise = np.zeros(len(delta), dtype=float)
    for j in range(W.shape[1]):
        dw = np.gradient(W[:, j], delta)
        pointwise += dw ** 2
    return pointwise / max(W.shape[1], 1)


def adaptive_te_coefficients(delta: np.ndarray, memory_gain: float, memory_decay: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    delta = np.asarray(delta, dtype=float)
    ad = np.abs(delta)
    center = np.exp(-0.5 * (ad / 0.90) ** 2)
    tail = 1.0 - center
    memory_ratio = memory_gain / (1.0 + memory_gain + memory_decay)
    decay_ratio = memory_decay / (1.0 + memory_decay)

    alpha = 1.05 + 0.90 * center - 0.20 * memory_ratio * tail
    beta = 0.18 + 0.42 * tail + 0.08 * memory_ratio
    gamma = 0.03 + 0.07 * tail + 0.04 * decay_ratio
    return alpha, beta, gamma


def compute_measurement_te(rows: List[Dict[str, float]], estimator_name: str, weights: np.ndarray,
                           alpha: float = 1.0, beta: float = 0.25, gamma: float = 0.05) -> Dict[str, float]:
    delta = np.array([r["delta0"] for r in rows], dtype=float)
    ref_mse_point = np.array([r[f"error_{estimator_name}"] for r in rows], dtype=float) ** 2
    family_tension_point = compute_pointwise_family_tension(rows, estimator_name)
    roughness_point = compute_pointwise_weight_roughness(delta, weights)

    mse = float(np.mean(ref_mse_point))
    family_tension = float(np.mean(family_tension_point))
    roughness = float(np.mean(roughness_point))
    total = alpha * mse + beta * family_tension + gamma * roughness

    memory_gain = float(rows[0].get("memory_gain", 0.0)) if rows else 0.0
    memory_decay = float(rows[0].get("memory_decay", 0.0)) if rows else 0.0
    a_loc, b_loc, c_loc = adaptive_te_coefficients(delta, memory_gain, memory_decay)
    adaptive_total_point = a_loc * ref_mse_point + b_loc * family_tension_point + c_loc * roughness_point

    return {
        "te_reference_mse": mse,
        "te_family_tension": family_tension,
        "te_weight_roughness": roughness,
        "te_total": total,
        "te_alpha_mean": float(np.mean(a_loc)),
        "te_beta_mean": float(np.mean(b_loc)),
        "te_gamma_mean": float(np.mean(c_loc)),
        "te_adaptive_reference_mse": float(np.mean(a_loc * ref_mse_point)),
        "te_adaptive_family_tension": float(np.mean(b_loc * family_tension_point)),
        "te_adaptive_weight_roughness": float(np.mean(c_loc * roughness_point)),
        "te_adaptive_total": float(np.mean(adaptive_total_point)),
    }


def solve_sum_to_one_weights(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = X.shape[1]
    A = np.block([[2.0 * (X.T @ X), np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
    b = np.concatenate([2.0 * (X.T @ y), np.array([1.0])])
    sol = np.linalg.solve(A, b)
    return sol[:n]


def solve_sum_to_one_nonnegative_weights(
    X: np.ndarray,
    y: np.ndarray,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> np.ndarray:
    G = X.T @ X
    c = X.T @ y
    return solve_simplex_qp_from_gram(G, c, max_iter=max_iter, tol=tol)


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).ravel()
    if v.size == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1.0
    ind = np.arange(1, len(v) + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.full_like(v, 1.0 / len(v))
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    w = np.maximum(v - theta, 0.0)
    s = float(np.sum(w))
    return w / s if s > 1e-15 else np.full_like(v, 1.0 / len(v))


def solve_simplex_qp_from_gram(
    G: np.ndarray,
    c: np.ndarray,
    max_iter: int = 250,
    tol: float = 1e-10,
) -> np.ndarray:
    G = np.asarray(G, dtype=float)
    c = np.asarray(c, dtype=float).ravel()
    n = G.shape[0]
    try:
        A = np.block([[2.0 * G, np.ones((n, 1))], [np.ones((1, n)), np.zeros((1, 1))]])
        b = np.concatenate([2.0 * c, np.array([1.0])])
        w = np.linalg.solve(A, b)[:n]
    except np.linalg.LinAlgError:
        w = np.full(n, 1.0 / n)
    w = project_to_simplex(w)

    try:
        L = max(float(2.0 * np.max(np.linalg.eigvalsh(G))), 1e-8)
    except np.linalg.LinAlgError:
        L = max(float(2.0 * np.trace(G)), 1e-8)
    step = 1.0 / L

    for _ in range(max_iter):
        grad = 2.0 * (G @ w - c)
        w_next = project_to_simplex(w - step * grad)
        if np.linalg.norm(w_next - w) < tol:
            return w_next
        w = w_next
    return w

# === TIKHONOV SOLVER (STAGE 9: RESTORED UNITY GAIN) ===
def solve_tikhonov_sum_one(X: np.ndarray, y: np.ndarray, lambda_reg: float = 0.05) -> np.ndarray:
    n = X.shape[1]
    X_std = np.std(X, axis=0) + 1e-9
    X_scaled = X / X_std 

    G = X_scaled.T @ X_scaled + lambda_reg * np.eye(n)
    C = (1.0 / X_std).reshape(n, 1)
    A = np.block([[2.0 * G, C], [C.T, np.zeros((1, 1))]])
    b = np.concatenate([2.0 * (X_scaled.T @ y), np.array([1.0])])
    
    sol = np.linalg.solve(A, b)
    w_scaled = sol[:n]
    return w_scaled / X_std

# === REGIME SPLIT HELPER ===
def split_core_tail(delta_values, threshold=1.05):
    delta_values = np.asarray(delta_values)
    core_mask = np.abs(delta_values) <= threshold
    tail_mask = ~core_mask
    return core_mask, tail_mask


def weighted_design(X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    if sample_weights is None:
        return X, y
    sw = np.sqrt(np.maximum(np.asarray(sample_weights, dtype=float), 0.0))
    return X * sw[:, None], y * sw


def project_sum1(w: np.ndarray) -> np.ndarray:
    s = float(np.sum(w))
    if abs(s) <= 1e-15:
        return np.array([1.0/len(w)] * len(w), dtype=float)
    return np.asarray(w, dtype=float) / s


def compute_delta_conditioned_estimator(
    panel_rows: List[Dict[str, float]],
    mode: str,
    bandwidth: float = 0.85,
) -> Tuple[np.ndarray, np.ndarray]:
    if not panel_rows:
        return np.empty((0, 3)), np.array([], dtype=float)

    delta = np.array([r["delta0"] for r in panel_rows], dtype=float)
    ref = np.array([r["reference_excitation"] for r in panel_rows], dtype=float)
    X = np.column_stack([
        np.array([r["estimator_phase"] for r in panel_rows], dtype=float),
        np.array([r["estimator_centroid"] for r in panel_rows], dtype=float),
        np.array([r["estimator_peak"] for r in panel_rows], dtype=float),
    ])
    memory_gain = float(panel_rows[0].get("memory_gain", 0.0))
    memory_decay = float(panel_rows[0].get("memory_decay", 0.0))
    rho = memory_gain / (1.0 + memory_gain + memory_decay)
    sigma = bandwidth * (1.0 + 0.35 * memory_decay)
    sigma = max(float(sigma), 1e-9)

    K = np.exp(-0.5 * ((delta[:, None] - delta[None, :]) / sigma) ** 2)

    W = np.zeros((len(panel_rows), 3), dtype=float)
    prev_w = None
    Xt = X.T
    for i in range(len(panel_rows)):
        k = K[i]
        Xk = X * k[:, None]
        G = Xt @ Xk
        c = Xt @ (k * ref)

        if mode == "opt_local":
            try:
                w = np.linalg.solve(G + 1e-10 * np.eye(3), c)
            except np.linalg.LinAlgError:
                w, *_ = np.linalg.lstsq(X * np.sqrt(k)[:, None], ref * np.sqrt(k), rcond=None)
        elif mode == "opt_local_sum1":
            try:
                A = np.block([[2.0 * G, np.ones((3, 1))], [np.ones((1, 3)), np.zeros((1, 1))]])
                b = np.concatenate([2.0 * c, np.array([1.0])])
                w = np.linalg.solve(A, b)[:3]
            except np.linalg.LinAlgError:
                w = solve_sum_to_one_weights(X * np.sqrt(k)[:, None], ref * np.sqrt(k))
        elif mode == "opt_local_sum1_nn":
            w = solve_simplex_qp_from_gram(G, c, max_iter=120, tol=1e-9)
        elif mode == "opt_local_tikhonov":
            # === STAGE 9: THE BOUND INFINITE (SUM-TO-ONE UNLEASHED) ===
            # Boundary 1.05, Knee Width 0.30
            # Center = Unleashed Specialist (0.0001), Edge = Protective Guardian (0.85)
            # Weights MUST sum to 1 to prevent tension explosion
            transition = 1.0 / (1.0 + np.exp((abs(delta[i]) - 1.05) / 0.30))
            l_reg = 0.0001 * transition + 0.85 * (1.0 - transition)
            
            w = solve_tikhonov_sum_one(X * np.sqrt(k)[:, None], ref * np.sqrt(k), lambda_reg=l_reg)
            w = project_sum1(w)
        else:
            raise ValueError(f"unknown local mode: {mode}")

        w = np.asarray(w, dtype=float)
        w_state = w if prev_w is None else (1.0 - rho) * w + rho * prev_w

        if mode in ["opt_local_sum1", "opt_local_tikhonov"]:
            w_state = project_sum1(w_state)
        elif mode == "opt_local_sum1_nn":
            w_state = project_to_simplex(w_state)

        W[i] = w_state
        prev_w = w_state

    est = np.sum(X * W, axis=1)
    return W, est


def append_estimator_to_rows(panel_rows: List[Dict[str, float]], est_name: str, est_values: np.ndarray) -> List[Dict[str, float]]:
    out_rows = []
    for r, est_val in zip(panel_rows, est_values):
        new_r = dict(r)
        err = float(est_val - r["reference_excitation"])
        new_r[f"estimator_{est_name}"] = float(est_val)
        new_r[f"error_{est_name}"] = err
        new_r[f"abs_error_{est_name}"] = float(abs(err))
        new_r[f"squared_error_{est_name}"] = float(err ** 2)
        new_r[f"normalized_error_{est_name}"] = float(err / max(r["reference_scale"], 1e-12))
        out_rows.append(new_r)
    return out_rows


def summarize_composite_estimator(panel_rows: List[Dict[str, float]], est_name: str, weights: np.ndarray) -> Dict[str, float]:
    rows = summarize_estimator(panel_rows, est_name)
    rows["estimator"] = est_name
    W = np.asarray(weights, dtype=float)
    if W.ndim == 1:
        W = W.reshape(1, -1)
    rows["weight_phase"] = float(np.mean(W[:, 0]))
    rows["weight_centroid"] = float(np.mean(W[:, 1]))
    rows["weight_peak"] = float(np.mean(W[:, 2]))
    rows["weight_phase_std"] = float(np.std(W[:, 0]))
    rows["weight_centroid_std"] = float(np.std(W[:, 1]))
    rows["weight_peak_std"] = float(np.std(W[:, 2]))
    rows["weight_phase_min"] = float(np.min(W[:, 0]))
    rows["weight_centroid_min"] = float(np.min(W[:, 1]))
    rows["weight_peak_min"] = float(np.min(W[:, 2]))
    rows["weight_phase_max"] = float(np.max(W[:, 0]))
    rows["weight_centroid_max"] = float(np.max(W[:, 1]))
    rows["weight_peak_max"] = float(np.max(W[:, 2]))
    te = compute_measurement_te(panel_rows, est_name, W)
    rows.update(te)
    return rows


def compute_optimal_estimator_for_panel(panel_rows: List[Dict[str, float]]) -> Tuple[Dict[str, float], List[Dict[str, float]], Dict[str, Dict[str, float]]]:
    if not panel_rows:
        return {}, [], {}

    ref = np.array([r["reference_excitation"] for r in panel_rows], dtype=float)
    X = np.column_stack([
        np.array([r["estimator_phase"] for r in panel_rows], dtype=float),
        np.array([r["estimator_centroid"] for r in panel_rows], dtype=float),
        np.array([r["estimator_peak"] for r in panel_rows], dtype=float),
    ])

    weights_opt, *_ = np.linalg.lstsq(X, ref, rcond=None)
    weights_sum1 = solve_sum_to_one_weights(X, ref)
    weights_sum1_nn = solve_sum_to_one_nonnegative_weights(X, ref)
    weights_tikhonov = solve_tikhonov_sum_one(X, ref, lambda_reg=0.05)

    opt = X @ weights_opt
    opt_sum1 = X @ weights_sum1
    opt_sum1_nn = X @ weights_sum1_nn
    opt_tikhonov = X @ weights_tikhonov

    W_local, opt_local = compute_delta_conditioned_estimator(panel_rows, "opt_local")
    W_local_sum1, opt_local_sum1 = compute_delta_conditioned_estimator(panel_rows, "opt_local_sum1")
    W_local_sum1_nn, opt_local_sum1_nn = compute_delta_conditioned_estimator(panel_rows, "opt_local_sum1_nn")
    W_local_tikhonov, opt_local_tikhonov = compute_delta_conditioned_estimator(panel_rows, "opt_local_tikhonov")

    out_rows = append_estimator_to_rows(panel_rows, "opt", opt)
    out_rows = append_estimator_to_rows(out_rows, "opt_sum1", opt_sum1)
    out_rows = append_estimator_to_rows(out_rows, "opt_sum1_nn", opt_sum1_nn)
    out_rows = append_estimator_to_rows(out_rows, "opt_tikhonov", opt_tikhonov)
    
    out_rows = append_estimator_to_rows(out_rows, "opt_local", opt_local)
    out_rows = append_estimator_to_rows(out_rows, "opt_local_sum1", opt_local_sum1)
    out_rows = append_estimator_to_rows(out_rows, "opt_local_sum1_nn", opt_local_sum1_nn)
    out_rows = append_estimator_to_rows(out_rows, "opt_local_tikhonov", opt_local_tikhonov)

    summaries = {
        "opt": summarize_composite_estimator(out_rows, "opt", weights_opt),
        "opt_sum1": summarize_composite_estimator(out_rows, "opt_sum1", weights_sum1),
        "opt_sum1_nn": summarize_composite_estimator(out_rows, "opt_sum1_nn", weights_sum1_nn),
        "opt_tikhonov": summarize_composite_estimator(out_rows, "opt_tikhonov", weights_tikhonov),
        "opt_local": summarize_composite_estimator(out_rows, "opt_local", W_local),
        "opt_local_sum1": summarize_composite_estimator(out_rows, "opt_local_sum1", W_local_sum1),
        "opt_local_sum1_nn": summarize_composite_estimator(out_rows, "opt_local_sum1_nn", W_local_sum1_nn),
        "opt_local_tikhonov": summarize_composite_estimator(out_rows, "opt_local_tikhonov", W_local_tikhonov),
    }

    baseline_summary = {
        "estimator": "opt",
        "weight_phase": float(weights_opt[0]),
        "weight_centroid": float(weights_opt[1]),
        "weight_peak": float(weights_opt[2]),
        "sum1_weight_phase": float(weights_sum1[0]),
        "sum1_weight_centroid": float(weights_sum1[1]),
        "sum1_weight_peak": float(weights_sum1[2]),
        "sum1_nn_weight_phase": float(weights_sum1_nn[0]),
        "sum1_nn_weight_centroid": float(weights_sum1_nn[1]),
        "sum1_nn_weight_peak": float(weights_sum1_nn[2]),
        "tikhonov_weight_phase": float(weights_tikhonov[0]),
        "tikhonov_weight_centroid": float(weights_tikhonov[1]),
        "tikhonov_weight_peak": float(weights_tikhonov[2]),
        "local_weight_phase_mean": float(np.mean(W_local_tikhonov[:, 0])),
        "local_weight_centroid_mean": float(np.mean(W_local_tikhonov[:, 1])),
        "local_weight_peak_mean": float(np.mean(W_local_tikhonov[:, 2])),
        "bias": summaries["opt"]["bias"],
        "variance": summaries["opt"]["variance"],
        "mean_squared_error": summaries["opt"]["mean_squared_error"],
        "root_mean_squared_error": summaries["opt"]["root_mean_squared_error"],
        "mean_absolute_error": summaries["opt"]["mean_absolute_error"],
        "normalized_bias": summaries["opt"]["normalized_bias"],
        "normalized_variance": summaries["opt"]["normalized_variance"],
        "normalized_mse": summaries["opt"]["normalized_mse"],
        "normalized_rmse": summaries["opt"]["normalized_rmse"],
        "normalized_mae": summaries["opt"]["normalized_mae"],
    }
    return baseline_summary, out_rows, summaries


def compute_optimal_estimator_by_memory(base: SimParams, decays: List[float], gains: List[float], detunings: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for md in decays:
        print(f"[progress] optimal rows decay={md:.2f}", flush=True)
        for mg in gains:
            p = SimParams(**{**asdict(base), "memory_decay": md, "memory_gain": mg})
            panel = sweep_detuning_to_panel(detunings, p)
            _, opt_panel, summaries = compute_optimal_estimator_for_panel(panel)
            row = {
                "memory_decay": md,
                "memory_gain": mg,
            }
            for prefix, s in summaries.items():
                row[f"{prefix}_weight_phase"] = s["weight_phase"]
                row[f"{prefix}_weight_centroid"] = s["weight_centroid"]
                row[f"{prefix}_weight_peak"] = s["weight_peak"]
                row[f"{prefix}_bias"] = s["bias"]
                row[f"{prefix}_variance"] = s["variance"]
                row[f"{prefix}_mse"] = s["mean_squared_error"]
                row[f"{prefix}_rmse"] = s["root_mean_squared_error"]
                row[f"{prefix}_mae"] = s["mean_absolute_error"]
                row[f"{prefix}_normalized_bias"] = s["normalized_bias"]
                row[f"{prefix}_normalized_variance"] = s["normalized_variance"]
                row[f"{prefix}_normalized_mse"] = s["normalized_mse"]
                row[f"{prefix}_te_reference_mse"] = s.get("te_reference_mse", s["mean_squared_error"])
                row[f"{prefix}_te_family_tension"] = s.get("te_family_tension", np.nan)
                row[f"{prefix}_te_weight_roughness"] = s.get("te_weight_roughness", np.nan)
                row[f"{prefix}_te_total"] = s.get("te_total", np.nan)
                row[f"{prefix}_te_alpha_mean"] = s.get("te_alpha_mean", np.nan)
                row[f"{prefix}_te_beta_mean"] = s.get("te_beta_mean", np.nan)
                row[f"{prefix}_te_gamma_mean"] = s.get("te_gamma_mean", np.nan)
                row[f"{prefix}_te_adaptive_reference_mse"] = s.get("te_adaptive_reference_mse", np.nan)
                row[f"{prefix}_te_adaptive_family_tension"] = s.get("te_adaptive_family_tension", np.nan)
                row[f"{prefix}_te_adaptive_weight_roughness"] = s.get("te_adaptive_weight_roughness", np.nan)
                row[f"{prefix}_te_adaptive_total"] = s.get("te_adaptive_total", np.nan)
            rows.append(row)
    return rows


def sweep_detuning_to_panel(detunings: np.ndarray, base: SimParams) -> List[Dict[str, float]]:
    return [run_single_panel(SimParams(**{**asdict(base), "delta0": float(d)})) for d in detunings]


def sweep_memory_statistics(base: SimParams, decays: List[float], gains: List[float], detunings: np.ndarray) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for md in decays:
        print(f"[progress] memory stats decay={md:.2f}", flush=True)
        for mg in gains:
            p = SimParams(**{**asdict(base), "memory_decay": md, "memory_gain": mg})
            panel = sweep_detuning_to_panel(detunings, p)
            summary = {name: summarize_estimator(panel, name) for name in ["phase", "centroid", "peak"]}
            _, opt_panel, composite_summaries = compute_optimal_estimator_for_panel(panel)
            summary.update(composite_summaries)
            pairwise = summarize_pairwise(opt_panel)

            row: Dict[str, float] = {
                "memory_decay": md,
                "memory_gain": mg,
                "reference_width_mean": float(np.mean([r["reference_width"] for r in panel])),
                "reference_skew_mean": float(np.mean([r["reference_skew"] for r in panel])),
                "te_current_peak_mean": float(np.mean([r["te_current_peak"] for r in panel])),
                "te_memory_peak_mean": float(np.mean([r["te_memory_peak"] for r in panel])),
            }
            for name in ESTIMATORS:
                s = summary[name]
                row[f"{name}_bias"] = s["bias"]
                row[f"{name}_variance"] = s["variance"]
                row[f"{name}_mse"] = s["mean_squared_error"]
                row[f"{name}_rmse"] = s["root_mean_squared_error"]
                row[f"{name}_mae"] = s["mean_absolute_error"]
                row[f"{name}_normalized_bias"] = s["normalized_bias"]
                row[f"{name}_normalized_variance"] = s["normalized_variance"]
                row[f"{name}_normalized_mse"] = s["normalized_mse"]
                row[f"{name}_resonance_mse"] = s["resonance_mse"]
                row[f"{name}_offresonance_mse"] = s["offresonance_mse"]
                row[f"{name}_resonance_concentration_ratio"] = s["resonance_concentration_ratio"]
            for pair in pairwise:
                tag = f"{pair['estimator_a']}_vs_{pair['estimator_b']}"
                row[f"{tag}_difference_mse"] = pair["difference_mse"]
                row[f"{tag}_difference_rmse"] = pair["difference_rmse"]
                row[f"{tag}_difference_variance"] = pair["difference_variance"]
                row[f"{tag}_mean_absolute_difference"] = pair["mean_absolute_difference"]
            row["best_estimator_by_mse"] = best_estimator_name(summary)
            out.append(row)
    return out


def add_memory_sensitivities(memory_rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    rows = [dict(r) for r in memory_rows]
    metrics = []
    for name in ESTIMATORS:
        metrics.extend([
            f"{name}_bias",
            f"{name}_variance",
            f"{name}_mse",
            f"{name}_resonance_concentration_ratio",
        ])
    for a, b in PAIRWISE:
        metrics.append(f"{a}_vs_{b}_difference_mse")

    decays = sorted({r["memory_decay"] for r in rows})
    for md in decays:
        idxs = [i for i, r in enumerate(rows) if abs(r["memory_decay"] - md) < 1e-12]
        idxs.sort(key=lambda i: rows[i]["memory_gain"])
        gains = np.array([rows[i]["memory_gain"] for i in idxs], dtype=float)
        for metric in metrics:
            vals = np.array([rows[i][metric] for i in idxs], dtype=float)
            grad = np.gradient(vals, gains) if len(gains) > 1 else np.zeros_like(vals)
            for i, g in zip(idxs, grad):
                rows[i][f"d_dmemory_gain__{metric}"] = float(g)
    return rows


def fit_surface_coefficients(memory_rows: List[Dict[str, float]], metric_names: List[str]) -> List[Dict[str, float]]:
    M = np.array([r["memory_gain"] for r in memory_rows], dtype=float)
    L = np.array([r["memory_decay"] for r in memory_rows], dtype=float)
    X = np.column_stack([np.ones_like(M), M, L, M * L])
    out: List[Dict[str, float]] = []
    for metric in metric_names:
        y = np.array([r[metric] for r in memory_rows], dtype=float)
        coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
        yhat = X @ coeffs
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-15 else 1.0
        out.append({
            "metric": metric,
            "intercept": float(coeffs[0]),
            "coef_memory_gain": float(coeffs[1]),
            "coef_memory_decay": float(coeffs[2]),
            "coef_interaction": float(coeffs[3]),
            "r_squared": float(r2),
        })
    return out


def write_csv(path_or_rows, rows_or_path) -> None:
    if isinstance(path_or_rows, str):
        path = path_or_rows
        rows = rows_or_path
    else:
        rows = path_or_rows
        path = rows_or_path
    if not rows:
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(panel: List[Dict[str, float]], memory_rows: List[Dict[str, float]], optimal_rows: List[Dict[str, float]], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    delta = np.array([r["delta0"] for r in panel], dtype=float)
    ref = np.array([r["reference_excitation"] for r in panel], dtype=float)
    phase = np.array([r["estimator_phase"] for r in panel], dtype=float)
    cent = np.array([r["estimator_centroid"] for r in panel], dtype=float)
    peak = np.array([r["estimator_peak"] for r in panel], dtype=float)
    opt = np.array([r["estimator_opt"] for r in panel], dtype=float)
    opt_sum1 = np.array([r["estimator_opt_sum1"] for r in panel], dtype=float)
    opt_sum1_nn = np.array([r["estimator_opt_sum1_nn"] for r in panel], dtype=float)
    opt_local_tikhonov = np.array([r["estimator_opt_local_tikhonov"] for r in panel], dtype=float)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(delta, ref, label="reference_excitation")
    ax.plot(delta, phase, label="estimator_phase")
    ax.plot(delta, cent, label="estimator_centroid")
    ax.plot(delta, peak, label="estimator_peak")
    ax.plot(delta, opt, label="estimator_opt")
    ax.plot(delta, opt_sum1, label="estimator_opt_sum1")
    ax.plot(delta, opt_sum1_nn, label="estimator_opt_sum1_nn")
    ax.plot(delta, opt_local_tikhonov, label="estimator_opt_local_tikhonov")
    ax.set_xlabel("detuning")
    ax.set_ylabel("estimator value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(out_dir, "stat_v19_figure1_estimators_vs_reference.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    for name in ESTIMATORS:
        ax.plot(delta, [r[f"error_{name}"] for r in panel], label=f"error_{name}")
    ax.axhline(0.0, linestyle="--", linewidth=1.0)
    ax.set_xlabel("detuning")
    ax.set_ylabel("signed error")
    ax.grid(True, alpha=0.3)
    ax.legend()
    p = os.path.join(out_dir, "stat_v19_figure2_signed_errors.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    decays = sorted({r["memory_decay"] for r in memory_rows})
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for md in decays:
        sub = sorted([r for r in memory_rows if abs(r["memory_decay"] - md) < 1e-12], key=lambda x: x["memory_gain"])
        g = [r["memory_gain"] for r in sub]
        axes[0, 0].plot(g, [r["phase_bias"] for r in sub], label=f"decay={md:.2f}")
        axes[0, 1].plot(g, [r["phase_variance"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 0].plot(g, [r["phase_mse"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 1].plot(g, [r["phase_vs_centroid_difference_mse"] for r in sub], label=f"decay={md:.2f}")
    axes[0, 0].set_ylabel("phase bias")
    axes[0, 1].set_ylabel("phase variance")
    axes[1, 0].set_ylabel("phase mse")
    axes[1, 1].set_ylabel("phase-centroid divergence mse")
    for ax in axes.flat:
        ax.set_xlabel("memory_gain")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    p = os.path.join(out_dir, "stat_v19_figure3_memory_surfaces_slices.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, ["phase", "centroid", "peak"]):
        for md in decays:
            sub = sorted([r for r in memory_rows if abs(r["memory_decay"] - md) < 1e-12], key=lambda x: x["memory_gain"])
            ax.plot([r["memory_gain"] for r in sub], [r[f"{name}_resonance_concentration_ratio"] for r in sub], label=f"{md:.2f}")
        ax.set_title(f"{name} concentration")
        ax.set_xlabel("memory_gain")
        ax.set_ylabel("resonance/off-res mse")
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="decay", fontsize=8)
    p = os.path.join(out_dir, "stat_v19_figure4_resonance_concentration.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axes = plt.subplots(3, 2, figsize=(11, 10), constrained_layout=True)
    for md in decays:
        sub = sorted([r for r in optimal_rows if abs(r["memory_decay"] - md) < 1e-12], key=lambda x: x["memory_gain"])
        g = [r["memory_gain"] for r in sub]
        axes[0, 0].plot(g, [r["opt_weight_phase"] for r in sub], label=f"decay={md:.2f}")
        axes[0, 1].plot(g, [r["opt_mse"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 0].plot(g, [r["opt_tikhonov_weight_phase"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 1].plot(g, [r["opt_tikhonov_mse"] for r in sub], label=f"decay={md:.2f}")
        axes[2, 0].plot(g, [r["opt_sum1_nn_weight_phase"] for r in sub], label=f"decay={md:.2f}")
        axes[2, 1].plot(g, [r["opt_sum1_nn_mse"] for r in sub], label=f"decay={md:.2f}")
    axes[0, 0].set_ylabel("opt weight phase")
    axes[0, 1].set_ylabel("opt mse")
    axes[1, 0].set_ylabel("opt_tikhonov weight phase")
    axes[1, 1].set_ylabel("opt_tikhonov mse")
    axes[2, 0].set_ylabel("opt_sum1_nn weight phase")
    axes[2, 1].set_ylabel("opt_sum1_nn mse")
    for ax in axes.flat:
        ax.set_xlabel("memory_gain")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    p = os.path.join(out_dir, "stat_v19_figure5_optimal_estimator_weights.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)
    return out


def print_baseline(panel: List[Dict[str, float]]) -> None:
    print("delta\treference_excitation\testimator_phase\testimator_centroid\testimator_peak\terror_phase\terror_centroid\terror_peak")
    for r in panel:
        print(
            f"{r['delta0']:+.2f}\t{r['reference_excitation']:+.4f}\t{r['estimator_phase']:+.4f}\t"
            f"{r['estimator_centroid']:+.4f}\t{r['estimator_peak']:+.4f}\t{r['error_phase']:+.4f}\t"
            f"{r['error_centroid']:+.4f}\t{r['error_peak']:+.4f}"
        )


# ============================================================
# SECTION 7. PREDICTIVE WEIGHT SURFACE MODEL
# ============================================================

def weight_surface_feature_matrix(delta: np.ndarray, memory_gain: np.ndarray, memory_decay: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    mg = np.asarray(memory_gain, dtype=float)
    md = np.asarray(memory_decay, dtype=float)
    
    pulse = 1.0 / (1.0 + np.exp((np.abs(delta) - 1.05) / 0.30))
    
    return np.column_stack([
        np.ones_like(delta),
        delta,
        delta**2,
        delta**3,
        mg,
        md,
        delta * mg,
        delta * md,
        mg * md,
        (delta**2) * mg,
        (delta**2) * md,
        mg**2,
        md**2,
        np.exp(-0.5 * (delta / 0.5)**2), 
        pulse, 
    ])


def collect_local_weight_training(base: SimParams, decays: List[float], gains: List[float], detunings: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for md in decays:
        print(f"[progress] predictive training decay={md:.2f}", flush=True)
        for mg in gains:
            p = SimParams(**{**asdict(base), "memory_decay": md, "memory_gain": mg})
            panel = sweep_detuning_to_panel(detunings, p)
            W, _ = compute_delta_conditioned_estimator(panel, "opt_local_tikhonov")
            for i, r in enumerate(panel):
                rows.append({
                    "delta0": float(r["delta0"]),
                    "memory_gain": float(mg),
                    "memory_decay": float(md),
                    "weight_phase": float(W[i, 0]),
                    "weight_centroid": float(W[i, 1]),
                    "weight_peak": float(W[i, 2]),
                })
    return rows


def fit_predictive_weight_surface(training_rows: List[Dict[str, float]]) -> Dict[str, Dict[str, np.ndarray]]:
    delta = np.array([r["delta0"] for r in training_rows], dtype=float)
    mg = np.array([r["memory_gain"] for r in training_rows], dtype=float)
    md = np.array([r["memory_decay"] for r in training_rows], dtype=float)
    core_mask, tail_mask = split_core_tail(delta, threshold=1.05)
    
    results = {"core": {}, "tail": {}}
    for mask, regime in [(core_mask, "core"), (tail_mask, "tail")]:
        if not np.any(mask): continue
        X = weight_surface_feature_matrix(delta[mask], mg[mask], md[mask])
        for name in ["phase", "centroid", "peak"]:
            y = np.array([r[f"weight_{name}"] for i, r in enumerate(training_rows) if mask[i]], dtype=float)
            beta, *_ = np.linalg.lstsq(X, y, rcond=None)
            results[regime][name] = beta
    return results


def predict_weight_surface(delta: np.ndarray, memory_gain: float, memory_decay: float, coeffs: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
    delta = np.asarray(delta, dtype=float)
    mg = np.full_like(delta, float(memory_gain), dtype=float)
    md = np.full_like(delta, float(memory_decay), dtype=float)
    
    W = np.zeros((len(delta), 3))
    core_mask, tail_mask = split_core_tail(delta, threshold=1.05)
    
    for mask, regime in [(core_mask, "core"), (tail_mask, "tail")]:
        if not np.any(mask): continue
        Xf = weight_surface_feature_matrix(delta[mask], mg[mask], md[mask])
        reg_coeffs = coeffs[regime]
        w_regime = np.column_stack([
            Xf @ reg_coeffs["phase"],
            Xf @ reg_coeffs["centroid"],
            Xf @ reg_coeffs["peak"],
        ])
        for i in range(w_regime.shape[0]):
            w_regime[i] = project_sum1(w_regime[i])
        W[mask] = w_regime
        
    return W 


def apply_predictive_surface_to_panel(panel_rows: List[Dict[str, float]], coeffs: Dict[str, Dict[str, np.ndarray]], estimator_name: str = "opt_predict_tikhonov") -> Tuple[List[Dict[str, float]], Dict[str, float], np.ndarray]:
    if not panel_rows:
        return [], {}, np.empty((0, 3), dtype=float)
    delta = np.array([r["delta0"] for r in panel_rows], dtype=float)
    mg = float(panel_rows[0].get("memory_gain", 0.0))
    md = float(panel_rows[0].get("memory_decay", 0.0))
    W = predict_weight_surface(delta, mg, md, coeffs)
    X = np.column_stack([
        np.array([r["estimator_phase"] for r in panel_rows], dtype=float),
        np.array([r["estimator_centroid"] for r in panel_rows], dtype=float),
        np.array([r["estimator_peak"] for r in panel_rows], dtype=float),
    ])
    est = np.sum(X * W, axis=1)
    out_rows = append_estimator_to_rows(panel_rows, estimator_name, est)
    summary = summarize_composite_estimator(out_rows, estimator_name, W)
    return out_rows, summary, W


def summarize_predictive_surface_by_memory(base: SimParams, coeffs: Dict[str, Dict[str, np.ndarray]], decays: List[float], gains: List[float], detunings: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for md in decays:
        print(f"[progress] predictive stats decay={md:.2f}", flush=True)
        for mg in gains:
            p = SimParams(**{**asdict(base), "memory_decay": md, "memory_gain": mg})
            panel = sweep_detuning_to_panel(detunings, p)
            out_rows, summary, W = apply_predictive_surface_to_panel(panel, coeffs)
            rows.append({
                "memory_decay": md,
                "memory_gain": mg,
                "predict_weight_phase_mean": float(np.mean(W[:, 0])),
                "predict_weight_centroid_mean": float(np.mean(W[:, 1])),
                "predict_weight_peak_mean": float(np.mean(W[:, 2])),
                "predict_bias": summary["bias"],
                "predict_variance": summary["variance"],
                "predict_mse": summary["mean_squared_error"],
                "predict_rmse": summary["root_mean_squared_error"],
                "predict_mae": summary["mean_absolute_error"],
                "predict_normalized_bias": summary["normalized_bias"],
                "predict_normalized_variance": summary["normalized_variance"],
                "predict_normalized_mse": summary["normalized_mse"],
                "predict_resonance_mse": summary["resonance_mse"],
                "predict_offresonance_mse": summary["offresonance_mse"],
                "predict_resonance_concentration_ratio": summary["resonance_concentration_ratio"],
                "predict_te_reference_mse": summary.get("te_reference_mse", summary["mean_squared_error"]),
                "predict_te_family_tension": summary.get("te_family_tension", np.nan),
                "predict_te_weight_roughness": summary.get("te_weight_roughness", np.nan),
                "predict_te_total": summary.get("te_total", np.nan),
                "predict_te_alpha_mean": summary.get("te_alpha_mean", np.nan),
                "predict_te_beta_mean": summary.get("te_beta_mean", np.nan),
                "predict_te_gamma_mean": summary.get("te_gamma_mean", np.nan),
                "predict_te_adaptive_reference_mse": summary.get("te_adaptive_reference_mse", np.nan),
                "predict_te_adaptive_family_tension": summary.get("te_adaptive_family_tension", np.nan),
                "predict_te_adaptive_weight_roughness": summary.get("te_adaptive_weight_roughness", np.nan),
                "predict_te_adaptive_total": summary.get("te_adaptive_total", np.nan),
            })
    return rows


def write_weight_surface_coefficients_csv(coeffs: Dict[str, Dict[str, np.ndarray]], out_path: str) -> None:
    headers = ["regime", "weight_name", "term_index", "coefficient"]
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for regime, reg_coeffs in coeffs.items():
            for name, beta in reg_coeffs.items():
                for i, c in enumerate(beta):
                    writer.writerow([regime, name, i, float(c)])


def make_predictive_surface_plots(panel_rows: List[Dict[str, float]], coeffs: Dict[str, Dict[str, np.ndarray]], predict_rows: List[Dict[str, float]], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    pred_panel, _, W = apply_predictive_surface_to_panel(panel_rows, coeffs)
    delta = np.array([r["delta0"] for r in pred_panel], dtype=float)
    ref = np.array([r["reference_excitation"] for r in pred_panel], dtype=float)
    pred = np.array([r["estimator_opt_predict_tikhonov"] for r in pred_panel], dtype=float)
    err_pred = np.array([r["error_opt_predict_tikhonov"] for r in pred_panel], dtype=float)
    err_local = np.array([r["error_opt_local_tikhonov"] for r in pred_panel], dtype=float) if "error_opt_local_tikhonov" in pred_panel[0] else None

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), constrained_layout=True)
    axes[0].plot(delta, ref, label="reference_excitation")
    axes[0].plot(delta, pred, label="estimator_opt_predict_tikhonov")
    if err_local is not None:
        axes[0].plot(delta, [r["estimator_opt_local_tikhonov"] for r in pred_panel], label="estimator_opt_local_tikhonov")
    axes[0].set_xlabel("detuning")
    axes[0].set_ylabel("estimator value")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(delta, err_pred, label="error_opt_predict_tikhonov")
    if err_local is not None:
        axes[1].plot(delta, err_local, label="error_opt_local_tikhonov")
    axes[1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("detuning")
    axes[1].set_ylabel("signed error")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    p = os.path.join(out_dir, "stat_v19_figure6_predictive_surface_baseline.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    axes = axes.ravel()
    axes[0].plot(delta, W[:, 0], label="predict_weight_phase")
    axes[1].plot(delta, W[:, 1], label="predict_weight_centroid")
    axes[2].plot(delta, W[:, 2], label="predict_weight_peak")
    axes[3].plot(delta, np.sum(W, axis=1), label="sum(weights)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlabel("detuning")
    p = os.path.join(out_dir, "stat_v19_figure7_predictive_surface_weights.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    decays = sorted(set(r["memory_decay"] for r in predict_rows))
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for md in decays:
        sub = [r for r in predict_rows if abs(r["memory_decay"] - md) < 1e-12]
        sub = sorted(sub, key=lambda x: x["memory_gain"])
        g = [r["memory_gain"] for r in sub]
        axes[0, 0].plot(g, [r["predict_weight_phase_mean"] for r in sub], label=f"decay={md:.2f}")
        axes[0, 1].plot(g, [r["predict_weight_centroid_mean"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 0].plot(g, [r["predict_weight_peak_mean"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 1].plot(g, [r["predict_mse"] for r in sub], label=f"decay={md:.2f}")
    labels = ["predict weight phase mean", "predict weight centroid mean", "predict weight peak mean", "predict mse"]
    for ax, lab in zip(axes.ravel(), labels):
        ax.set_xlabel("memory_gain")
        ax.set_ylabel(lab)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    p = os.path.join(out_dir, "stat_v19_figure8_predictive_surface_memory.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    # TE-managed objective comparison across memory conditions
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for md in decays:
        sub = [r for r in predict_rows if abs(r["memory_decay"] - md) < 1e-12]
        sub = sorted(sub, key=lambda r: r["memory_gain"])
        g = [r["memory_gain"] for r in sub]
        axes[0, 0].plot(g, [r["predict_te_adaptive_total"] for r in sub], label=f"decay={md:.2f}")
        axes[0, 1].plot(g, [r["predict_te_adaptive_family_tension"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 0].plot(g, [r["predict_te_adaptive_weight_roughness"] for r in sub], label=f"decay={md:.2f}")
        axes[1, 1].plot(g, [r["predict_te_adaptive_reference_mse"] for r in sub], label=f"decay={md:.2f}")
    titles=["predict adaptive TE total","predict adaptive family tension","predict adaptive weight roughness","predict adaptive reference MSE"]
    for ax,title in zip(axes.ravel(),titles):
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xlabel("memory_gain")
    p = os.path.join(out_dir, "stat_v19_figure9_te_managed_objective.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    return out


def write_training_rows_csv(rows: List[Dict[str, float]], out_path: str) -> None:
    if not rows:
        return
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# SECTION 7. ROBUSTNESS / PERTURBATION TESTING
# ============================================================

def perturb_panel_estimators(panel_rows: List[Dict[str, float]], noise_sigma_frac: float, rng: np.random.Generator) -> List[Dict[str, float]]:
    out = []
    for r in panel_rows:
        nr = dict(r)
        scale = float(max(r.get("reference_scale", abs(r["reference_excitation"])), 1e-6))
        for name in ["phase", "centroid", "peak"]:
            key = f"estimator_{name}"
            err_key = f"error_{name}"
            n = float(rng.normal(0.0, noise_sigma_frac * scale))
            est = float(r[key] + n)
            nr[key] = est
            err = est - nr["reference_excitation"]
            nr[err_key] = err
            nr[f"abs_error_{name}"] = float(abs(err))
            nr[f"squared_error_{name}"] = float(err ** 2)
            nr[f"normalized_error_{name}"] = float(err / max(nr["reference_scale"], 1e-12))
        out.append(nr)
    return out


def apply_predictive_surface_to_panel_with_weight_jitter(panel_rows: List[Dict[str, float]], coeffs: Dict[str, Dict[str, np.ndarray]],
                                                         estimator_name: str = "opt_predict_tikhonov_perturbed",
                                                         weight_jitter_sigma: float = 0.0,
                                                         rng: np.random.Generator | None = None) -> Tuple[List[Dict[str, float]], Dict[str, float], np.ndarray]:
    if not panel_rows:
        return [], {}, np.empty((0, 3), dtype=float)
    if rng is None:
        rng = np.random.default_rng(0)
    delta = np.array([r["delta0"] for r in panel_rows], dtype=float)
    mg = float(panel_rows[0].get("memory_gain", 0.0))
    md = float(panel_rows[0].get("memory_decay", 0.0))
    W = predict_weight_surface(delta, mg, md, coeffs)
    if weight_jitter_sigma > 0.0:
        Wj = W + rng.normal(0.0, weight_jitter_sigma, size=W.shape)
        for i in range(W.shape[0]):
            W[i] = project_sum1(Wj[i]) 
    X = np.column_stack([
        np.array([r["estimator_phase"] for r in panel_rows], dtype=float),
        np.array([r["estimator_centroid"] for r in panel_rows], dtype=float),
        np.array([r["estimator_peak"] for r in panel_rows], dtype=float),
    ])
    est = np.sum(X * W, axis=1)
    out_rows = append_estimator_to_rows(panel_rows, estimator_name, est)
    summary = summarize_composite_estimator(out_rows, estimator_name, W)
    return out_rows, summary, W


def compute_global_te_band(predict_rows: List[Dict[str, float]]) -> Tuple[float, float, float]:
    vals = np.array([r["predict_te_adaptive_total"] for r in predict_rows], dtype=float)
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    width = float(hi - lo)
    return lo, hi, width


def summarize_predictive_robustness(base: SimParams, coeffs: Dict[str, Dict[str, np.ndarray]], decays: List[float], gains: List[float],
                                    detunings: np.ndarray, baseline_predict_rows: List[Dict[str, float]],
                                    n_trials: int = 24, estimator_noise_sigma_frac: float = 0.03,
                                    weight_jitter_sigma: float = 0.035, detuning_shift_sigma: float = 0.10,
                                    memory_gain_jitter_sigma: float = 0.12, rng_seed: int = 12345) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rng = np.random.default_rng(rng_seed)
    baseline_map = {(r["memory_decay"], r["memory_gain"]): r for r in baseline_predict_rows}
    band_lo, band_hi, band_width = compute_global_te_band(baseline_predict_rows)
    summary_rows: List[Dict[str, float]] = []
    trial_rows: List[Dict[str, float]] = []

    for md in decays:
        print(f"[progress] robustness decay={md:.2f}", flush=True)
        for mg in gains:
            key = (md, mg)
            baseline = baseline_map[key]
            per_trial = []
            for tr in range(n_trials):
                dshift = float(rng.normal(0.0, detuning_shift_sigma))
                mg_eff = float(max(0.0, mg + rng.normal(0.0, memory_gain_jitter_sigma)))
                p = SimParams(**{**asdict(base), "memory_decay": md, "memory_gain": mg_eff})
                panel = sweep_detuning_to_panel(detunings + dshift, p)
                panel = perturb_panel_estimators(panel, estimator_noise_sigma_frac, rng)
                out_rows, summary, W = apply_predictive_surface_to_panel_with_weight_jitter(
                    panel, coeffs,
                    estimator_name="opt_predict_tikhonov_perturbed",
                    weight_jitter_sigma=weight_jitter_sigma,
                    rng=rng,
                )
                te = float(summary.get("te_adaptive_total", np.nan))
                baseline_te = float(baseline["predict_te_adaptive_total"])
                tol = max(0.05 * baseline_te, 0.20 * max(band_width, 1e-9), 0.002)
                in_local_band = abs(te - baseline_te) <= tol
                in_global_band = (band_lo - 0.10 * band_width) <= te <= (band_hi + 0.10 * band_width)
                row = {
                    "memory_decay": md,
                    "memory_gain": mg,
                    "trial": tr,
                    "effective_memory_gain": mg_eff,
                    "detuning_shift": dshift,
                    "perturb_noise_sigma_frac": estimator_noise_sigma_frac,
                    "perturb_weight_jitter_sigma": weight_jitter_sigma,
                    "perturb_te_adaptive_total": te,
                    "perturb_te_adaptive_reference_mse": float(summary.get("te_adaptive_reference_mse", np.nan)),
                    "perturb_te_adaptive_family_tension": float(summary.get("te_adaptive_family_tension", np.nan)),
                    "perturb_te_adaptive_weight_roughness": float(summary.get("te_adaptive_weight_roughness", np.nan)),
                    "perturb_rmse": float(summary.get("root_mean_squared_error", np.nan)),
                    "perturb_mse": float(summary.get("mean_squared_error", np.nan)),
                    "perturb_bias": float(summary.get("bias", np.nan)),
                    "perturb_weight_phase_mean": float(np.mean(W[:, 0])),
                    "perturb_weight_centroid_mean": float(np.mean(W[:, 1])),
                    "perturb_weight_peak_mean": float(np.mean(W[:, 2])),
                    "baseline_predict_te_adaptive_total": baseline_te,
                    "te_ratio_to_baseline": float(te / baseline_te) if baseline_te > 1e-15 else np.nan,
                    "within_local_band": int(in_local_band),
                    "within_global_band": int(in_global_band),
                }
                trial_rows.append(row)
                per_trial.append(row)

            def arr(name: str) -> np.ndarray:
                return np.array([r[name] for r in per_trial], dtype=float)

            summary_rows.append({
                "memory_decay": md,
                "memory_gain": mg,
                "baseline_predict_te_adaptive_total": float(baseline["predict_te_adaptive_total"]),
                "baseline_predict_te_adaptive_reference_mse": float(baseline["predict_te_adaptive_reference_mse"]),
                "baseline_predict_te_adaptive_family_tension": float(baseline["predict_te_adaptive_family_tension"]),
                "baseline_predict_te_adaptive_weight_roughness": float(baseline["predict_te_adaptive_weight_roughness"]),
                "robust_te_adaptive_total_mean": float(np.mean(arr("perturb_te_adaptive_total"))),
                "robust_te_adaptive_total_std": float(np.std(arr("perturb_te_adaptive_total"))),
                "robust_te_adaptive_reference_mse_mean": float(np.mean(arr("perturb_te_adaptive_reference_mse"))),
                "robust_te_adaptive_family_tension_mean": float(np.mean(arr("perturb_te_adaptive_family_tension"))),
                "robust_te_adaptive_weight_roughness_mean": float(np.mean(arr("perturb_te_adaptive_weight_roughness"))),
                "robust_rmse_mean": float(np.mean(arr("perturb_rmse"))),
                "robust_rmse_std": float(np.std(arr("perturb_rmse"))),
                "robust_bias_mean": float(np.mean(arr("perturb_bias"))),
                "robust_weight_phase_mean": float(np.mean(arr("perturb_weight_phase_mean"))),
                "robust_weight_centroid_mean": float(np.mean(arr("perturb_weight_centroid_mean"))),
                "robust_weight_peak_mean": float(np.mean(arr("perturb_weight_peak_mean"))),
                "robust_te_ratio_mean": float(np.mean(arr("te_ratio_to_baseline"))),
                "robust_te_ratio_std": float(np.std(arr("te_ratio_to_baseline"))),
                "local_band_survival_rate": float(np.mean(arr("within_local_band"))),
                "global_band_survival_rate": float(np.mean(arr("within_global_band"))),
            })
    return summary_rows, trial_rows


def make_robustness_plots(robust_rows: List[Dict[str, float]], out_dir: str) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    out = []
    if not robust_rows:
        return out
    decays = sorted(set(r["memory_decay"] for r in robust_rows))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for md in decays:
        sub = [r for r in robust_rows if abs(r["memory_decay"] - md) < 1e-12]
        sub = sorted(sub, key=lambda r: r["memory_gain"])
        g = [r["memory_gain"] for r in sub]
        axes[0, 0].plot(g, [r["baseline_predict_te_adaptive_total"] for r in sub], linestyle='--', label=f'baseline d={md:.2f}')
        axes[0, 0].plot(g, [r["robust_te_adaptive_total_mean"] for r in sub], label=f'robust d={md:.2f}')
        axes[0, 1].plot(g, [r["robust_te_adaptive_total_std"] for r in sub], label=f'decay={md:.2f}')
        axes[1, 0].plot(g, [r["local_band_survival_rate"] for r in sub], label=f'decay={md:.2f}')
        axes[1, 1].plot(g, [r["robust_te_ratio_mean"] for r in sub], label=f'decay={md:.2f}')
    labels = [
        ("memory_gain", "TE adaptive total", "baseline vs robust TE"),
        ("memory_gain", "TE adaptive total std", "robust TE spread"),
        ("memory_gain", "local band survival", "stable-band survival rate"),
        ("memory_gain", "TE / baseline TE", "robust TE ratio"),
    ]
    for ax, (xl, yl, title) in zip(axes.ravel(), labels):
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    p = os.path.join(out_dir, "stat_v19_figure10_robustness.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for md in decays:
        sub = [r for r in robust_rows if abs(r["memory_decay"] - md) < 1e-12]
        sub = sorted(sub, key=lambda r: r["memory_gain"])
        g = [r["memory_gain"] for r in sub]
        axes[0].plot(g, [r["robust_weight_phase_mean"] for r in sub], label=f'decay={md:.2f}')
        axes[1].plot(g, [r["robust_weight_centroid_mean"] for r in sub], label=f'decay={md:.2f}')
        axes[2].plot(g, [r["robust_weight_peak_mean"] for r in sub], label=f'decay={md:.2f}')
    for ax, name in zip(axes, ["phase", "centroid", "peak"]):
        ax.set_xlabel("memory_gain")
        ax.set_ylabel(f"robust weight {name} mean")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)
    p = os.path.join(out_dir, "stat_v19_figure11_robust_weights.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out.append(p)

    return out


def moving_average_same(x: np.ndarray, window: int = 5) -> np.ndarray:
    window = max(1, int(window))
    if window == 1:
        return x.copy()
    k = np.ones(window, dtype=float) / float(window)
    return np.convolve(x, k, mode="same")


def measure_recovery_metrics(series: np.ndarray, baseline_idx: np.ndarray, shock_idx: np.ndarray,
                             post_idx: np.ndarray, tol_frac: float = 0.10) -> Dict[str, float]:
    arr = np.asarray(series, dtype=float)
    baseline = float(np.mean(arr[baseline_idx])) if baseline_idx.size else float(np.mean(arr))
    shock_peak = float(np.max(arr[shock_idx])) if shock_idx.size else baseline
    shock_min = float(np.min(arr[shock_idx])) if shock_idx.size else baseline
    peak_dev = max(abs(shock_peak - baseline), abs(shock_min - baseline))
    if peak_dev <= 1e-12:
        return {
            "baseline": baseline,
            "shock_peak": shock_peak,
            "shock_min": shock_min,
            "peak_dev": peak_dev,
            "settling_index": 0.0,
            "settling_steps": 0.0,
            "post_mean": float(np.mean(arr[post_idx])) if post_idx.size else baseline,
            "post_std": float(np.std(arr[post_idx])) if post_idx.size else 0.0,
            "recovered": 1.0,
        }
    tol = tol_frac * peak_dev
    settle_steps = np.nan
    recovered = 0.0
    if post_idx.size:
        post_vals = arr[post_idx]
        abs_dev = np.abs(post_vals - baseline)
        for k in range(len(post_vals)):
            if np.all(abs_dev[k:] <= tol):
                settle_steps = float(k)
                recovered = 1.0
                break
    return {
        "baseline": baseline,
        "shock_peak": shock_peak,
        "shock_min": shock_min,
        "peak_dev": peak_dev,
        "settling_index": settle_steps if not np.isnan(settle_steps) else -1.0,
        "settling_steps": settle_steps if not np.isnan(settle_steps) else float(len(post_idx)),
        "post_mean": float(np.mean(arr[post_idx])) if post_idx.size else baseline,
        "post_std": float(np.std(arr[post_idx])) if post_idx.size else 0.0,
        "recovered": recovered,
    }


def simulate_temporal_shock_response(base: SimParams, coeffs: Dict[str, Dict[str, np.ndarray]],
                                     shock_delta_jump: float = 1.0,
                                     shock_noise_sigma_frac: float = 0.05,
                                     shock_weight_jitter_sigma: float = 0.03,
                                     steps_pre: int = 20,
                                     steps_shock: int = 8,
                                     steps_post: int = 32,
                                     detuning_shift: float = 0.0,
                                     rng_seed: int = 123) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
    rng = np.random.default_rng(rng_seed)
    rows = []
    traj_rows = []
    gains = [0.00, 0.50, 1.00, 2.00]
    decays = [0.00, 0.15, 0.40]
    deltas = [-2.0 + detuning_shift, -1.0 + detuning_shift, 0.0 + detuning_shift, 1.0 + detuning_shift, 2.0 + detuning_shift]
    total_steps = steps_pre + steps_shock + steps_post

    def predict_weights(delta0: float, mg: float, md: float):
        regime = "core" if abs(delta0) <= 1.05 else "tail"
        feats = weight_surface_feature_matrix(np.array([delta0]), np.array([mg]), np.array([md]))[0]
        vals = [float(feats @ coeffs[regime][name]) for name in ('phase','centroid','peak')]
        w = project_sum1(np.array(vals, dtype=float))
        return w

    for md in decays:
        print(f"[progress] temporal shock decay={md:.2f}", flush=True)
        for mg in gains:
            for delta_base in deltas:
                te_series = []
                mse_series = []
                fam_series = []
                rough_series = []
                prev_w = None
                for t in range(total_steps):
                    in_shock = steps_pre <= t < (steps_pre + steps_shock)
                    delta_t = delta_base + (shock_delta_jump if in_shock else 0.0)
                    p = SimParams(
                        gamma=base.gamma, alpha=base.alpha, delta0=float(delta_t), sigma=base.sigma,
                        n_freq=base.n_freq, w_max=base.w_max, z=base.z, kappa=base.kappa,
                        memory_decay=float(md), memory_gain=float(mg),
                        te_delta_feedback=base.te_delta_feedback, te_kappa_feedback=base.te_kappa_feedback,
                        observer_sigma=base.observer_sigma,
                    )
                    res = run_single_panel(p) 
                    ref = float(res['reference_excitation'])
                    est = np.array([float(res['estimator_phase']), float(res['estimator_centroid']), float(res['estimator_peak'])], dtype=float)
                    scale = max(abs(ref), 1e-6)
                    noise_sigma = shock_noise_sigma_frac * scale * (1.0 if in_shock else 0.35)
                    est = est + rng.normal(0.0, noise_sigma, size=3)
                    w = predict_weights(delta_t, mg, md)
                    jitter = rng.normal(0.0, shock_weight_jitter_sigma * (1.0 if in_shock else 0.5), size=3)
                    w = project_sum1(w + jitter)
                    pred = float(np.dot(w, est))
                    ref_mse = (pred - ref) ** 2
                    fam_tension = float(np.mean((pred - est) ** 2))
                    if prev_w is None:
                        rough = 0.0
                    else:
                        rough = float(np.sum((w - prev_w) ** 2))
                    alpha, beta, gamma = adaptive_te_coefficients(delta_t, mg, md)
                    te_total = alpha * ref_mse + beta * fam_tension + gamma * rough
                    prev_w = w.copy()
                    te_series.append(te_total)
                    mse_series.append(ref_mse)
                    fam_series.append(fam_tension)
                    rough_series.append(rough)
                    traj_rows.append({
                        'memory_decay': md, 'memory_gain': mg, 'delta_base': delta_base,
                        'step': t, 'in_shock': int(in_shock), 'delta_t': delta_t,
                        'te_total': te_total, 'ref_mse': ref_mse, 'family_tension': fam_tension,
                        'weight_roughness': rough, 'weight_phase': w[0], 'weight_centroid': w[1], 'weight_peak': w[2],
                        'prediction': pred, 'reference': ref,
                    })
                baseline_idx = np.arange(0, steps_pre)
                shock_idx = np.arange(steps_pre, steps_pre + steps_shock)
                post_idx = np.arange(steps_pre + steps_shock, total_steps)
                te_metrics = measure_recovery_metrics(np.array(te_series), baseline_idx, shock_idx, post_idx)
                mse_metrics = measure_recovery_metrics(np.array(mse_series), baseline_idx, shock_idx, post_idx)
                rows.append({
                    'memory_decay': md, 'memory_gain': mg, 'delta_base': delta_base,
                    'shock_delta_jump': shock_delta_jump,
                    'shock_noise_sigma_frac': shock_noise_sigma_frac,
                    'shock_weight_jitter_sigma': shock_weight_jitter_sigma,
                    'te_baseline': te_metrics['baseline'],
                    'te_peak_dev': te_metrics['peak_dev'],
                    'te_settling_steps': te_metrics['settling_steps'],
                    'te_recovered': te_metrics['recovered'],
                    'te_post_mean': te_metrics['post_mean'],
                    'te_post_std': te_metrics['post_std'],
                    'mse_baseline': mse_metrics['baseline'],
                    'mse_peak_dev': mse_metrics['peak_dev'],
                    'mse_settling_steps': mse_metrics['settling_steps'],
                    'mse_recovered': mse_metrics['recovered'],
                    'mse_post_mean': mse_metrics['post_mean'],
                    'mse_post_std': mse_metrics['post_std'],
                })
    return rows, traj_rows


def main() -> None:
    print("[start] csn_te_stats_v19_robustness (The Bound Infinite)", flush=True)
    base = SimParams(
        gamma=1.0,
        alpha=1.5,
        delta0=0.0,
        sigma=0.4,
        n_freq=2**14,
        w_max=25.0,
        z=1.0,
        kappa=2.0,
        memory_decay=0.15,
        memory_gain=1.0,
        te_delta_feedback=0.25,
        te_kappa_feedback=0.25,
        observer_sigma=1.0,
    )

    detunings = np.linspace(-4.0, 4.0, 25)
    decays = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40]
    gains = [0.00, 0.25, 0.50, 0.75, 1.00, 1.50, 2.00]

    panel = sweep_detuning_to_panel(detunings, base)
    opt_summary_baseline, panel, _ = compute_optimal_estimator_for_panel(panel)
    memory_rows = sweep_memory_statistics(base, decays, gains, detunings)
    optimal_rows = compute_optimal_estimator_by_memory(base, decays, gains, detunings)
    memory_rows_sens = add_memory_sensitivities(memory_rows)

    training_rows = collect_local_weight_training(base, decays, gains, detunings)
    coeffs = fit_predictive_weight_surface(training_rows)
    panel_pred, pred_summary, predW = apply_predictive_surface_to_panel(panel, coeffs, estimator_name="opt_predict_tikhonov")
    predict_rows = summarize_predictive_surface_by_memory(base, coeffs, decays, gains, detunings)
    robust_rows, robust_trial_rows = summarize_predictive_robustness(
        base, coeffs, decays, gains, detunings, predict_rows,
        n_trials=24,
        estimator_noise_sigma_frac=0.03,
        weight_jitter_sigma=0.035,
        detuning_shift_sigma=0.10,
        memory_gain_jitter_sigma=0.12,
        rng_seed=12345,
    )

    fit_metrics = [
        "phase_bias", "phase_variance", "phase_mse",
        "centroid_bias", "centroid_variance", "centroid_mse",
        "peak_bias", "peak_variance", "peak_mse",
        "opt_bias", "opt_variance", "opt_mse",
        "opt_sum1_bias", "opt_sum1_variance", "opt_sum1_mse",
        "opt_sum1_nn_bias", "opt_sum1_nn_variance", "opt_sum1_nn_mse",
        "opt_tikhonov_bias", "opt_tikhonov_variance", "opt_tikhonov_mse",
        "opt_local_bias", "opt_local_variance", "opt_local_mse",
        "opt_local_sum1_bias", "opt_local_sum1_variance", "opt_local_sum1_mse",
        "opt_local_sum1_nn_bias", "opt_local_sum1_nn_variance", "opt_local_sum1_nn_mse",
        "opt_local_tikhonov_bias", "opt_local_tikhonov_variance", "opt_local_tikhonov_mse",
    ]
    fit_rows = fit_surface_coefficients(memory_rows, fit_metrics)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csn_te_stats_v19_robustness_outputs")
    os.makedirs(out_dir, exist_ok=True)
    write_csv(panel_pred, os.path.join(out_dir, "baseline_panel.csv"))
    write_csv(memory_rows, os.path.join(out_dir, "memory_statistics.csv"))
    write_csv(memory_rows_sens, os.path.join(out_dir, "memory_statistics_with_sensitivity.csv"))
    write_csv(optimal_rows, os.path.join(out_dir, "optimal_estimator_statistics.csv"))
    write_csv(predict_rows, os.path.join(out_dir, "predictive_surface_statistics.csv"))
    write_csv(robust_rows, os.path.join(out_dir, "robustness_statistics.csv"))
    write_csv(robust_trial_rows, os.path.join(out_dir, "robustness_trials.csv"))
    write_csv(fit_rows, os.path.join(out_dir, "surface_fit_coefficients.csv"))
    write_training_rows_csv(training_rows, os.path.join(out_dir, "predictive_weight_training.csv"))
    write_weight_surface_coefficients_csv(coeffs, os.path.join(out_dir, "predictive_weight_surface_coefficients.csv"))

    fig_paths = make_plots(panel_pred, memory_rows, optimal_rows, out_dir)
    fig_paths += make_predictive_surface_plots(panel_pred, coeffs, predict_rows, out_dir)
    fig_paths += make_robustness_plots(robust_rows, out_dir)

    print("\nBaseline predictive summary:")
    print(pred_summary)
    print("\nSaved files:")
    for p in fig_paths:
        print(p)


if __name__ == "__main__":
    main()
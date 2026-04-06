"""phi_fitter.py — Offline Phi-matrix fitter.

Collects rollouts under LQR + multisine excitation and fits a causal
linear multi-step predictor via ridge least squares.
Called automatically by PhiEnv.get_predictor() when the matrix file is missing.
"""

import math
import os

import numpy as np

from safe_control_gym.controllers.lqr.lqr_utils import compute_lqr_gain

_DT    = 1.0 / 50.0
_FREQS = np.array([0.4, 0.8, 1.2, 1.8, 2.6])

_ROBUST = dict(
    name="robust",
    init=dict(x=0.30, x_dot=1.00, z=0.35, z_dot=1.00, theta=0.12, theta_dot=2.00),
    amp=np.array([0.030, 0.030]),
    delta_u_clip=np.array([0.10, 0.10]),
    accept_input_dev_max=np.array([0.14, 0.14]),
    accept_x_max=1.6,
    accept_z_max=1.2,
    accept_theta_max=1.2,
    accept_xdot_max=8.0,
    accept_zdot_max=8.0,
    accept_thetadot_max=25.0,
    coupled_weight=0.8,
    coupled_max=2.4,
    sat_frac_max=0.50,
    mean_clip_max=0.18,
    mean_explicit_clip_max=0.18,
    action_noise_std=0.003,
    pulse_prob=0.04,
    pulse_min_steps=6,
    pulse_max_steps=18,
    pulse_diff_amplitude=0.06,
    pulse_collective_amplitude=0.03,
    require_violent=True,
)


def _configure_regime(env, regime, x_eq):
    b = regime["init"]
    env.RANDOMIZED_INIT = True
    env.INIT_STATE_RAND_INFO = {
        "init_x":         {"distrib": "uniform", "low": x_eq[0] - b["x"],         "high": x_eq[0] + b["x"]},
        "init_x_dot":     {"distrib": "uniform", "low": x_eq[1] - b["x_dot"],     "high": x_eq[1] + b["x_dot"]},
        "init_z":         {"distrib": "uniform", "low": x_eq[2] - b["z"],         "high": x_eq[2] + b["z"]},
        "init_z_dot":     {"distrib": "uniform", "low": x_eq[3] - b["z_dot"],     "high": x_eq[3] + b["z_dot"]},
        "init_theta":     {"distrib": "uniform", "low": x_eq[4] - b["theta"],     "high": x_eq[4] + b["theta"]},
        "init_theta_dot": {"distrib": "uniform", "low": x_eq[5] - b["theta_dot"], "high": x_eq[5] + b["theta_dot"]},
    }


def _collect_rollout(env, K, u_eq, x_eq, y_indices, regime, rollout_steps, rng):
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
    phases = rng.uniform(-np.pi, np.pi, size=(len(u_eq), len(_FREQS)))
    lo, hi = env.action_space.low, env.action_space.high
    n_sat = 0

    y0 = np.asarray(obs, dtype=np.float64)[y_indices]
    u_list, y_list = [], [y0]
    state_list = [np.asarray(env.state, dtype=np.float64).copy()]
    u_unclipped_list = []
    u_cmd_list = []

    headroom = np.minimum(hi - u_eq, u_eq - lo)
    delta_u_clip = np.minimum(np.asarray(regime["delta_u_clip"], dtype=np.float64), 0.98 * headroom)
    pulse_remaining = 0
    pulse_vec = np.zeros_like(u_eq, dtype=np.float64)

    for t in range(rollout_steps):
        x_hat = np.asarray(obs, dtype=np.float64)
        s = np.sin(2.0 * np.pi * _FREQS[None, :] * (t * _DT) + phases)

        if pulse_remaining <= 0 and rng.uniform() < regime["pulse_prob"]:
            pulse_remaining = int(
                rng.integers(regime["pulse_min_steps"], regime["pulse_max_steps"] + 1)
            )
            diff = float(
                rng.uniform(-regime["pulse_diff_amplitude"], regime["pulse_diff_amplitude"])
            )
            collective = float(
                rng.uniform(
                    -regime["pulse_collective_amplitude"],
                    regime["pulse_collective_amplitude"],
                )
            )
            pulse_vec = np.array([collective - diff, collective + diff], dtype=np.float64)
        elif pulse_remaining <= 0:
            pulse_vec = np.zeros_like(u_eq, dtype=np.float64)

        pulse_remaining = max(0, pulse_remaining - 1)
        noise = rng.normal(0.0, regime["action_noise_std"], size=len(u_eq)).astype(np.float64)

        u_unclipped = u_eq - K @ (x_hat - x_eq) + regime["amp"] * np.mean(s, axis=1) + pulse_vec + noise
        u_cmd = u_eq + np.clip(u_unclipped - u_eq, -delta_u_clip, delta_u_clip)
        u_clipped = np.clip(u_cmd, lo, hi)
        n_sat += int(np.any(u_cmd != u_clipped))

        result = env.step(u_clipped)
        obs, done = result[0], result[2]
        u_applied = np.asarray(getattr(env, "current_clipped_action", u_clipped), dtype=np.float64)
        u_unclipped_list.append(u_unclipped.copy())
        u_cmd_list.append(u_cmd.copy())
        u_list.append(u_applied.copy())
        y_list.append(np.asarray(obs, dtype=np.float64)[y_indices].copy())
        state_list.append(np.asarray(env.state, dtype=np.float64).copy())

        if done and t < rollout_steps - 1:
            return None

    u_arr = np.stack(u_list)
    y_arr = np.stack(y_list)
    state_arr = np.stack(state_list)
    u_unclipped_arr = np.stack(u_unclipped_list)
    u_cmd_arr = np.stack(u_cmd_list)

    state_dev = state_arr - x_eq[None, :]
    u_dev = u_arr - u_eq[None, :]
    state_ratio = np.maximum.reduce(
        [
            np.abs(state_dev[:, 0]) / regime["accept_x_max"],
            np.abs(state_dev[:, 2]) / regime["accept_z_max"],
            np.abs(state_dev[:, 4]) / regime["accept_theta_max"],
            np.abs(state_dev[:, 1]) / regime["accept_xdot_max"],
            np.abs(state_dev[:, 3]) / regime["accept_zdot_max"],
            np.abs(state_dev[:, 5]) / regime["accept_thetadot_max"],
        ]
    )
    input_ratio = np.max(
        np.abs(u_dev) / regime["accept_input_dev_max"][None, :],
        axis=1,
    )
    coupled_score = state_ratio[:-1] + regime["coupled_weight"] * input_ratio
    mean_clip = float(np.mean(np.mean(np.abs(u_arr - u_unclipped_arr), axis=1)))
    mean_explicit_clip = float(np.mean(np.mean(np.abs(u_cmd_arr - u_unclipped_arr), axis=1)))

    if np.max(state_ratio) > 1.0:
        return None
    if np.max(input_ratio) > 1.0:
        return None
    if np.max(coupled_score) > regime["coupled_max"]:
        return None
    if n_sat / rollout_steps > regime["sat_frac_max"]:
        return None
    if mean_clip > regime["mean_clip_max"]:
        return None
    if mean_explicit_clip > regime["mean_explicit_clip_max"]:
        return None

    if regime["require_violent"]:
        theta_max  = np.max(np.abs(y_arr[:, 2]))           # theta is y-channel 2
        u_diff_max = np.max(np.abs(u_arr[:, 0] - u_arr[:, 1]))
        if theta_max < 0.15 and u_diff_max < 0.04:
            return None

    return {"u": u_arr, "y": y_arr}


def _make_windows(rollout, t_ini, horizon, stride, n_u, n_y):
    u, y, T = rollout["u"], rollout["y"], rollout["u"].shape[0]
    z_list, y_list = [], []
    for t in range(t_ini, T - horizon + 1, stride):
        z_list.append(np.concatenate([
            u[t - t_ini : t].reshape(-1),
            y[t - t_ini : t + 1].reshape(-1),
            u[t : t + horizon].reshape(-1),
        ]))
        y_list.append(y[t + 1 : t + horizon + 1].reshape(-1))
    if not z_list:
        n_reg = t_ini * n_u + (t_ini + 1) * n_y + horizon * n_u
        return np.zeros((0, n_reg)), np.zeros((0, horizon * n_y))
    return np.array(z_list, dtype=np.float64), np.array(y_list, dtype=np.float64)


def _compute_scale(rollouts, t_ini, horizon, stride, n_u, n_y, eps=1e-8):
    n_reg = t_ini * n_u + (t_ini + 1) * n_y + horizon * n_u
    n_tar = horizon * n_y
    z_sum = np.zeros(n_reg); z_sq = np.zeros(n_reg)
    y_sum = np.zeros(n_tar); y_sq = np.zeros(n_tar)
    n = 0
    for r in rollouts:
        z, y = _make_windows(r, t_ini, horizon, stride, n_u, n_y)
        if z.shape[0] == 0:
            continue
        z_sum += z.sum(0); z_sq += (z ** 2).sum(0)
        y_sum += y.sum(0); y_sq += (y ** 2).sum(0)
        n += z.shape[0]
    def _std(s, sq):
        return np.maximum(np.sqrt(np.maximum(sq / n - (s / n) ** 2, 0.0)), eps)
    return _std(z_sum, z_sq), _std(y_sum, y_sq)


def _build_allowed(t_ini, horizon, n_y, n_u):
    n_past   = t_ini * n_u + (t_ini + 1) * n_y
    past_idx = np.arange(n_past)
    allowed  = []
    for row in range(horizon * n_y):
        fut_idx = np.arange(n_past, n_past + (row // n_y + 1) * n_u)
        allowed.append(np.concatenate([past_idx, fut_idx]))
    return allowed


def _fit_phi(rollouts, t_ini, horizon, stride, n_y, n_u, ridge, z_scale, y_scale):
    n_reg = t_ini * n_u + (t_ini + 1) * n_y + horizon * n_u
    n_tar = horizon * n_y
    gram  = np.zeros((n_reg, n_reg))
    cross = np.zeros((n_reg, n_tar))
    for r in rollouts:
        z, y = _make_windows(r, t_ini, horizon, stride, n_u, n_y)
        if z.shape[0] == 0:
            continue
        zn = z / z_scale; yn = y / y_scale
        gram += zn.T @ zn; cross += zn.T @ yn
    allowed  = _build_allowed(t_ini, horizon, n_y, n_u)
    phi_norm = np.zeros((n_tar, n_reg))
    for row, idx in enumerate(allowed):
        g = gram[np.ix_(idx, idx)] + ridge * np.eye(len(idx))
        c = cross[idx, row]
        try:
            phi_norm[row, idx] = np.linalg.solve(g, c)
        except np.linalg.LinAlgError:
            phi_norm[row, idx] = np.linalg.pinv(g) @ c
    return (y_scale[:, None] * phi_norm) / z_scale[None, :]


def fit_and_save_phi(
    env_func,
    t_ini: int,
    horizon: int,
    y_indices: list,
    n_robust: int = 2000,
    ridge: float = 1e-3,
    stride: int = 5,
    rollout_steps: int = 180,
    seed: int = 0,
    save_path: str = None,
) -> np.ndarray:
    """Collect rollout data and fit a causal Phi predictor matrix.

    Args:
        env_func:      Callable returning a raw (un-wrapped) environment.
        t_ini:         I/O history length.
        horizon:       Prediction horizon.
        y_indices:     Output state indices (e.g. [0, 2, 4] for x, z, theta).
        n_robust:      Target number of accepted robust rollouts.
        ridge:         Ridge regularisation coefficient.
        stride:        Sliding-window stride within each rollout.
        rollout_steps: Steps per rollout.
        seed:          RNG seed.
        save_path:     If given, save the Phi matrix as a .npy file here.

    Returns:
        phi (np.ndarray): Shape (horizon*n_y, t_ini*n_u + (t_ini+1)*n_y + horizon*n_u).
    """
    rng = np.random.default_rng(seed)
    env = env_func(episode_len_sec=int(math.ceil(rollout_steps / 50)) + 2)

    n_u  = int(np.prod(env.action_space.shape))
    n_y  = len(y_indices)
    x_eq = np.asarray(env.symbolic.X_EQ, dtype=np.float64).reshape(-1)
    u_eq = np.asarray(env.symbolic.U_EQ, dtype=np.float64).reshape(-1)
    K = compute_lqr_gain(
        env.symbolic, env.symbolic.X_EQ, env.symbolic.U_EQ,
        np.diag([1.0] * env.symbolic.nx),
        np.diag([0.1] * env.symbolic.nu),
        discrete_dynamics=True,
    )

    rollouts = []
    regime = _ROBUST
    _configure_regime(env, regime, x_eq)
    accepted, attempts = 0, 0
    while accepted < n_robust and attempts < 5 * n_robust:
        r = _collect_rollout(env, K, u_eq, x_eq, y_indices, regime, rollout_steps, rng)
        attempts += 1
        if r is not None:
            rollouts.append(r)
            accepted += 1
    print(f"[phi_fitter] {regime['name']}: {accepted}/{n_robust} ({attempts} attempts)", flush=True)

    env.close()

    print(f"[phi_fitter] Fitting Phi from {len(rollouts)} rollouts (t_ini={t_ini}, horizon={horizon}) ...", flush=True)
    z_scale, y_scale = _compute_scale(rollouts, t_ini, horizon, stride, n_u, n_y)
    phi = _fit_phi(rollouts, t_ini, horizon, stride, n_y, n_u, ridge, z_scale, y_scale)

    if save_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.save(save_path, phi)
        print(f"[phi_fitter] Phi saved → {save_path}", flush=True)

    return phi

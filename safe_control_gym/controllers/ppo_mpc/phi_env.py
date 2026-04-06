import os
import numpy as np
from collections import deque

from gymnasium import spaces


class PhiEnv:
    """Quadrotor wrapper used by Phi-MPC."""

    def __init__(self, env, y_indices, t_ini, phi_path, horizon, raw_env_func=None):
        self._env = env
        self.y_indices = list(y_indices)
        self.t_ini = int(t_ini)
        self._phi_path = phi_path
        self._horizon = int(horizon)
        self._raw_env_func = raw_env_func

        self.n_y = len(self.y_indices)
        self.n_u = int(np.asarray(env.action_space.shape).prod())

        # Only expose the output coordinates
        full_low = env.observation_space.low
        full_high = env.observation_space.high
        self.observation_space = spaces.Box(
            low=full_low[self.y_indices],
            high=full_high[self.y_indices],
            dtype=full_low.dtype,
        )

        self.action_space = env.action_space

        assert hasattr(env, "symbolic") and hasattr(env.symbolic, "U_EQ"), (
            "PhiEnv requires env.symbolic.U_EQ to initialise the I/O history."
        )
        self.u_eq = np.asarray(env.symbolic.U_EQ, dtype=np.float64).reshape(-1)

        self._u_hist = None
        self._y_hist = None
        # Keep the last history around for time-limit handling
        self._terminal_z_past = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, **kwargs):
        if self._u_hist is not None:
            self._terminal_z_past = self.z_past.copy()

        result = self._env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}

        y = np.asarray(obs, dtype=np.float64)[self.y_indices]

        self._u_hist = deque(
            [self.u_eq.copy() for _ in range(self.t_ini)],
            maxlen=self.t_ini,
        )
        self._y_hist = deque(
            [y.copy() for _ in range(self.t_ini + 1)],
            maxlen=self.t_ini + 1,
        )

        return y.astype(np.float32), info

    def step(self, action):
        result = self._env.step(action)
        obs, reward, *rest = result

        y = np.asarray(obs, dtype=np.float64)[self.y_indices]

        if self._y_hist is not None:
            self._y_hist.append(y.copy())
        if self._u_hist is not None:
            u_raw = np.asarray(action, dtype=np.float64)
            u_applied = np.asarray(
                getattr(self._env, "current_clipped_action", action),
                dtype=np.float64,
            ).copy()
            if (
                not hasattr(self, "_warned_action_mismatch")
                and np.max(np.abs(u_raw - u_applied)) > 1e-9
            ):
                self._warned_action_mismatch = True
                print(
                    "[PhiEnv] Raw action differed from applied action; "
                    "recording current_clipped_action in u_hist.",
                    flush=True,
                )
            self._u_hist.append(u_applied)

        return (y.astype(np.float32), reward) + tuple(rest)

    def close(self):
        return self._env.close()

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)

    @property
    def z_past_dim(self):
        """Size of z_past."""
        return self.t_ini * self.n_u + (self.t_ini + 1) * self.n_y

    @property
    def z_past(self):
        """Current I/O history as a flat vector."""
        if self._u_hist is None or self._y_hist is None:
            raise RuntimeError("PhiEnv.z_past accessed before reset().")
        u_stack = np.stack(list(self._u_hist)).reshape(-1)
        y_stack = np.stack(list(self._y_hist)).reshape(-1)
        return np.concatenate([u_stack, y_stack])

    def get_predictor(self):
        """Load the Phi matrix and split it into past and future blocks."""
        if not os.path.exists(self._phi_path):
            if self._raw_env_func is None:
                raise FileNotFoundError(
                    f"[PhiEnv] Phi matrix not found at {self._phi_path} and no "
                    f"raw_env_func was provided to fit one automatically."
                )
            print(
                f"[PhiEnv] Phi matrix not found — fitting from data "
                f"(t_ini={self.t_ini}, horizon={self._horizon}) ...",
                flush=True,
            )
            from safe_control_gym.controllers.ppo_mpc.phi_fitter import fit_and_save_phi
            fit_and_save_phi(
                self._raw_env_func,
                t_ini=self.t_ini,
                horizon=self._horizon,
                y_indices=self.y_indices,
                save_path=self._phi_path,
            )

        phi_full = np.load(self._phi_path).astype(np.float64)

        n_past = self.t_ini * self.n_u + (self.t_ini + 1) * self.n_y
        n_future_u = phi_full.shape[1] - n_past
        horizon_inferred = n_future_u // self.n_u
        assert horizon_inferred == self._horizon, (
            f"[PhiEnv.get_predictor] Phi matrix implies horizon={horizon_inferred} "
            f"but YAML specifies horizon={self._horizon}. "
            f"Check phi_path or mpc_config.horizon."
        )

        phi_past_init = phi_full[:, :n_past].copy()
        phi_future_init = phi_full[:, n_past:].copy()

        n_target = self._horizon * self.n_y
        mask_future = np.zeros((n_target, n_future_u), dtype=bool)
        for i in range(self._horizon):
            for j in range(i + 1):
                mask_future[i * self.n_y:(i + 1) * self.n_y,
                            j * self.n_u:(j + 1) * self.n_u] = True

        non_causal_energy = np.abs(phi_future_init[~mask_future]).sum()
        if non_causal_energy > 0.0:
            print(
                f"[PhiEnv.get_predictor] Warning: Phi_future has "
                f"{non_causal_energy:.2e} energy in non-causal positions — forcing to zero.",
                flush=True,
            )
            phi_future_init[~mask_future] = 0.0

        return phi_past_init, phi_future_init, mask_future

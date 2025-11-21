"""Benchmark action-value feature subsets against LQR ground truth using train/test splits."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inverted_pendulum import InvertedPendulum


# Mask search options (constant feature index 0 is always on)
MIN_ACTIVE_FEATURES = 7  # require at least this many active features (including the constant)
CUSTOM_MASKS: list[tuple[int, ...]] = []  # optionally provide explicit masks instead of enumerating
TRAIN_FRACTION = 0.8
RNG_SEED = 0


def load_ground_truth(path: Path):
    data = np.load(path, allow_pickle=True)
    params = data["params"].item()
    equilibrium_state = data["equilibrium_state"]
    discount = data["discount"]
    max_sim_time = data["max_sim_time"]
    pos_range = data["pos_range"]
    vel_range = data["vel_range"]
    torque_range = data["torque_range"]
    q = data["q"]
    return (
        params,
        equilibrium_state,
        discount,
        max_sim_time,
        pos_range,
        vel_range,
        torque_range,
        q,
    )


def enumerate_masks(feature_len: int) -> list[tuple[int, ...]]:
    if CUSTOM_MASKS:
        return [tuple(mask) for mask in CUSTOM_MASKS]

    masks = []
    for mask in itertools.product([0, 1], repeat=feature_len):
        if mask[0] != 1:  # constant term must be on
            continue
        if sum(mask) < MIN_ACTIVE_FEATURES:
            continue
        masks.append(mask)
    return masks


def build_feature_matrix(env, pos_range, vel_range, torque_range):
    res_pos, res_vel, res_torque = pos_range.size, vel_range.size, torque_range.size
    feature_len = env.action_feature_vector(0.0, np.array([0.0, 0.0])).shape[0]
    X = np.zeros((res_pos * res_vel * res_torque, feature_len))
    idx = 0
    for pos in pos_range:
        for vel in vel_range:
            state = np.array([pos, vel])
            for torque in torque_range:
                X[idx, :] = env.action_feature_vector(torque, state)
                idx += 1
    return X


def fit_and_evaluate(env, pos_range, vel_range, torque_range, q, masks, out_dir: Path):
    y = q.reshape(-1)
    X_full = build_feature_matrix(env, pos_range, vel_range, torque_range)
    true_max = q.max(axis=2)

    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(y.shape[0])
    n_train = int(TRAIN_FRACTION * y.shape[0])
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X_full[train_idx]
    y_train = y[train_idx]
    X_test = X_full[test_idx]
    y_test = y[test_idx]

    errors = []
    weights_by_mask = {}

    for mask in masks:
        mask_arr = np.array(mask, dtype=float)
        X_train_masked = X_train * mask_arr
        weights, _, _, _ = np.linalg.lstsq(X_train_masked, y_train, rcond=None)
        weights_by_mask[mask] = weights

        test_pred = (X_test * mask_arr) @ weights
        error = np.linalg.norm(test_pred - y_test)
        errors.append({"mask": mask, "error": error})
        print(f"Mask {mask} test error {error:.4f}")

    errors = sorted(errors, key=lambda e: e["error"])
    best = errors[0]
    print(f"Best mask {best['mask']} with test error {best['error']:.4f}")

    plot_true_max_q(true_max, pos_range, vel_range, out_dir)

    for entry in errors:
        mask = entry["mask"]
        mask_arr = np.array(mask, dtype=float)
        weights = weights_by_mask[mask]
        approx_full = (X_full * mask_arr) @ weights
        approx_full = approx_full.reshape(q.shape)
        approx_max = approx_full.max(axis=2)
        error_map = true_max - approx_max
        plot_mask_results(
            approx_max=approx_max,
            error_map=error_map,
            pos_range=pos_range,
            vel_range=vel_range,
            out_dir=out_dir,
            mask=mask,
            test_error=entry["error"],
        )

    top_k = min(15, len(errors))
    plt.figure(figsize=(9, 4))
    plt.bar(range(top_k), [errors[i]["error"] for i in range(top_k)])
    plt.xticks(
        range(top_k),
        [str(errors[i]["mask"]) for i in range(top_k)],
        rotation=70,
        ha="right",
        fontsize=7,
    )
    plt.ylabel("Test error (L2)")
    plt.title("Top feature masks by test error")
    plt.tight_layout()
    plt.savefig(out_dir / "lqr_mask_ranking.png")
    plt.close()


def plot_true_max_q(true_max, pos_range, vel_range, out_dir: Path):
    extent = [pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]]
    plt.figure(figsize=(6, 4))
    plt.imshow(true_max.T, origin="lower", extent=extent, aspect="auto")
    plt.colorbar()
    plt.xlabel("pos [rad]")
    plt.ylabel("vel [rad/s]")
    plt.title("True max Q on full grid (LQR policy)")
    plt.tight_layout()
    plt.savefig(out_dir / "lqr_true_max_q.png")
    plt.close()
    print(f"Saved true max Q to {out_dir / 'lqr_true_max_q.png'}")


def plot_mask_results(approx_max, error_map, pos_range, vel_range, out_dir: Path, mask, test_error: float):
    extent = [pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]]
    mask_label = "".join(str(bit) for bit in mask)

    plt.figure(figsize=(6, 4))
    plt.imshow(approx_max.T, origin="lower", extent=extent, aspect="auto")
    plt.colorbar()
    plt.xlabel("pos [rad]")
    plt.ylabel("vel [rad/s]")
    plt.title(f"Approx max Q (mask {mask_label}, test err {test_error:.3f})")
    plt.tight_layout()
    approx_path = out_dir / f"lqr_approx_max_q_mask_{mask_label}.png"
    plt.savefig(approx_path)
    plt.close()
    print(f"Saved approx heatmap for mask {mask_label} to {approx_path}")

    max_abs_error = np.max(np.abs(error_map))
    plt.figure(figsize=(6, 4))
    plt.imshow(
        error_map.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-max_abs_error,
        vmax=max_abs_error,
    )
    plt.colorbar()
    plt.xlabel("pos [rad]")
    plt.ylabel("vel [rad/s]")
    plt.title(f"Error (true - approx) for mask {mask_label}")
    plt.tight_layout()
    error_path = out_dir / f"lqr_error_heatmap_mask_{mask_label}.png"
    plt.savefig(error_path)
    plt.close()
    print(f"Saved error heatmap for mask {mask_label} to {error_path}")


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    data_path = out_dir / "lqr_ground_truth.npz"
    (
        params,
        equilibrium_state,
        discount,
        max_sim_time,
        pos_range,
        vel_range,
        torque_range,
        q,
    ) = load_ground_truth(data_path)
    env = InvertedPendulum(
        params=params,
        initial_state=equilibrium_state.copy(),
        sim_timestep=0.01,
        control_timestep=0.05,
    )

    feature_len = env.action_feature_vector(0.0, np.array([0.0, 0.0])).shape[0]
    masks = enumerate_masks(feature_len)

    fit_and_evaluate(env, pos_range, vel_range, torque_range, q, masks, out_dir)


if __name__ == "__main__":
    main()

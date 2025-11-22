"""Benchmark state-value feature subsets against LQR ground truth using train/test splits."""

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from inverted_pendulum import InvertedPendulum

# Mask search options (constant feature index 0 is always on)
MIN_ACTIVE_FEATURES = 1  # including the constant term
CUSTOM_MASKS: list[tuple[int, ...]] = []  # optionally provide explicit masks
TRAIN_FRACTION = 0.8
RNG_SEED = 0
MAX_MASKS = (
    10  #: int | None = None  # set to limit number of masks evaluated/visualized
)


def load_ground_truth(path: Path):
    data = np.load(path, allow_pickle=True)
    params = data["params"].item()
    equilibrium_state = data["equilibrium_state"]
    discount = data["discount"]
    max_sim_time = data["max_sim_time"]
    pos_range = data["pos_range"]
    vel_range = data["vel_range"]
    v_grid = data["v"]
    return (
        params,
        equilibrium_state,
        discount,
        max_sim_time,
        pos_range,
        vel_range,
        v_grid,
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


def build_feature_matrix(env, pos_range, vel_range):
    res_pos, res_vel = pos_range.size, vel_range.size
    base_len = env.state_feature_vector(np.array([0.0, 0.0])).shape[0]
    feature_len = base_len + 1  # add constant term
    X = np.zeros((res_pos * res_vel, feature_len))
    idx = 0
    for pos in pos_range:
        for vel in vel_range:
            state = np.array([pos, vel])
            feats = env.state_feature_vector(state)
            X[idx, 0] = 1.0  # constant
            X[idx, 1:] = feats
            idx += 1
    return X


def fit_and_evaluate(env, pos_range, vel_range, v_grid, masks, out_dir: Path):
    y = v_grid.reshape(-1)
    X_full = build_feature_matrix(env, pos_range, vel_range)

    mask_list = [tuple(m) for m in masks]

    global_v_max = np.max(np.abs(v_grid))
    global_err_max = 0.0

    rng = np.random.default_rng(RNG_SEED)
    perm = rng.permutation(y.shape[0])
    n_train = int(TRAIN_FRACTION * y.shape[0])
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    X_train = X_full[train_idx]
    y_train = y[train_idx]
    X_test = X_full[test_idx]
    y_test = y[test_idx]

    results = []
    weights_by_mask = {}

    for mask in mask_list:
        mask_arr = np.array(mask, dtype=float)
        X_train_masked = X_train * mask_arr
        weights, _, _, _ = np.linalg.lstsq(X_train_masked, y_train, rcond=None)
        weights_by_mask[mask] = weights

        test_pred = (X_test * mask_arr) @ weights
        error = np.linalg.norm(test_pred - y_test)
        approx_full = (X_full * mask_arr) @ weights
        approx_full = approx_full.reshape(v_grid.shape)
        error_map = v_grid - approx_full

        global_v_max = max(global_v_max, np.max(np.abs(approx_full)))
        global_err_max = max(global_err_max, np.max(np.abs(error_map)))

        results.append(
            {
                "mask": mask,
                "error": error,
                "approx_v": approx_full,
                "error_map": error_map,
            }
        )
        print(f"Mask {mask} test error {error:.4f}")

    results = sorted(results, key=lambda e: e["error"])
    best = results[0]
    worst = results[-1]
    print(f"Best mask {best['mask']} with test error {best['error']:.4f}")

    plot_true_value(v_grid, pos_range, vel_range, out_dir, global_v_max)

    if MAX_MASKS is None or len(results) <= MAX_MASKS:
        to_plot = results
    else:
        to_plot = []
        if best not in to_plot:
            to_plot.append(best)
        for entry in results[1:]:
            if len(to_plot) >= max(1, MAX_MASKS - 1):
                break
            if entry is not worst:
                to_plot.append(entry)
        if worst not in to_plot:
            to_plot.append(worst)

    for entry in to_plot:
        plot_mask_results(
            approx_v=entry["approx_v"],
            error_map=entry["error_map"],
            pos_range=pos_range,
            vel_range=vel_range,
            out_dir=out_dir,
            mask=entry["mask"],
            test_error=entry["error"],
            global_v_max=global_v_max,
            global_err_max=global_err_max,
        )

    top_k = min(15, len(results))
    plt.figure(figsize=(9, 4))
    plt.bar(range(top_k), [results[i]["error"] for i in range(top_k)])
    plt.xticks(
        range(top_k),
        [str(results[i]["mask"]) for i in range(top_k)],
        rotation=70,
        ha="right",
        fontsize=7,
    )
    plt.ylabel("Test error (L2)")
    plt.title("Top feature masks by test error (value fit)")
    plt.tight_layout()
    plt.savefig(out_dir / "lqr_value_mask_ranking.png")
    plt.close()


def plot_true_value(v_grid, pos_range, vel_range, out_dir: Path, global_v_max: float):
    extent = [pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]]
    plt.figure(figsize=(6, 4))
    plt.imshow(
        v_grid.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-global_v_max,
        vmax=global_v_max,
    )
    plt.colorbar()
    plt.xlabel("pos [rad]")
    plt.ylabel("vel [rad/s]")
    plt.title("True value V(s) on full grid (LQR policy)")
    plt.tight_layout()
    path = out_dir / "lqr_value_true.png"
    plt.savefig(path)
    plt.close()
    print(f"Saved true value heatmap to {path}")


def plot_mask_results(
    approx_v,
    error_map,
    pos_range,
    vel_range,
    out_dir: Path,
    mask,
    test_error: float,
    global_v_max: float,
    global_err_max: float,
):
    extent = [pos_range[0], pos_range[-1], vel_range[0], vel_range[-1]]
    mask_label = "".join(str(bit) for bit in mask)

    plt.figure(figsize=(6, 4))
    plt.imshow(
        approx_v.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        vmin=-global_v_max,
        vmax=global_v_max,
    )
    plt.colorbar()
    plt.xlabel("pos [rad]")
    plt.ylabel("vel [rad/s]")
    plt.title(f"Approx V(s) (mask {mask_label}, test err {test_error:.3f})")
    plt.tight_layout()
    approx_path = out_dir / f"lqr_value_approx_mask_{mask_label}.png"
    plt.savefig(approx_path)
    plt.close()
    print(f"Saved approx value heatmap for mask {mask_label} to {approx_path}")

    plt.figure(figsize=(6, 4))
    plt.imshow(
        error_map.T,
        origin="lower",
        extent=extent,
        aspect="auto",
        cmap="seismic",
        vmin=-global_err_max,
        vmax=global_err_max,
    )
    plt.colorbar()
    plt.xlabel("pos [rad]")
    plt.ylabel("vel [rad/s]")
    plt.title(f"Error (true - approx) V(s) for mask {mask_label}")
    plt.tight_layout()
    error_path = out_dir / f"lqr_value_error_mask_{mask_label}.png"
    plt.savefig(error_path)
    plt.close()
    print(f"Saved error heatmap for mask {mask_label} to {error_path}")


def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    data_path = Path("data") / "lqr_value_ground_truth.npz"
    (
        params,
        equilibrium_state,
        discount,
        max_sim_time,
        pos_range,
        vel_range,
        v_grid,
    ) = load_ground_truth(data_path)
    env = InvertedPendulum(
        params=params,
        initial_state=equilibrium_state.copy(),
        sim_timestep=0.01,
        control_timestep=0.05,
    )

    base_len = env.state_feature_vector(np.array([0.0, 0.0])).shape[0]
    feature_len = base_len + 1  # add constant
    masks = enumerate_masks(feature_len)

    fit_and_evaluate(env, pos_range, vel_range, v_grid, masks, out_dir)


if __name__ == "__main__":
    main()

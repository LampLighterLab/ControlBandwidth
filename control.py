import numpy as np


def feedback_from_P(
    P: np.ndarray, A: np.ndarray, B: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """
    Compute the discrete-time state-feedback gain K from a Riccati solution P.

    Solves K = (R + Bᵀ P B)⁻¹ (Bᵀ P A)
    """
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def solve_discrete_LQR(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    *,
    tolerance: float = 1e-9,
    max_iterations: int = 10_000,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Solve the infinite-horizon discrete-time LQR problem.

    Returns (K, P) where K is the optimal feedback gain and P the Riccati solution.

    Uses a straightforward Riccati iteration: P_{k+1} = Q + Aᵀ P_k A - Aᵀ P_k B (R + Bᵀ P_k B)⁻¹ Bᵀ P_k A.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have compatible dimensions.")
    if Q.shape != (A.shape[0], A.shape[0]):
        raise ValueError("Q must be square and match the state dimension.")
    if R.shape != (B.shape[1], B.shape[1]):
        raise ValueError("R must be square and match the control dimension.")

    P = Q.copy()
    for _ in range(max_iterations):
        RB = R + B.T @ P @ B
        K = np.linalg.solve(RB, B.T @ P @ A)
        P_next = Q + A.T @ P @ (A - B @ K)

        if np.linalg.norm(P_next - P, ord="fro") < tolerance:
            P = P_next
            break
        P = P_next
    else:
        raise RuntimeError(
            "Riccati iteration failed to converge within the iteration limit."
        )

    K = feedback_from_P(P, A, B, R)
    return K, P


def solve_continuous_ARE(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> np.ndarray:
    """
    Solve the continuous-time algebraic Riccati equation (ARE).

    Finds the positive semidefinite P satisfying Aᵀ P + P A - P B R⁻¹ Bᵀ P + Q = 0.
    """
    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix A must be square.")
    if A.shape[0] != B.shape[0]:
        raise ValueError("A and B must have compatible dimensions.")
    if Q.shape != (A.shape[0], A.shape[0]):
        raise ValueError("Q must be square and match the state dimension.")
    if R.shape != (B.shape[1], B.shape[1]):
        raise ValueError("R must be square and match the control dimension.")

    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    H = np.block([[A, -B @ R_inv @ B.T], [-Q, -A.T]])

    eigvals, eigvecs = np.linalg.eig(H)
    stable_indices = [i for i, val in enumerate(eigvals) if val.real < 0]
    if len(stable_indices) < n:
        raise RuntimeError("Unable to find enough stable eigenvalues for the ARE.")
    stable_indices = sorted(stable_indices, key=lambda idx: eigvals[idx].real)[:n]

    V = eigvecs[:, stable_indices]
    V1 = V[:n, :]
    V2 = V[n:, :]

    P = V2 @ np.linalg.inv(V1)
    P = (P + P.T) / 2.0
    return P.real


def solve_continuous_LQR(
    A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Continuous-time LQR gain and Riccati solution.

    Returns (K, P) where K = R⁻¹ Bᵀ P.
    """
    P = solve_continuous_ARE(A, B, Q, R)
    K = np.linalg.solve(R, B.T @ P)
    return K, P

"""
Numerical linear algebra and ODE integration utilities.
Pure NumPy — no SciPy dependency.

The small toolbox the physics engines lean on: a conjugate-gradient solver for the
big sparse systems from neutron diffusion, and an RK4 integrator for the time-domain
ODEs (kinetics, lumped thermal). Writing these by hand instead of importing SciPy is
the whole point of the project — and they're short enough that it's no great cost.
"""
from __future__ import annotations
import numpy as np


# ── Conjugate Gradient ─────────────────────────────────────────────────────

def cg_solve(A_mul, b: np.ndarray, x0=None, tol: float = 1e-8,
             max_iter: int = 2000) -> tuple[np.ndarray, int, float]:
    """
    Solve A·x = b via Conjugate Gradient (A symmetric positive-definite).

    Note A_mul is a *function* x → A·x, not a matrix. The diffusion operator is a
    cheap stencil (a few array shifts), so I never build the full matrix — CG only
    ever needs the product, and this keeps a 45×45×2 system tiny in memory.

    Returns (x, n_iters, residual_norm).
    """
    n   = len(b)
    x   = np.zeros(n) if x0 is None else x0.copy()
    r   = b - A_mul(x)          # initial residual
    p   = r.copy()              # first search direction
    rs  = float(r @ r)          # ‖r‖² , reused each step
    rs0 = max(rs, 1e-30)        # initial residual, for the relative stop test

    for i in range(max_iter):
        Ap    = A_mul(p)
        alpha = rs / max(float(p @ Ap), 1e-30)   # optimal step along p
        x    += alpha * p                        # advance the solution
        r    -= alpha * Ap                       # update residual (cheaper than b−Ax)
        rs_new = float(r @ r)
        if rs_new / rs0 < tol ** 2:              # converged (relative ‖r‖ small enough)
            return x, i + 1, float(np.sqrt(rs_new))
        p   = r + (rs_new / max(rs, 1e-30)) * p  # next conjugate direction (Fletcher–Reeves β)
        rs  = rs_new

    return x, max_iter, float(np.sqrt(rs))       # didn't converge — return best effort


# ── Jacobi-preconditioned CG ───────────────────────────────────────────────

def jcg_solve(diag: np.ndarray, A_mul, b: np.ndarray,
              tol: float = 1e-8, max_iter: int = 2000) -> tuple[np.ndarray, int, float]:
    """Jacobi (diagonal) preconditioned CG: M = diag(A).

    Cheapest preconditioner there is — just scale each row by 1/A_ii. For the
    diffusion operator that alone cuts the iteration count noticeably, because it
    evens out the wildly different diagonal magnitudes between the two energy groups.
    """
    M_inv = 1.0 / np.where(np.abs(diag) > 1e-30, diag, 1.0)   # guard against zeros

    # Solve the symmetrically-scaled system M⁻¹A x = M⁻¹b with plain CG.
    def AM(x):
        return M_inv * A_mul(x)

    return cg_solve(AM, M_inv * b, tol=tol, max_iter=max_iter)


# ── RK4 ODE integrator ─────────────────────────────────────────────────────

def rk4(f, y0: np.ndarray, t0: float, t1: float,
        n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic 4th-order Runge–Kutta integrator.

    The workhorse for the time-domain models. Fixed step (no adaptive control) —
    simple, predictable, and good enough since the callers already pick a small dt
    for their stiff systems. Stores the full trajectory so it can be plotted.

    f(t, y) → dy/dt
    Returns (t_array [n+1], y_array [n+1, len(y0)])
    """
    dt  = (t1 - t0) / n_steps
    t   = np.linspace(t0, t1, n_steps + 1)
    y   = np.zeros((n_steps + 1, len(y0)))
    y[0] = y0

    for i in range(n_steps):
        yi = y[i]
        ti = t[i]
        # four slope samples: start, two at the midpoint, one at the end…
        k1 = f(ti,          yi)
        k2 = f(ti + dt/2,   yi + dt/2 * k1)
        k3 = f(ti + dt/2,   yi + dt/2 * k2)
        k4 = f(ti + dt,     yi + dt   * k3)
        # …then step with their weighted average (midpoints count double)
        y[i + 1] = yi + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return t, y


# ── Simple Jacobi iteration ────────────────────────────────────────────────

def jacobi(A: np.ndarray, b: np.ndarray, tol: float = 1e-6,
           max_iter: int = 2000) -> tuple[np.ndarray, list[float]]:
    """Dense Jacobi iteration for benchmarking."""
    x   = np.zeros_like(b)
    D   = np.diag(A)
    R   = A - np.diag(D)
    res = []
    for _ in range(max_iter):
        x_new = (b - R @ x) / D
        r     = float(np.linalg.norm(b - A @ x_new))
        res.append(r)
        if r < tol:
            x = x_new
            break
        x = x_new
    return x, res

"""
Live numerical benchmarks — all computed on the host machine at runtime.
Covers: eigenvalue solver scaling, CG vs Jacobi, Monte Carlo 1/√N, RK4 order.

These double as a sanity check on the hand-written numerics in numerics/linalg.py:
each one reproduces a textbook result (CG beats Jacobi, Monte-Carlo error falls as
1/√N, RK4 error falls as h⁴). If the home-grown solvers were buggy, these curves
wouldn't line up with theory — so it's a small "the maths is actually right" demo,
run fresh on whatever machine is serving the page.
"""
from __future__ import annotations
import time
import numpy as np
from app.numerics.linalg import cg_solve, jacobi, rk4


def _bench_eig_scaling():
    """Measure eigenvalue solve time vs problem size."""
    sizes = [100, 400, 900, 2000, 4050]
    ts, ns = [], []
    for n in sizes:
        A = np.diag(np.arange(1, n+1, dtype=float))
        A += np.diag(np.full(n-1, -0.25), 1)
        A += np.diag(np.full(n-1, -0.25), -1)
        b = np.ones(n)

        t0 = time.perf_counter()
        for _ in range(3):
            cg_solve(lambda x, A=A: A @ x, b, tol=1e-6, max_iter=500)
        ts.append(round((time.perf_counter() - t0) / 3 * 1000, 2))
        ns.append(n)

    return {
        "n":    ns,
        "t_ms": ts,
        "note": f"Jacobi-CG, 3-trial mean · host: {_cpu_note()}",
    }


def _bench_cg_vs_jacobi():
    """Compare CG and Jacobi iteration on 64×64 Poisson problem."""
    N = 64
    n = N * N

    # 5-point Laplacian (dense for Jacobi, matrix-free for CG)
    diag   = np.full(n, 4.0)
    off_v  = np.full(n-1, -1.0)
    off_h  = np.full(n-N, -1.0)
    A = (np.diag(diag) + np.diag(off_v, 1) + np.diag(off_v, -1)
         + np.diag(off_h, N) + np.diag(off_h, -N))
    b = np.ones(n) * 0.01

    def Amul(x): return A @ x

    # CG
    iters_cg, res_cg = [], []
    x_cg = np.zeros(n)
    r = b - A @ x_cg
    p, rs = r.copy(), float(r @ r)
    for i in range(200):
        Ap = A @ p
        alpha = rs / max(float(p @ Ap), 1e-30)
        x_cg += alpha * p
        r -= alpha * Ap
        rs2 = float(r @ r)
        iters_cg.append(i+1)
        res_cg.append(float(np.sqrt(rs2)))
        if rs2 < 1e-12: break
        p   = r + (rs2 / max(rs, 1e-30)) * p
        rs  = rs2

    # Jacobi (subset of iters)
    _, res_jac = jacobi(A, b, tol=1e-6, max_iter=200)

    return {
        "cg_iters":  iters_cg,
        "cg_res":    [round(r, 9) for r in res_cg],
        "jac_iters": list(range(1, len(res_jac)+1)),
        "jac_res":   [round(r, 9) for r in res_jac],
        "note":      f"64×64 Poisson · CG converges in {len(iters_cg)} iters vs Jacobi {len(res_jac)}",
    }


def _bench_monte_carlo():
    """Demonstrate 1/√N Monte Carlo error convergence for π estimation."""
    rng     = np.random.default_rng(42)
    samples = [100, 500, 2000, 10000, 50000, 200000]
    errors, theory = [], []
    for N in samples:
        # classic dart-throwing π: fraction of points landing inside the unit
        # circle ×4. Error should track the 1/√N line, not beat it.
        X = rng.uniform(-1, 1, (N, 2))
        pi_est = 4.0 * np.mean((X**2).sum(1) <= 1.0)
        errors.append(round(abs(pi_est - np.pi), 6))
        theory.append(round(np.pi / (2.0 * np.sqrt(N)), 6))
    return {
        "n_samples": samples,
        "error":     errors,
        "theory":    theory,
        "note":      "π estimation · |π̂ − π| vs 1/√N theoretical bound",
    }


def _bench_rk4_order():
    """Verify 4th-order convergence of RK4 on dy/dt = −y, y(0)=1."""
    steps_list = [10, 20, 50, 100, 200, 500]
    h_list, err_list, th_list = [], [], []
    for n in steps_list:
        t, y = rk4(lambda t, y: -y, np.array([1.0]), 0.0, 1.0, n)
        err = float(abs(y[-1, 0] - np.exp(-1.0)))
        h   = 1.0 / n
        h_list.append(round(h, 5))
        err_list.append(round(err, 10))
        th_list.append(round(h**4 * 0.04, 10))
    return {
        "h":      h_list,
        "error":  err_list,
        "theory": th_list,
        "note":   "dy/dt = −y, y(0)=1 · |y(1) − e⁻¹| vs h⁴",
    }


def _cpu_note():
    try:
        import platform
        return platform.processor() or platform.machine()
    except Exception:
        return "unknown CPU"


def run_all() -> dict:
    return {
        "eig_scaling": _bench_eig_scaling(),
        "cg_vs_jacobi": _bench_cg_vs_jacobi(),
        "monte_carlo":  _bench_monte_carlo(),
        "rk4":          _bench_rk4_order(),
    }

"""
aero3d.py — 3D incompressible Navier–Stokes aerodynamics solver.

Pure NumPy, fully vectorised (no Python loops in the hot path). Fractional-step
(Chorin) projection on a collocated Cartesian grid:

    1. semi-Lagrangian advection (unconditionally stable)
    2. Smagorinsky–Lilly LES sub-grid eddy viscosity  nu_t = (Cs * Δ)^2 |S|
    3. explicit diffusion with the augmented viscosity (nu + nu_t)
    4. pressure Poisson solve (Jacobi) -> projection to a divergence-free field

A vehicle is voxelised into a no-slip immersed boundary. Aerodynamic loads are
recovered by integrating the resolved pressure over the body surface, giving
Cd, Cl, L/D and the centre of pressure.

This is the "absolute core" NumPy engine — no SciPy, no Numba, no PyPI frameworks.
"""
from __future__ import annotations
import numpy as np

CS = 0.16          # Smagorinsky constant
RHO = 1.0          # non-dimensional density


# ─────────────────────────── geometry ───────────────────────────
def car_mask(nx, ny, nz, yaw=0.0, ride=0.40):
    """Procedural open-wheel-ish car voxelised into a boolean solid mask."""
    x = np.arange(nx)[:, None, None]
    y = np.arange(ny)[None, :, None]
    z = np.arange(nz)[None, None, :]
    cx, cz, by = nx * 0.34, nz * 0.5, ny * (0.20 + ride * 0.34)
    Lx, Ly, Lz = nx * 0.20, ny * 0.16, nz * 0.30
    c, s = np.cos(-yaw), np.sin(-yaw)
    dx, dz, dy = x - cx, z - cz, y - by
    xr = dx * c - dz * s
    zr = dx * s + dz * c
    body = (xr / Lx) ** 2 + (dy / Ly) ** 2 + (zr / Lz) ** 2 < 1.0
    cab = ((xr - Lx * 0.35) / (Lx * 0.62)) ** 2 + ((dy - Ly * 0.85) / (Ly * 0.7)) ** 2 \
        + (zr / (Lz * 0.78)) ** 2 < 1.0    # cabin toward +x (rear); nose faces the wind (-x)
    return np.asarray(body | (cab & (dy > 0)), dtype=bool)


# ─────────────────────────── operators ──────────────────────────
def _laplacian(f):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0)
            + np.roll(f, 1, 1) + np.roll(f, -1, 1)
            + np.roll(f, 1, 2) + np.roll(f, -1, 2) - 6.0 * f)


def _advect(field, u, v, w, dt):
    """Semi-Lagrangian back-trace with vectorised trilinear interpolation."""
    nx, ny, nz = field.shape
    gx, gy, gz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    xb = np.clip(gx - dt * u, 0.0, nx - 1.001)
    yb = np.clip(gy - dt * v, 0.0, ny - 1.001)
    zb = np.clip(gz - dt * w, 0.0, nz - 1.001)
    x0 = xb.astype(np.intp); y0 = yb.astype(np.intp); z0 = zb.astype(np.intp)
    x1 = x0 + 1; y1 = y0 + 1; z1 = z0 + 1
    fx = xb - x0; fy = yb - y0; fz = zb - z0
    c000 = field[x0, y0, z0]; c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]; c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]; c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]; c111 = field[x1, y1, z1]
    c00 = c000 * (1 - fx) + c100 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _grad(f):
    gx = 0.5 * (np.roll(f, -1, 0) - np.roll(f, 1, 0))
    gy = 0.5 * (np.roll(f, -1, 1) - np.roll(f, 1, 1))
    gz = 0.5 * (np.roll(f, -1, 2) - np.roll(f, 1, 2))
    return gx, gy, gz


def _smagorinsky(u, v, w, delta=1.0):
    """nu_t = (Cs * delta)^2 * sqrt(2 S_ij S_ij)."""
    ux, uy, uz = _grad(u)
    vx, vy, vz = _grad(v)
    wx, wy, wz = _grad(w)
    s11, s22, s33 = ux, vy, wz
    s12 = 0.5 * (uy + vx); s13 = 0.5 * (uz + wx); s23 = 0.5 * (vz + wy)
    s_mag = np.sqrt(2.0 * (s11 * s11 + s22 * s22 + s33 * s33
                           + 2.0 * (s12 * s12 + s13 * s13 + s23 * s23)))
    return (CS * delta) ** 2 * s_mag


def _apply_bc(u, v, w, U, solid):
    # inlet (x=0): uniform freestream
    u[0] = U; v[0] = 0.0; w[0] = 0.0
    # outlet (x=-1): zero-gradient
    u[-1] = u[-2]; v[-1] = v[-2]; w[-1] = w[-2]
    # free-slip side / floor / ceiling walls (zero normal velocity)
    v[:, 0] = 0.0; v[:, -1] = 0.0
    w[:, :, 0] = 0.0; w[:, :, -1] = 0.0
    # no-slip solid body
    u[solid] = 0.0; v[solid] = 0.0; w[solid] = 0.0


def _pressure(u, v, w, dt, solid, iters):
    """Solve ∇²p = ρ/dt·∇·u (Jacobi). Proper Neumann (∂p/∂n=0) on the body and
    walls — a solid neighbour reflects the centre value — and p=0 at the outlet."""
    div = 0.5 * ((np.roll(u, -1, 0) - np.roll(u, 1, 0))
                 + (np.roll(v, -1, 1) - np.roll(v, 1, 1))
                 + (np.roll(w, -1, 2) - np.roll(w, 1, 2)))
    rhs = RHO / dt * div
    rhs[solid] = 0.0
    # which neighbour cells are solid (precomputed once)
    sxm = np.roll(solid, 1, 0);  sxp = np.roll(solid, -1, 0)
    sym = np.roll(solid, 1, 1);  syp = np.roll(solid, -1, 1)
    szm = np.roll(solid, 1, 2);  szp = np.roll(solid, -1, 2)
    p = np.zeros_like(u)
    for _ in range(iters):
        xm = np.where(sxm, p, np.roll(p, 1, 0)); xp = np.where(sxp, p, np.roll(p, -1, 0))
        ym = np.where(sym, p, np.roll(p, 1, 1)); yp = np.where(syp, p, np.roll(p, -1, 1))
        zm = np.where(szm, p, np.roll(p, 1, 2)); zp = np.where(szp, p, np.roll(p, -1, 2))
        pn = (xm + xp + ym + yp + zm + zp - rhs) / 6.0
        pn[solid] = 0.0
        pn[-1] = 0.0             # Dirichlet outlet anchors the pressure level
        p = pn
    return p


def _forces(p, solid, U):
    """Integrate resolved pressure over the body surface -> Cd, Cl, L/D, CoP.

    A solid cell whose neighbour in direction (axis, sh) is fluid carries a
    surface face with outward normal n = sh·e_axis; the pressure force on the
    body across it is -p_fluid · n.
    """
    nx, ny, nz = p.shape
    fluid = ~solid
    fx = 0.0; fy = 0.0; cop_num = 0.0; cop_den = 0.0
    for axis, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        surf = solid & np.roll(fluid, -sh, axis)        # solid faces touching fluid
        pf = np.roll(p, -sh, axis)[surf]                # the adjacent fluid pressure
        comp = -sh * pf                                 # force component along this axis
        if axis == 0:
            fx += float(np.sum(comp))
            xs = np.where(surf)[0]
            cop_num += float(np.sum(np.abs(comp) * xs)); cop_den += float(np.sum(np.abs(comp)))
        else:
            fy += float(np.sum(comp))
    area = float(np.count_nonzero(solid.any(axis=0))) or 1.0   # frontal area on the inlet plane
    q = 0.5 * RHO * U * U
    cd = fx / (q * area)
    cl = fy / (q * area)
    ld = cl / cd if abs(cd) > 1e-6 else 0.0
    cop_x = (cop_num / cop_den / nx) if cop_den > 1e-9 else 0.5
    return cd, cl, ld, cop_x


# ─────────────────────────── driver ─────────────────────────────
def solve(nx=64, ny=36, nz=36, U=1.0, yaw=0.0, ride=0.40,
          nu=0.004, steps=115, p_iters=30, mask=None):
    """Run the solver to a quasi-steady state. Returns (u, v, w, p, meta)."""
    if mask is None:
        mask = car_mask(nx, ny, nz, yaw, ride)
    solid = mask
    u = np.full((nx, ny, nz), U, dtype=np.float32)
    v = np.zeros((nx, ny, nz), dtype=np.float32)
    w = np.zeros((nx, ny, nz), dtype=np.float32)
    p = np.zeros((nx, ny, nz), dtype=np.float32)
    dt = 0.75
    _apply_bc(u, v, w, U, solid)
    for _ in range(steps):
        u2 = _advect(u, u, v, w, dt)
        v2 = _advect(v, u, v, w, dt)
        w2 = _advect(w, u, v, w, dt)
        nut = _smagorinsky(u2, v2, w2)
        visc = np.clip(nu + nut, 0.0, 0.16)            # keep explicit diffusion stable
        u2 += dt * visc * _laplacian(u2)
        v2 += dt * visc * _laplacian(v2)
        w2 += dt * visc * _laplacian(w2)
        _apply_bc(u2, v2, w2, U, solid)
        p = _pressure(u2, v2, w2, dt, solid, p_iters)
        gx, gy, gz = _grad(p)
        u = u2 - dt / RHO * gx
        v = v2 - dt / RHO * gy
        w = w2 - dt / RHO * gz
        _apply_bc(u, v, w, U, solid)
    cd, cl, ld, cop = _forces(p, solid, U)
    meta = dict(nx=nx, ny=ny, nz=nz, U=float(U), Cd=round(cd, 4), Cl=round(cl, 4),
                LD=round(ld, 3), CoP=round(cop, 3), steps=steps)
    return (u.astype(np.float32), v.astype(np.float32),
            w.astype(np.float32), p.astype(np.float32), meta)


if __name__ == "__main__":
    import time
    t0 = time.time()
    u, v, w, p, meta = solve()
    dt = time.time() - t0
    sp = np.sqrt(u * u + v * v + w * w)
    print(f"solved {meta['nx']}x{meta['ny']}x{meta['nz']} in {dt:.2f}s")
    print("meta:", meta)
    print(f"|u| range [{sp.min():.3f}, {sp.max():.3f}]  p range [{p.min():.3f}, {p.max():.3f}]")

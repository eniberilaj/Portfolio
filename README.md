# Computational Physics Portfolio

Five interactive computational-physics and applied-maths projects, each built from
scratch to understand the method behind it. A nuclear reactor digital twin,
hand-written neural networks, a 3D wind tunnel, an orbital-debris cascade
simulator, and a Monte-Carlo wealth engine.

Every solver is written by hand in **NumPy** (server) and **vanilla JavaScript**
(browser). No web framework, no ORM, no SciPy, no ML libraries — the only PyPI
dependency is NumPy. One single-file SPA talks to a small Python standard-library
server.

---

## Quick Start

```bash
python run.py        # starts on http://127.0.0.1:8050
```

No build step and no `pip install` beyond NumPy. Python 3.10+ recommended.

```bash
pip install numpy    # the one and only dependency
```

---

## The Projects

| # | Project | What it is | Core techniques |
|---|---------|------------|-----------------|
| 1 | **Nuclear Reactor** | Live PWR digital twin | 2-group neutron diffusion · 6-group point kinetics · thermal-hydraulic feedback |
| 2 | **Neural Network Lab** | Three from-scratch ML labs | Manual backprop · Adam · PINN · autoencoder (pure JS) |
| 3 | **3D Wind Tunnel** | Aerodynamics over real CAD | NumPy LES Navier–Stokes · WebGL streamlines · Cd/Cl loads |
| 4 | **Orbital Debris Simulator** | Kessler-cascade sandbox | Kepler + J₂ propagator · spatial-hash collisions · drag/re-entry |
| 5 | **Monte Carlo Wealth Engine** | Personal-finance projection | N-path Monte-Carlo · FIRE/4% rule · sensitivity tornado |

---

### 1 · Nuclear Reactor (`app/physics/`, `app/api/reactor.py`)

A real-time digital twin of a 4-loop PWR-class core. Two-group steady-state
neutron diffusion is solved on a 2D Cartesian mesh, coupled to a fuel/coolant
thermal model and 6-group point kinetics so the core responds live as you move
control rods, enrichment, and coolant conditions.

```
Group 1 (fast):    -D₁∇²φ₁ + (Σa1 + Σs12)φ₁ = (1/k)(νΣf1·φ₁ + νΣf2·φ₂)
Group 2 (thermal): -D₂∇²φ₂ + Σa2·φ₂        = Σs12·φ₁
```

- Solved by **power iteration** (outer) with a **Jacobi-preconditioned conjugate
  gradient** inner solve (`app/numerics/linalg.py`).
- Mesh 30×30 → 45×45; circular active region; stencil multiply via NumPy slicing —
  no Python loops in the hot path (~40 ms/solve).
- Cross-section feedback applied before each solve: enrichment, moderator
  temperature coefficient (calibrated **MTC ≈ −28 pcm/°C**), void/density,
  per-cell control-rod banks (**rod worth ≈ 2130 pcm**), and burnup depletion.
- Transient behaviour driven by 6-group **point kinetics** (`kinetics.py`) with
  xenon poisoning (`xenon.py`) and depletion via the Bateman equations
  (`burnup.py`).

### 2 · Neural Network Lab (`app/static/index.html`, `app/neural/`)

Three interactive teaching labs that run **entirely client-side in vanilla
JavaScript** — no TensorFlow, no PyTorch, no WASM. Every forward pass, gradient,
and weight update is computed by hand on plain 2-D arrays.

All three labs share a ~30-line 2-D-array library `M`: `dot`, `T` (transpose),
`add`/`sub`, `emul` (Hadamard), `sc` (scalar), `app` (elementwise map), `bias`,
`sum0` (column sums), and a Box–Muller `rn` for **He-initialised** weights
(`W ~ N(0, 2/n_in)`). There is no autodiff — every gradient below is derived by
hand.

#### Lab 1 — Classification MLP

A general multi-layer perceptron for binary classification, trained by **manual
backpropagation** with Adam or SGD-with-momentum.

```
Forward:   aₗ = σ(Wₗ·aₗ₋₁ + bₗ),     p = sigmoid(z_L)
Loss:      BCE = −1/N Σ [ y·log p + (1−y)·log(1−p) ]
Backward:  δ_L = (p − y);   δₗ = (δₗ₊₁·Wₗ₊₁ᵀ) ⊙ σ′(zₗ)
Grads:     ∂L/∂Wₗ = aₗ₋₁ᵀ·δₗ,   ∂L/∂bₗ = Σ δₗ
Adam:      mₜ = β₁mₜ₋₁+(1−β₁)g,  vₜ = β₂vₜ₋₁+(1−β₂)g²   (β₁=0.9, β₂=0.999)
           θ ← θ − α·m̂ₜ/(√v̂ₜ+ε)   with bias-corrected m̂, v̂
```

Datasets: two spirals, moons, blobs, concentric rings, XOR. Activations ReLU /
tanh / sigmoid; user-defined hidden layers (e.g. `64,32,16`). A live SVG node-link
diagram recolours its edges from the **actual `W` matrices** — positive weights in
the accent colour, negative in orange, opacity/thickness ∝ |w|/max|w|.

#### Lab 2 — Physics-Informed Neural Network (PINN)

A `1 → 32 → 32 → 1` tanh network that simultaneously fits sparse flux observations
*and* is penalised for violating the 1-D steady-state neutron-diffusion PDE:

```
−D·φ″(x) + Σₐ·φ(x) = S,   x ∈ [0,1],   φ(0) = φ(1) = 0
Loss = ℒ_data + λ·ℒ_pde + 10·ℒ_bc
```

The PDE residual `r(x) = −D·φ″ + Σₐ·φ − S` is evaluated at 40 collocation points
using a finite-difference stencil on the **network's own outputs**
(`φ(x−h), φ(x), φ(x+h)`), so the second derivative needs no autodiff. A heatmap of
`|r(x)|` and the three glowing loss terms show which objective currently dominates.

#### Lab 3 — Autoencoder (latent space)

A deep autoencoder compresses 8×8 reactor thermal maps into a 2-D latent code and
reconstructs them, trained on reconstruction MSE:

```
Encoder:  64 → 32 → 16 → 2        Decoder:  2 → 16 → 32 → 64 (sigmoid)
Loss:     MSE = 1/N Σ ‖x − x̂‖²
```

Pin two latent points A and B, then scrub `z = (1−t)·A + t·B`: the decoder
interpolates each intermediate code and animates the reconstruction — smooth
transitions show it learned a continuous manifold, not memorised samples.

Each lab's control column ends in a telemetry terminal streaming real per-step
metrics (`‖W‖`, gradient/residual magnitudes, ms/step) read straight off the
arrays mid-loop.

### 3 · 3D Wind Tunnel (`app/physics/aero3d.py`, `app/api/cfd.py`)

Airflow over real car geometry, solved by a NumPy incompressible Navier–Stokes
engine on a collocated Cartesian grid (default 64×36×36). Each time step is a
**fractional-step (Chorin) projection**, fully vectorised — no Python loops in the
hot path:

```
1. Advect    semi-Lagrangian back-trace + vectorised trilinear interpolation
             (unconditionally stable):   u* = u(x − Δt·u)
2. LES        Smagorinsky–Lilly sub-grid eddy viscosity
             ν_t = (Cₛ·Δ)²·|S|,   |S| = √(2 SᵢⱼSᵢⱼ),   Cₛ = 0.16
3. Diffuse    explicit, with the augmented viscosity:  u* += Δt·(ν+ν_t)·∇²u*
4. Project    pressure Poisson (Jacobi):  ∇²p = (ρ/Δt)·∇·u*
             then  u = u* − (Δt/ρ)·∇p   → divergence-free
```

- The vehicle is voxelised into a **no-slip immersed boundary**; the Poisson solve
  applies proper Neumann (∂p/∂n = 0) on the body and walls (a solid neighbour
  reflects the centre value) and a Dirichlet `p = 0` outlet to anchor the level.
- Aerodynamic loads are recovered by integrating the resolved pressure over the
  body's surface faces (force = −p·n̂):
  ```
  q = ½ρU²,   C_d = F_x/(q·A),   C_l = F_y/(q·A),   L/D = C_l/C_d
  ```
  plus the centre of pressure. Imports real CAD meshes (`CAD/` — Bugatti, McLaren
  Senna, Audi R8, …).
- Binary protocol: `[uint32 headerLen][JSON meta][float32 RGBA volume]`, laid out
  x-fastest (R=u, G=v, B=w, A=p) so the browser drops it straight into a WebGL
  `DataTexture3D` and traces GPU streamlines with zero per-voxel parsing.

### 4 · Orbital Debris Simulator (`app/physics/orbital.py`, `app/api/spacelab.py`)

A Kessler-syndrome sandbox: a single collision can cascade until debris makes whole
orbits unusable. A pure-NumPy engine (no sgp4 / astropy / numba) propagates 20,000+
objects as `(N,3)` array algebra — not a single `for` loop in any hot path
(~4.6 ms/step @ 13.4k objects) — and streams them to a Three.js globe.

**Propagation — two-body + J₂ secular precession.** The population starts as
Walker-delta constellation shells plus a background debris belt. Kepler's equation
is solved by a fully vectorised Newton–Raphson over the whole array:

```
M = E − e·sin E       solved by   E ← E − (E − e sinE − M)/(1 − e cosE)
                      seeded with E₀ = M + e·sinM   (6 fixed iterations)
state:  r = Rz(Ω)·Rx(i)·Rz(ω)·r_perifocal       (vectorised coe2rv)
J₂:     k = 1.5·J₂·(Rₑ/p)²·n
        Ω̇ = −k·cos i        (node regression)
        ω̇ =  k·(2 − 2.5·sin²i)   (apsidal rotation)
        Ṁ =  n + k·√(1−e²)·(1 − 1.5·sin²i)
```

**Drag → decay → re-entry** is the cascade's sink. Density is an exponential
atmosphere `ρ(h) = ρ₀·exp(−(h−h_ref)/H)` evaluated at perigee; the semi-major axis
bleeds off as `da/dt ≈ −G·(C_d A/m)·ρ·√(μa)`, the orbit circularises, and an object
burns up once its perigee drops below ~100 km. Per-object **ballistic coefficient**
(`C_d A/m`) makes tumbling debris decay far faster than compact satellites.

**Collisions — vectorised spatial hashing**, no O(N²) pairwise scan: `np.floor`
buckets the ECI cloud into voxels → a packed 1-D key → `np.lexsort` makes cell-mates
adjacent → equal-key neighbours (shifts 1 & 2) are candidate pairs, filtered by true
Euclidean separation < threshold.

**Kessler cascade** — a saturating actuarial accumulator:

```
collisions = n_conj · P_hit · (1 + (occ/occ_crit)²)     density feedback
debris    += collisions · FRAG  −  re-entries            source minus sink
P_c        = 1 − exp(−Λ),   Λ = Σ collisions·(debris/debris₀)·c
```

A **kinetic-impact / ASAT** event detonates a dormant reserve into an isotropic Δv
fragment shell, recovers each fragment's osculating elements with `rv2coe`, and
re-injects them into the propagator. Positions stream as raw `float32` over a
**hand-rolled stdlib WebSocket** (RFC 6455 handshake + frame codec) into a
`THREE.InstancedMesh` with a procedural day/night Earth, eclipse shading, and
re-entry flashes.

### 5 · Monte Carlo Wealth Engine (`app/finance/engine.py`, `app/api/finance.py`)

A personal-finance projection built to learn Monte-Carlo methods properly — a
native, zero-dependency reimplementation of a Dash/pandas/yfinance app, rewritten
as vectorised NumPy so a 400-path × 25-year simulation runs server-side in ~19 ms.

**Inputs → cash flow.** Payroll converts gross to net and grows annually;
categorised expenses have finite durations (loans roll off); mortgages use the
standard fixed-rate annuity; one-off milestones drain capital.

```
net      = gross·(1 − tax% − social%),     grows (1 + g)^year
mortgage = L·r(1+r)ⁿ / ((1+r)ⁿ − 1)        r = monthly rate, n = term in months
surplus  = net − active_expenses + rental_net_cashflow
```

**The Monte-Carlo.** Only the invested portfolio is stochastic — cash and
real-estate follow deterministic paths, so the whole `(n_sims, months)` invested
matrix is banked and reduced to percentiles in one vectorised pass:

```
real return  μ = gross_return − fees − inflation
monthly       μ_m = μ/12,    σ_m = volatility/√12
returns       r_t ~ 𝒩(μ_m, σ_m)              shape (n_sims, months)
evolution     invₜ = invₜ₋₁·(1 + r_t) + min(surplus, contribution)
net worth     NWₜ = investedₜ + cashₜ + (property_valueₜ − debtₜ)
fan chart     p10/p50/p90 = percentile(NW, [10,50,90], axis=0)
```

**FIRE analytics (4% rule).** From the median path it derives a
financial-independence date, success probability, and safe-withdrawal income:

```
fire_number  = 25 × annual_expenses                 (4% safe-withdrawal target)
FIRE date    = first month the p50 path crosses fire_number
P(success)   = mean( final_net_worth ≥ fire_number )
SWR income   = 0.04 × p50_final / 12
```

A **sensitivity tornado** then re-runs the model with a one-at-a-time ±
perturbation of each key driver (market return, contribution, inflation,
volatility, salary growth, gross salary), ranking them by the swing in final median
net worth.

> Returns are assumed independent and normally distributed; real markets have fat
> tails and sequence-of-returns risk. Treat it as intuition, not financial advice.

---

## Zero-Dependency Architecture

One single-file page talking to a small Python server. Five independent projects,
each with its own engine, all on the standard library and NumPy.

- **Frontend** — Vanilla ES2022 · `Three.js r128` WebGL2 (InstancedMesh + GLSL
  shaders) · `Plotly` · `MathJax`, with raw Float32 ingestion and custom render
  passes.
- **Transport** — REST JSON · a hand-rolled `stdlib WebSocket` (RFC 6455) · packed
  `float32` binary streams via `.tobytes()`.
- **API** — `/reactor` · `/neural` · `/cfd` · `/spacelab` · `/finance` dispatched
  on a stdlib `ThreadingHTTPServer`.
- **Engines** — NumPy: 2-group neutron diffusion · LES Navier–Stokes · Kepler + J₂
  propagator with spatial-hash collisions · Monte-Carlo wealth model · point
  kinetics & burnup.
- **Runtime** — Python standard library and NumPy only. `python run.py`, no build
  step, no `pip install`.

---

## Project Structure

```
run.py                     # entry point → app.server.main()  (port 8050)
requirements.txt           # numpy>=1.24  (the only dependency)
CAD/                       # car meshes for the wind tunnel (obj/blend/fbx)
app/
├── server.py              # ThreadingHTTPServer + WebSocket upgrade hook
├── ws.py                  # hand-rolled RFC 6455 WebSocket (stdlib only)
├── state.py               # shared simulation state
├── api/                   # REST/WS routers
│   ├── reactor.py  neural.py  cfd.py  spacelab.py  finance.py
│   └── experiments.py
├── physics/               # reactor, CFD & orbital engines
│   ├── diffusion.py  thermal.py  kinetics.py  burnup.py  xenon.py
│   ├── coupled.py  fluctuations.py
│   ├── aero3d.py          # LES Navier–Stokes
│   └── orbital.py         # Kepler + J₂ + drag + collisions
├── finance/engine.py      # Monte-Carlo wealth engine
├── neural/                # dataset & training helpers
├── numerics/linalg.py     # Jacobi-preconditioned CG solver
├── optimization/  computing/
└── static/
    ├── index.html         # the entire single-page app (CSS + HTML + JS)
    └── cv.pdf
```

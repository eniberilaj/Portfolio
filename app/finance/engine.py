"""
Strategic wealth & cash-flow engine — pure, vectorised NumPy.

A native, zero-dependency reimplementation of a personal-finance Monte-Carlo
calculator (originally a Dash/Plotly/pandas/yfinance app). Everything that broke
the portfolio's "stdlib + NumPy only" thesis has been stripped — no pandas, no
sqlite persistence, no yfinance live market calls — and the maths is expressed as
vectorised NumPy so the whole simulation runs server-side in a few milliseconds.

What it models
--------------
1.  Payroll      gross → net (tax + social contributions), with annual growth.
2.  Cash flow    categorised expenses with finite durations (loans roll off).
3.  Investing    geometric Brownian portfolio (real return = gross − fees − inflation)
                 run as an **N-path Monte-Carlo** with p10/p50/p90 fan charts.
4.  Real estate  annuity-amortised mortgages: down-payment drain, rental net
                 cash-flow, property value and outstanding debt over time.
5.  Milestones   large one-off capital withdrawals (cars, renovations…).

Beyond the original it also derives **actuarial-style** outputs that suit the
portfolio: a Financial-Independence (FIRE) date from the 4 % rule, a safe-
withdrawal income, the Monte-Carlo probability of hitting the FI number, and a
one-at-a-time **sensitivity tornado** over the key drivers.

All money is in the user's own currency unit (defaults presented as €).
"""
from __future__ import annotations
import numpy as np

MONTHS = 12

# ── payroll defaults ──
DEFAULT_SOCIAL_PCT = 8.0           # generic pension + social-contribution placeholder


# ════════════════════════════════════════════════════════════════════════════
#  small helpers
# ════════════════════════════════════════════════════════════════════════════
def _num(x, default=0.0):
    try:
        if x is None or x == "":
            return float(default)
        return float(x)
    except (TypeError, ValueError):
        return float(default)


def net_salary(gross, tax_pct, social_pct=DEFAULT_SOCIAL_PCT):
    """Monthly net pay after income tax and social contributions."""
    gross = _num(gross)
    return gross * (1.0 - _num(tax_pct) / 100.0 - _num(social_pct) / 100.0)


def annuity_payment(principal, annual_rate_pct, term_years):
    """Standard fixed-rate mortgage annuity payment (per month)."""
    L = _num(principal)
    n = int(_num(term_years) * MONTHS)
    if L <= 0 or n <= 0:
        return 0.0
    r = (_num(annual_rate_pct) / 100.0) / MONTHS
    if r <= 0:
        return L / n
    return L * (r * (1 + r) ** n) / ((1 + r) ** n - 1)


# ════════════════════════════════════════════════════════════════════════════
#  core Monte-Carlo simulation
# ════════════════════════════════════════════════════════════════════════════
def simulate(p: dict) -> dict:
    """Run the full wealth projection. `p` is the JSON params from the frontend."""
    rng = np.random.default_rng(int(p.get("seed", 0)) or None)

    # ── payroll & instantaneous KPIs ──
    gross = _num(p.get("gross"), 5500)
    tax_pct = _num(p.get("tax_pct"), 25.0)
    social_pct = _num(p.get("social_pct"), DEFAULT_SOCIAL_PCT)
    salary_growth = _num(p.get("salary_growth"), 2.0)
    net0 = net_salary(gross, tax_pct, social_pct)

    # ── expenses (category, monthly, years) ──
    expenses = p.get("expenses") or []
    exp_monthly = np.array([_num(e.get("monthly")) for e in expenses], dtype=float)
    exp_years = np.array([_num(e.get("years")) for e in expenses], dtype=float)
    monthly_expense_now = float(exp_monthly[exp_years > 0].sum()) if exp_monthly.size else 0.0

    # ── investing parameters ──
    principal = _num(p.get("principal"), 25000)
    liquid = _num(p.get("liquid"), 25000)
    contrib = _num(p.get("monthly_contribution"), 1000)
    gross_return = _num(p.get("gross_return"), 7.0)
    inflation = _num(p.get("inflation"), 2.0)
    volatility = _num(p.get("volatility"), 15.0)
    fees = _num(p.get("fees"), 0.20)
    years = int(max(1, _num(p.get("years"), 15)))
    n_sims = int(min(2000, max(50, _num(p.get("n_sims"), 400))))

    total_months = years * MONTHS
    start_year = int(p.get("start_year", 0)) or _now_year()

    # real (inflation-adjusted) expected monthly return + monthly sigma
    real_annual = gross_return - fees - inflation
    mu_m = real_annual / 100.0 / MONTHS
    sigma_m = volatility / 100.0 / np.sqrt(MONTHS)
    sim_returns = rng.normal(mu_m, sigma_m, (n_sims, total_months))

    # ── real-estate cash-flow tensors ──
    re_cf, re_value, re_debt, re_drain = (np.zeros(total_months) for _ in range(4))
    for prop in (p.get("real_estate") or []):
        if not prop.get("enabled", True):
            continue
        sm = int(_num(prop.get("year")) * MONTHS)
        if sm >= total_months:
            continue
        price = _num(prop.get("price"))
        down = _num(prop.get("downpayment"))
        loan = max(0.0, price - down)
        r_m = (_num(prop.get("rate")) / 100.0) / MONTHS
        term_m = int(_num(prop.get("term_years")) * MONTHS)
        pmt = annuity_payment(loan, _num(prop.get("rate")), _num(prop.get("term_years")))
        ncf = _num(prop.get("rent")) - _num(prop.get("upkeep")) - pmt
        re_drain[sm] += down
        bal = loan
        for m in range(sm, total_months):
            re_value[m] += price
            re_cf[m] += ncf
            if bal > 0:
                bal = max(0.0, bal - (pmt - bal * r_m))
            re_debt[m] += bal

    # ── milestones (one-off capital needs) ──
    milestones = [m for m in (p.get("milestones") or []) if m.get("enabled", True)]
    mil_year = np.array([_num(m.get("years_out")) for m in milestones], dtype=float)
    mil_cap = np.array([_num(m.get("capital")) for m in milestones], dtype=float)

    # ── month-by-month evolution (loop over months, vectorised over sims) ──
    #    only the invested portfolio is stochastic; cash / real-estate are
    #    deterministic paths, so we bank the invested matrix and reduce to
    #    percentiles in a single vectorised pass after the loop.
    inv = np.full(n_sims, principal)          # invested portfolio (scenario)
    cash = float(liquid)
    base_inv, base_cash = float(principal), float(liquid)

    inv_hist = np.empty((n_sims, total_months))
    cash_path = np.empty(total_months)
    base_path = np.empty(total_months)
    annual = {}

    for m in range(total_months):
        yr_frac = m / MONTHS
        grow = (1 + salary_growth / 100.0) ** int(m / MONTHS)
        active_exp = float(exp_monthly[exp_years > yr_frac].sum()) if exp_monthly.size else 0.0
        surplus_base = net0 * grow - active_exp
        surplus = surplus_base + re_cf[m]

        # one-off drains: real-estate down payment + milestones hitting this month
        drain = re_drain[m]
        if mil_cap.size:
            hit = (mil_year > yr_frac) & (mil_year <= (m + 1) / MONTHS)
            drain += float(mil_cap[hit].sum())
        if drain > 0:
            rem = max(0.0, drain - cash)
            cash = max(0.0, cash - drain)
            inv = inv - rem

        base_inv = base_inv * (1 + mu_m) + min(surplus_base, contrib)
        base_cash += max(0.0, surplus_base - contrib)

        inv = inv * (1 + sim_returns[:, m]) + min(surplus, contrib)
        cash += max(0.0, surplus - contrib)

        inv_hist[:, m] = inv
        cash_path[m] = cash
        base_path[m] = base_inv + base_cash

        yr = start_year + int(m / MONTHS)
        a = annual.setdefault(yr, {"year": yr, "baseline": 0.0, "scenario": 0.0})
        a["baseline"] += surplus_base
        a["scenario"] += surplus

    # ── reduce the Monte-Carlo cloud to percentile fan charts (vectorised) ──
    re_equity = re_value - re_debt
    nw_hist = inv_hist + cash_path[None, :] + re_equity[None, :]
    p10, p50, p90 = (np.percentile(nw_hist, q, axis=0).tolist() for q in (10, 50, 90))
    invmed_path = np.percentile(inv_hist, 50, axis=0).tolist()
    equity_path = re_value.tolist()
    debt_path = (-re_debt).tolist()
    cash_path = cash_path.tolist()
    base_path = base_path.tolist()

    final_nw = nw_hist[:, -1]
    final_p10, final_p50, final_p90 = (float(np.percentile(final_nw, q)) for q in (10, 50, 90))

    # ── FIRE / financial-independence analytics (4 % rule) ──
    annual_expenses = monthly_expense_now * MONTHS
    fire_number = 25.0 * annual_expenses          # 4% safe-withdrawal target
    fi = _first_crossing(p50, fire_number)
    fire_month = fi
    fire_year = (start_year + fi // MONTHS) if fi is not None else None
    years_to_fi = round(fi / MONTHS, 1) if fi is not None else None
    success_prob = float(np.mean(final_nw >= fire_number)) if fire_number > 0 else 0.0
    swr_income_monthly = 0.04 * final_p50 / MONTHS

    # ── timeline labels (ISO month, monthly cadence) ──
    timeline = _month_labels(start_year, total_months)

    return {
        "kpis": {
            "net_monthly": net0,
            "free_cashflow": net0 - monthly_expense_now,
            "liquid_wealth": principal + liquid,
            "monthly_expenses": monthly_expense_now,
        },
        "salary": {
            "net": net0,
            "tax": gross * tax_pct / 100.0,
            "social": gross * social_pct / 100.0,
            "gross": gross,
        },
        "timeline": timeline,
        "wealth": {"p10": p10, "p50": p50, "p90": p90, "baseline": base_path},
        "balance_sheet": {
            "cash": cash_path, "investments": invmed_path,
            "property": equity_path, "debt": debt_path,
        },
        "cashflow": sorted(annual.values(), key=lambda d: d["year"]),
        "final": {"p10": final_p10, "p50": final_p50, "p90": final_p90},
        "fire": {
            "annual_expenses": annual_expenses,
            "fire_number": fire_number,
            "fire_month": fire_month,
            "fire_year": fire_year,
            "years_to_fi": years_to_fi,
            "success_prob": success_prob,
            "swr_income_monthly": swr_income_monthly,
        },
        "meta": {"n_sims": n_sims, "years": years, "real_return": real_annual},
    }


# ════════════════════════════════════════════════════════════════════════════
#  sensitivity tornado — one-at-a-time perturbation of the key drivers
# ════════════════════════════════════════════════════════════════════════════
_SENS_DRIVERS = [
    ("gross_return", "Market return", 0.20),
    ("monthly_contribution", "Monthly invest", 0.20),
    ("inflation", "Inflation", 0.30),
    ("volatility", "Volatility", 0.30),
    ("salary_growth", "Salary growth", 0.50),
    ("gross", "Gross salary", 0.10),
]


def sensitivity(p: dict) -> dict:
    """Tornado: ± perturb each driver, measure Δ median final net worth."""
    base = simulate({**p, "n_sims": 250})["final"]["p50"]
    bars = []
    for key, label, frac in _SENS_DRIVERS:
        v = _num(p.get(key))
        lo = simulate({**p, key: v * (1 - frac), "n_sims": 250})["final"]["p50"]
        hi = simulate({**p, key: v * (1 + frac), "n_sims": 250})["final"]["p50"]
        bars.append({
            "param": label, "low": lo, "high": hi, "base": base,
            "swing": abs(hi - lo), "pct": round(frac * 100),
        })
    bars.sort(key=lambda b: b["swing"], reverse=True)
    return {"base": base, "bars": bars}


# ════════════════════════════════════════════════════════════════════════════
#  utilities
# ════════════════════════════════════════════════════════════════════════════
def _first_crossing(path, target):
    if target <= 0:
        return None
    arr = np.asarray(path)
    idx = np.argmax(arr >= target)
    if arr.size and arr[idx] >= target:
        return int(idx)
    return None


def _now_year():
    import datetime
    return datetime.date.today().year


def _month_labels(start_year, n):
    labels = []
    y, mo = start_year, 1
    for _ in range(n):
        labels.append(f"{y:04d}-{mo:02d}")
        mo += 1
        if mo > 12:
            mo = 1
            y += 1
    return labels


def default_params() -> dict:
    """Seed config the UI starts from (English, currency-agnostic)."""
    return {
        "gross": 5500, "tax_pct": 25.0, "social_pct": DEFAULT_SOCIAL_PCT,
        "salary_growth": 2.0,
        "principal": 25000, "liquid": 25000, "monthly_contribution": 1000,
        "gross_return": 7.0, "inflation": 2.0, "volatility": 15.0, "fees": 0.20,
        "years": 25, "n_sims": 400,
        "expenses": [
            {"category": "Groceries", "monthly": 500, "years": 99},
            {"category": "Transport", "monthly": 300, "years": 99},
            {"category": "Utilities & subscriptions", "monthly": 250, "years": 99},
            {"category": "Student loan", "monthly": 250, "years": 8},
        ],
        "real_estate": [
            {"label": "Primary home", "year": 3, "price": 280000, "downpayment": 50000,
             "rate": 4.0, "term_years": 30, "rent": 0, "upkeep": 200, "enabled": True},
            {"label": "Rental property", "year": 6, "price": 140000, "downpayment": 25000,
             "rate": 4.5, "term_years": 20, "rent": 950, "upkeep": 150, "enabled": True},
        ],
        "milestones": [
            {"label": "New car", "years_out": 5, "capital": 20000, "enabled": True},
        ],
    }


if __name__ == "__main__":
    import time, json
    p = default_params()
    t0 = time.perf_counter()
    out = simulate(p)
    dt = (time.perf_counter() - t0) * 1e3
    k = out["kpis"]
    f = out["fire"]
    print(f"simulate: {dt:.1f} ms  ({out['meta']['n_sims']} sims × {out['meta']['years']}y)")
    print(f"net €{k['net_monthly']:,.0f}/mo  surplus €{k['free_cashflow']:,.0f}  "
          f"liquid €{k['liquid_wealth']:,.0f}")
    print(f"final net worth  p10 €{out['final']['p10']:,.0f}  "
          f"p50 €{out['final']['p50']:,.0f}  p90 €{out['final']['p90']:,.0f}")
    print(f"FIRE number €{f['fire_number']:,.0f}  reached: "
          f"{f['fire_year']} (in {f['years_to_fi']}y)  "
          f"P(success)={f['success_prob']*100:.0f}%  SWR €{f['swr_income_monthly']:,.0f}/mo")
    t0 = time.perf_counter()
    s = sensitivity(p)
    print(f"sensitivity: {(time.perf_counter()-t0)*1e3:.0f} ms  top driver: "
          f"{s['bars'][0]['param']} (swing €{s['bars'][0]['swing']:,.0f})")

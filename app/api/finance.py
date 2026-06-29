"""Strategic Finance router — /api/finance/*

Pure NumPy wealth Monte-Carlo engine (no pandas / yfinance / sqlite / dash).
"""
from __future__ import annotations
from app.finance import engine


def api_defaults(q, body):
    """Seed configuration the dashboard initialises from."""
    return engine.default_params()


def api_simulate(q, body):
    """Full wealth projection: KPIs, Monte-Carlo fan, balance sheet, FIRE."""
    return engine.simulate(body or engine.default_params())


def api_sensitivity(q, body):
    """One-at-a-time sensitivity tornado over the key drivers."""
    return engine.sensitivity(body or engine.default_params())


ROUTES = {
    "/api/finance/defaults": api_defaults,
    "/api/finance/simulate": api_simulate,
    "/api/finance/sensitivity": api_sensitivity,
}

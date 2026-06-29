"""Experiments registry router — /api/experiments/*

Read-only window onto the in-memory run history (app/experiments/registry.py) —
list past runs, or fetch one by id. Writes happen wherever the run is created
(e.g. the optimiser), not here.
"""
from __future__ import annotations
from app.experiments import registry


def api_list(q, body):
    project = (body or {}).get("project") or q.get("project", [None])[0]
    return {"experiments": registry.list_all(project=project)}


def api_get(q, body):
    exp_id = (body or {}).get("id") or q.get("id", [None])[0]
    return registry.get(exp_id)


ROUTES = {
    "/api/experiments/list": api_list,
    "/api/experiments/get":  api_get,
}

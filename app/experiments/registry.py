"""
Experiment registry — lightweight in-memory store for all runs.

A poor man's experiment tracker. When something long-ish runs (a rod-pattern
optimisation, a training job) it gets a record here with a status and result, so the
UI can show a little "recent runs" history. In-memory on purpose — it resets on
restart, which is fine for a demo and keeps the zero-dependency promise (no sqlite).
"""
from __future__ import annotations
import threading
import time
import uuid

_LOCK   = threading.Lock()        # registry is touched from many request threads
_STORE: dict = {}                 # exp_id -> record


def create(project: str, exp_type: str, name: str, params: dict) -> dict:
    """Open a new run in the 'running' state. Pair with finish()/update()."""
    exp_id = str(uuid.uuid4())[:8]      # short id is plenty for a demo store
    rec = {
        "id":         exp_id,
        "project":    project,
        "type":       exp_type,
        "name":       name,
        "params":     params,
        "status":     "running",
        "created_at": round(time.time(), 3),
        "result":     None,
    }
    with _LOCK:
        _STORE[exp_id] = rec
    return rec


def finish(exp_id: str, result: dict) -> dict:
    with _LOCK:
        if exp_id in _STORE:
            _STORE[exp_id]["status"] = "done"
            _STORE[exp_id]["result"] = result
            _STORE[exp_id]["finished_at"] = round(time.time(), 3)
    return _STORE.get(exp_id, {})


def update(exp_id: str, status: str) -> dict:
    with _LOCK:
        if exp_id in _STORE:
            _STORE[exp_id]["status"] = status
    return _STORE.get(exp_id, {})


def get(exp_id: str) -> dict:
    with _LOCK:
        return dict(_STORE.get(exp_id, {}))


def list_all(project: str = None, limit: int = 50) -> list:
    with _LOCK:
        items = list(_STORE.values())
    if project:
        items = [i for i in items if i["project"] == project]
    return sorted(items, key=lambda x: x["created_at"], reverse=True)[:limit]

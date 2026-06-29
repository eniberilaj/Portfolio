"""Neural Network project router — /api/neural/*

Wiring for the ML labs: generate a dataset, create a model, kick off training, then
poll status/metrics. The actual maths lives in app/neural/{datasets,training}.py —
this file only unpacks the request body and forwards it with sane defaults.
"""
from __future__ import annotations
from app.neural import datasets, training


def api_dataset_generate(q, body):
    b = body or {}
    return datasets.generate(
        ds_type    = b.get("ds_type") or b.get("type", "two_spirals"),
        n          = int(b.get("n", 600)),
        noise      = float(b.get("noise", 0.12)),
        seed       = int(b.get("seed", 0)),
        classes    = int(b.get("classes", 3)),
    )


def api_dataset_list(q, body):
    return {"datasets": datasets.list_all(), "types": datasets.TYPES}


def api_model_create(q, body):
    b = body or {}
    return training.create_model(
        hidden     = b.get("hidden") or b.get("hidden_layers", [16, 16]),
        activation = b.get("activation", "tanh"),
        seed       = int(b.get("seed", 0)),
        name       = b.get("name"),
    )


def api_train_start(q, body):
    b          = body or {}
    model_id   = b.get("model_id")
    # Fall back to the most recently generated dataset if the client didn't name one
    # (handy when you just hit "generate" then "train" without tracking ids).
    dataset_id = b.get("dataset_id") or datasets.latest_id()
    if not dataset_id:
        raise ValueError("No dataset yet — call /api/neural/dataset/generate first")
    return training.start_training(
        model_id   = model_id,
        dataset_id = dataset_id,
        epochs     = int(b.get("epochs", 150)),
        lr         = float(b.get("lr", 0.01)),
        batch_size = int(b.get("batch_size", 32)),
        optimizer  = b.get("optimizer", "adam"),
    )


def api_train_status(q, body):
    b      = body or {}
    run_id = b.get("run_id") or (q.get("run_id", [None])[0])
    return training.status(run_id)


def api_metrics(q, body):
    b      = body or {}
    run_id = b.get("run_id") or (q.get("run_id", [None])[0])
    return training.metrics(run_id)


def api_compare(q, body):
    b       = body or {}
    run_ids = b.get("run_ids") or q.get("ids", [""])[0].split(",")
    return training.compare([str(i).strip() for i in run_ids if str(i).strip()])


ROUTES = {
    "/api/neural/dataset/generate": api_dataset_generate,
    "/api/neural/dataset/list":     api_dataset_list,
    "/api/neural/model/create":     api_model_create,
    "/api/neural/train/start":      api_train_start,
    "/api/neural/train/status":     api_train_status,
    "/api/neural/metrics":          api_metrics,
    "/api/neural/compare":          api_compare,
}

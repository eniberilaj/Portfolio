"""
Manual-backprop MLP in pure NumPy — no ML frameworks.
Adam and SGD optimisers, tanh/relu/sigmoid activations.
"""
from __future__ import annotations
import threading
import time
import uuid
import numpy as np
from app.neural import datasets as DS


_MODELS: dict = {}
_RUNS:   dict = {}
_LOCK = threading.Lock()


# ── Activation functions ──────────────────────────────────────────────────

def _act(z, name):
    if name == "tanh":    return np.tanh(z)
    if name == "relu":    return np.maximum(0, z)
    if name == "sigmoid": return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
    return z   # linear

def _dact(z, name):
    if name == "tanh":    return 1.0 - np.tanh(z)**2
    if name == "relu":    return (z > 0).astype(float)
    if name == "sigmoid":
        s = _act(z, "sigmoid")
        return s * (1.0 - s)
    return np.ones_like(z)


# ── Weight initialisation ─────────────────────────────────────────────────

def _init_weights(layer_sizes, rng):
    weights, biases = [], []
    for i in range(len(layer_sizes) - 1):
        fan_in = layer_sizes[i]
        fan_out = layer_sizes[i+1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights.append(rng.standard_normal((fan_in, fan_out)) * std)
        biases.append(np.zeros(fan_out))
    return weights, biases


# ── Forward pass ──────────────────────────────────────────────────────────

def _forward(X, weights, biases, act):
    activations = [X]
    pre_acts    = []
    a = X
    for W, b in zip(weights[:-1], biases[:-1]):
        z = a @ W + b
        pre_acts.append(z)
        a = _act(z, act)
        activations.append(a)
    # last layer: softmax for classification, linear for regression
    z = a @ weights[-1] + biases[-1]
    pre_acts.append(z)
    # softmax
    z_max = z.max(axis=1, keepdims=True)
    ez    = np.exp(z - z_max)
    out   = ez / ez.sum(axis=1, keepdims=True)
    activations.append(out)
    return activations, pre_acts


# ── Backward pass ─────────────────────────────────────────────────────────

def _backward(activations, pre_acts, y_onehot, weights, biases, act):
    n      = len(activations) - 1
    dW     = [None] * n
    db     = [None] * n

    # output layer gradient (cross-entropy + softmax)
    delta = activations[-1] - y_onehot   # (N, C)

    for i in range(n-1, -1, -1):
        dW[i] = activations[i].T @ delta / len(activations[0])
        db[i] = delta.mean(axis=0)
        if i > 0:
            delta = (delta @ weights[i].T) * _dact(pre_acts[i-1], act)

    return dW, db


# ── Adam update ───────────────────────────────────────────────────────────

def _adam_step(weights, biases, dW, db, m_W, v_W, m_b, v_b, t, lr,
               b1=0.9, b2=0.999, eps=1e-8):
    bc1 = 1.0 - b1**t
    bc2 = 1.0 - b2**t
    for i in range(len(weights)):
        m_W[i] = b1*m_W[i] + (1-b1)*dW[i]
        v_W[i] = b2*v_W[i] + (1-b2)*dW[i]**2
        mh = m_W[i] / bc1
        vh = v_W[i] / bc2
        weights[i] -= lr * mh / (np.sqrt(vh) + eps)

        m_b[i] = b1*m_b[i] + (1-b1)*db[i]
        v_b[i] = b2*v_b[i] + (1-b2)*db[i]**2
        mh = m_b[i] / bc1
        vh = v_b[i] / bc2
        biases[i]  -= lr * mh / (np.sqrt(vh) + eps)


# ── Public API ────────────────────────────────────────────────────────────

def create_model(hidden=(16, 16), activation="tanh", seed=0, name=None) -> dict:
    model_id = str(uuid.uuid4())[:8]
    rng      = np.random.default_rng(seed)
    _MODELS[model_id] = {
        "id": model_id, "hidden": list(hidden),
        "activation": activation, "seed": seed,
        "name": name or f"mlp_{model_id}",
        "rng": rng,
    }
    return {"id": model_id, "model_id": model_id,
            "hidden": list(hidden), "activation": activation, "name": name}


def start_training(model_id, dataset_id, epochs=150, lr=0.01,
                   batch_size=32, optimizer="adam") -> dict:
    run_id = str(uuid.uuid4())[:8]
    with _LOCK:
        _RUNS[run_id] = {
            "id": run_id, "run_id": run_id,
            "model_id": model_id, "dataset_id": dataset_id,
            "status": "running", "epoch": 0, "epochs": epochs,
            "history": {"train_loss": [], "val_loss": [],
                        "train_acc": [], "val_acc": []},
            "started_at": time.time(),
        }

    def _train():
        ds   = DS.get(dataset_id)
        X, y = np.array(ds["X"]), np.array(ds["y"])
        mdl  = _MODELS.get(model_id, {})

        # regression or classification?
        is_reg = ds["type"] == "damped_wave"
        n_out  = 1 if is_reg else int(y.max()) + 1
        n_in   = X.shape[1]

        # normalise inputs
        mu, sd = X.mean(0), X.std(0) + 1e-8
        X = (X - mu) / sd

        # train / val split
        N     = len(X)
        idx   = np.random.permutation(N)
        tr    = idx[:int(0.8*N)]
        va    = idx[int(0.8*N):]
        Xtr, ytr = X[tr], y[tr]
        Xva, yva = X[va], y[va]

        # one-hot
        def oh(y_, nc):
            Y = np.zeros((len(y_), nc))
            Y[np.arange(len(y_)), y_.astype(int)] = 1.0
            return Y

        hidden = mdl.get("hidden", [16, 16])
        act    = mdl.get("activation", "tanh")
        rng    = mdl.get("rng", np.random.default_rng(0))

        sizes  = [n_in] + list(hidden) + [n_out]
        W, b   = _init_weights(sizes, rng)

        # Adam state
        m_W = [np.zeros_like(w) for w in W]
        v_W = [np.zeros_like(w) for w in W]
        m_b = [np.zeros_like(bi) for bi in b]
        v_b = [np.zeros_like(bi) for bi in b]

        t_adam = 0

        for ep in range(1, epochs + 1):
            perm = np.random.permutation(len(Xtr))
            for start in range(0, len(Xtr), batch_size):
                idx2   = perm[start:start+batch_size]
                Xb     = Xtr[idx2]
                yb     = ytr[idx2]
                yb_oh  = oh(yb, n_out)
                acts, pre = _forward(Xb, W, b, act)
                dW, db    = _backward(acts, pre, yb_oh, W, b, act)
                t_adam   += 1
                if optimizer == "adam":
                    _adam_step(W, b, dW, db, m_W, v_W, m_b, v_b, t_adam, lr)
                else:
                    for i in range(len(W)):
                        W[i] -= lr * dW[i]
                        b[i] -= lr * db[i]

            # epoch metrics
            acts_tr, _ = _forward(Xtr, W, b, act)
            pred_tr    = acts_tr[-1]
            loss_tr    = float(-np.mean(oh(ytr, n_out) * np.log(pred_tr + 1e-9)))
            acc_tr     = float(np.mean(pred_tr.argmax(1) == ytr.astype(int)))
            acts_va, _ = _forward(Xva, W, b, act)
            pred_va    = acts_va[-1]
            loss_va    = float(-np.mean(oh(yva, n_out) * np.log(pred_va + 1e-9)))
            acc_va     = float(np.mean(pred_va.argmax(1) == yva.astype(int)))
            with _LOCK:
                run = _RUNS[run_id]
                run["epoch"] = ep
                run["history"]["train_loss"].append(round(loss_tr, 5))
                run["history"]["val_loss"].append(round(loss_va, 5))
                run["history"]["train_acc"].append(round(acc_tr, 4))
                run["history"]["val_acc"].append(round(acc_va, 4))

        # decision boundary
        gx = np.linspace(-3, 3, 60)
        gy = np.linspace(-3, 3, 60)
        GX, GY = np.meshgrid(gx, gy)
        G = np.c_[GX.ravel(), GY.ravel()]
        if n_in == 1: G = G[:, :1]
        G = (G - mu[:n_in]) / sd[:n_in]
        ga, _ = _forward(G, W, b, act)
        boundary = ga[-1].argmax(1).reshape(60, 60)

        # PCA embedding
        a_all, _ = _forward((X - 0) / 1, W, b, act)
        hidden_rep = a_all[-2]
        if hidden_rep.shape[1] >= 2:
            U, S, Vt = np.linalg.svd(hidden_rep - hidden_rep.mean(0), full_matrices=False)
            emb = U[:, :2] * S[:2]
        else:
            emb = hidden_rep

        with _LOCK:
            run = _RUNS[run_id]
            run["status"]          = "done"
            run["boundary"]        = np.round(boundary, 0).tolist()
            run["embedding"]       = np.round(emb[:200], 4).tolist()
            run["labels_emb"]      = ytr[:200].tolist()
            run["norm_mu"]         = mu.tolist()
            run["norm_sd"]         = sd.tolist()
            run["final_train_acc"] = run["history"]["train_acc"][-1]
            run["final_val_acc"]   = run["history"]["val_acc"][-1]

    t = threading.Thread(target=_train, daemon=True)
    t.start()
    return {"run_id": run_id, "id": run_id, "status": "running"}


def status(run_id) -> dict:
    with _LOCK:
        r = _RUNS.get(run_id)
    if not r: raise KeyError(f"run {run_id!r} not found")
    return {
        "run_id":  r["id"],
        "id":      r["id"],
        "status":  r["status"],
        "epoch":   r["epoch"],
        "epochs":  r["epochs"],
        "history": r["history"],
    }


def metrics(run_id) -> dict:
    with _LOCK:
        r = dict(_RUNS.get(run_id, {}))
    if not r: raise KeyError(f"run {run_id!r} not found")
    return {
        "run_id":          r["id"],
        "status":          r["status"],
        "boundary":        r.get("boundary"),
        "embedding":       r.get("embedding"),
        "labels_emb":      r.get("labels_emb"),
        "history":         r["history"],
        "final_train_acc": r.get("final_train_acc"),
        "final_val_acc":   r.get("final_val_acc"),
    }


def compare(run_ids: list) -> dict:
    curves, rows = {}, []
    for rid in run_ids:
        with _LOCK:
            r = _RUNS.get(rid)
        if not r: continue
        h = r["history"]
        curves[rid] = {"train_loss": h["train_loss"], "val_loss": h["val_loss"],
                       "train_acc":  h["train_acc"],  "val_acc":  h["val_acc"]}
        rows.append({
            "run_id":          rid,
            "model_id":        r.get("model_id"),
            "final_val_acc":   r.get("final_val_acc"),
            "final_train_acc": r.get("final_train_acc"),
            "epochs":          r["epochs"],
        })
    return {"curves": curves, "rows": rows}

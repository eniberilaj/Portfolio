"""3D aerodynamics solver router — /api/cfd/*

Runs the NumPy LES-projection engine and streams the raw float32 fields back to
the browser as one packed binary buffer:

    [uint32 headerLen][headerLen bytes UTF-8 JSON meta][float32 RGBA volume]

The RGBA volume is laid out x-fastest (R=u, G=v, B=w, A=p) so the frontend can
drop it straight into a THREE.DataTexture3D with no per-voxel parsing.
"""
from __future__ import annotations
import json
import struct
import math
import base64
from pathlib import Path
import numpy as np
from app.physics import aero3d

_GRIDS = {"low": (48, 28, 28), "med": (64, 36, 36), "high": (80, 44, 44)}
_CAD = Path(__file__).parent.parent.parent / "CAD"
_MAX_OBJ = 25 * 1024 * 1024     # OBJ files larger than this are too big to ship to the browser


def api_models(q, body):
    """List CAD models that are usable in the zero-dependency browser pipeline (OBJ, not huge)."""
    out = []
    if _CAD.is_dir():
        for f in sorted(_CAD.iterdir()):
            if f.suffix.lower() == ".obj" and f.stat().st_size <= _MAX_OBJ:
                out.append({"name": f.stem.replace("_", " "), "file": f.name,
                            "mb": round(f.stat().st_size / 1e6, 1)})
    # report what we could NOT load so the UI can explain it
    skipped = []
    if _CAD.is_dir():
        for f in sorted(_CAD.iterdir()):
            if f.suffix.lower() in (".blend", ".fbx"):
                skipped.append({"file": f.name, "why": f.suffix.lower()[1:] + " not browser-loadable"})
            elif f.suffix.lower() == ".obj" and f.stat().st_size > _MAX_OBJ:
                skipped.append({"file": f.name, "why": "%d MB, too large" % (f.stat().st_size // 1048576)})
    return {"models": out, "skipped": skipped}


def api_solve(q, body):
    b = body or {}
    U = float(b.get("U", 1.0))
    yaw = math.radians(float(b.get("yaw", 0.0)))
    ride = float(b.get("ride", 0.40))
    nx, ny, nz = _GRIDS.get(b.get("fidelity", "med"), _GRIDS["med"])

    mask = None
    if b.get("mask"):
        try:
            raw = np.frombuffer(base64.b64decode(b["mask"]), dtype=np.uint8)
            if raw.size == nx * ny * nz:                      # frontend layout: (z*ny+y)*nx+x
                mask = raw.reshape(nz, ny, nx).transpose(2, 1, 0).astype(bool)
        except Exception:
            mask = None

    u, v, w, p, meta = aero3d.solve(nx, ny, nz, U=U, yaw=yaw, ride=ride, mask=mask)
    meta["custom"] = mask is not None

    sp = np.sqrt(u * u + v * v + w * w)
    meta.update(umax=float(sp.max()),
                pmin=float(p.min()), pmax=float(p.max()),
                bytesPerVoxel=16)

    # interleave to RGBA, reorder to x-fastest for DataTexture3D(width=nx, height=ny, depth=nz)
    vol = np.stack([u, v, w, p], axis=-1).transpose(2, 1, 0, 3).ravel().astype("<f4")
    hdr = json.dumps(meta).encode("utf-8")
    return struct.pack("<I", len(hdr)) + hdr + vol.tobytes()


ROUTES = {"/api/cfd/solve": api_solve, "/api/cfd/models": api_models}

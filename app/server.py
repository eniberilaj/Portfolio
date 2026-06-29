"""Scientific Systems Lab -- HTTP API server.

The whole backend is this one stdlib ``ThreadingHTTPServer`` — no Flask, no
FastAPI, no uvicorn. Each project module exposes a ``ROUTES`` dict mapping a URL
path to a ``fn(query, body) -> result`` callable; I just merge them all into one
table and dispatch on the path. "Threading" matters: every request (and every
WebSocket stream) gets its own thread, so a slow CFD solve can't freeze the page.
"""
from __future__ import annotations
import os
import json
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
from app.api import reactor, neural, experiments, cfd, spacelab, finance
from app import ws

STATIC = Path(__file__).parent / "static"          # the single-file SPA lives here
CAD = Path(__file__).parent.parent / "CAD"          # optional local car meshes

# Collect every project's REST routes into one dispatch table. Adding a new
# project = write a module with a ROUTES dict and list it here; nothing else changes.
ROUTES: dict = {}
for _router in (reactor, neural, experiments, cfd, spacelab, finance):
    ROUTES.update(_router.ROUTES)

# WebSocket routes are separate — they're handled via the HTTP/1.1 Upgrade
# handshake (see _maybe_ws), not the normal JSON request path.
WS_ROUTES: dict = {}
for _wsrouter in (spacelab,):
    WS_ROUTES.update(getattr(_wsrouter, "WS_ROUTES", {}))


class NpEncoder(json.JSONEncoder):
    """The engines return NumPy types everywhere; the stdlib json module chokes on
    them. This teaches it to serialise np scalars/arrays as plain Python."""
    def default(self, o):
        if isinstance(o, np.integer):  return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray):  return o.tolist()
        return super().default(o)


MIME = {".html":"text/html",".js":"text/javascript",".css":"text/css",
        ".svg":"image/svg+xml",".png":"image/png",".ico":"image/x-icon"}


class Handler(BaseHTTPRequestHandler):
    server_version = "SSL/2.0"
    def log_message(self, fmt, *args): pass

    def _json(self, obj, code=200):
        data = json.dumps(obj, cls=NpEncoder).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _bin(self, data, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _route(self, body=None):
        """The one dispatcher for both GET and POST: API route first, then files."""
        u = urlparse(self.path)
        q = parse_qs(u.query)
        fn = ROUTES.get(u.path)
        if fn:
            # An API endpoint. It may return bytes (a packed binary field, e.g. the
            # CFD volume) or a JSON-able dict — pick the response type accordingly.
            try:
                out = fn(q, body)
                self._bin(out) if isinstance(out, (bytes, bytearray)) else self._json(out)
            except Exception as exc:
                traceback.print_exc()
                self._json({"error": str(exc)}, 500)
            return
        # Optional /cad/<file> passthrough for locally-stored car meshes. The
        # `in cf.parents` check keeps a crafted path from escaping the CAD folder.
        if u.path.startswith("/cad/"):
            cf = (CAD / u.path[5:]).resolve()
            if cf.is_file() and CAD.resolve() in cf.parents:
                data = cf.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "model/obj")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
        # Otherwise serve a static file (default to index.html — the SPA entry).
        # Same parent-containment guard against directory traversal.
        rel = u.path.lstrip("/") or "index.html"
        f = (STATIC / rel).resolve()
        if f.is_file() and STATIC.resolve() in f.parents:
            data = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", MIME.get(f.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self._json({"error": "not found"}, 404)

    def _maybe_ws(self) -> bool:
        """Additive hook: if this GET is a WebSocket upgrade for a known stream route,
        hand the connection to its streamer and return True. Otherwise return False so
        the normal HTTP routing below runs completely unchanged."""
        if not ws.is_ws_upgrade(self.headers):
            return False
        u = urlparse(self.path)
        fn = WS_ROUTES.get(u.path)
        if not fn:
            return False
        try:
            fn(self, parse_qs(u.query))
        except Exception:
            traceback.print_exc()
        return True

    def do_GET(self):
        # Try the WebSocket upgrade first; if it's an ordinary GET, fall through.
        if self._maybe_ws():
            return
        self._route()

    def do_POST(self):
        # POSTs carry a JSON body (simulation params). Read exactly Content-Length
        # bytes and hand the parsed dict to the same dispatcher GET uses.
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        self._route(body)

    def do_OPTIONS(self):
        # CORS preflight — answer the browser's permission check before a POST.
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def main(host=None, port=None):
    # Bind from the environment so hosting platforms (Render/Railway/Fly/…) can
    # inject $PORT; defaults keep `python run.py` working locally as before.
    host = host or os.environ.get("HOST", "0.0.0.0")
    port = int(port or os.environ.get("PORT", 8050))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"Scientific Systems Lab  ->  http://{host}:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()

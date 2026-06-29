"""
Minimal RFC 6455 WebSocket — pure Python standard library, zero dependencies.

The site's backend is a stdlib ``ThreadingHTTPServer`` (no FastAPI / uvicorn). To keep
the project's "zero PyPI deps · stdlib + NumPy" promise intact while still offering a
real streaming socket, the WebSocket protocol is implemented here by hand:

  • handshake()  — completes the HTTP/1.1 ``Upgrade: websocket`` opening handshake
                   (Sec-WebSocket-Accept = base64(sha1(key + GUID))).
  • WebSocket    — a thin frame codec over the handler's raw socket: send text/binary,
                   receive (unmasking client frames), respond to ping, honour close.

Because the server is threaded, each upgraded connection owns its thread and can run a
long-lived send loop without blocking anything else.
"""
from __future__ import annotations
import base64
import hashlib
import struct

_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"   # RFC 6455 magic value

# opcodes
OP_CONT = 0x0
OP_TEXT = 0x1
OP_BIN = 0x2
OP_CLOSE = 0x8
OP_PING = 0x9
OP_PONG = 0xA


def is_ws_upgrade(headers) -> bool:
    """True if the request headers ask for a WebSocket upgrade."""
    return (headers.get("Upgrade", "").lower() == "websocket"
            and "upgrade" in headers.get("Connection", "").lower())


def handshake(handler) -> "WebSocket | None":
    """Complete the opening handshake on a BaseHTTPRequestHandler.

    Returns a ready WebSocket bound to the handler's buffered streams, or None.
    """
    key = handler.headers.get("Sec-WebSocket-Key")
    if not key:
        return None
    accept = base64.b64encode(
        hashlib.sha1((key + _GUID).encode()).digest()).decode()
    handler.send_response(101, "Switching Protocols")
    handler.send_header("Upgrade", "websocket")
    handler.send_header("Connection", "Upgrade")
    handler.send_header("Sec-WebSocket-Accept", accept)
    handler.end_headers()
    handler.wfile.flush()
    return WebSocket(handler.rfile, handler.wfile)


class WebSocket:
    """Frame-level WebSocket bound to the handler's buffered rfile/wfile.

    Using the buffered streams (rather than the raw socket) avoids losing bytes that
    BaseHTTPRequestHandler may have already pulled into rfile during request parsing.
    Sends are serialised with a lock so a reader thread and a writer thread can share
    one connection (control messages in, position stream out).
    """

    def __init__(self, rfile, wfile):
        import threading
        self.rfile = rfile
        self.wfile = wfile
        self.open = True
        self._wlock = threading.Lock()

    # ── outbound ──────────────────────────────────────────────────────────
    def _send(self, opcode: int, payload: bytes):
        # Build the 2+ byte frame header by hand (RFC 6455 §5.2).
        n = len(payload)
        hdr = bytearray()
        hdr.append(0x80 | opcode)                     # byte 0: FIN bit (0x80) | opcode
        # byte 1: mask bit (0 for us — servers never mask) | payload length.
        # 0–125 fits inline; 126 means "next 2 bytes are the length"; 127 means 8.
        if n < 126:
            hdr.append(n)
        elif n < 65536:
            hdr.append(126)
            hdr += struct.pack("!H", n)               # 16-bit big-endian length
        else:
            hdr.append(127)
            hdr += struct.pack("!Q", n)               # 64-bit big-endian length
        try:
            with self._wlock:
                self.wfile.write(bytes(hdr) + payload)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError, ValueError):
            self.open = False

    def send_bytes(self, data: bytes):
        self._send(OP_BIN, data)

    def send_text(self, text: str):
        self._send(OP_TEXT, text.encode("utf-8"))

    def close(self, code: int = 1000):
        if self.open:
            self._send(OP_CLOSE, struct.pack("!H", code))
            self.open = False

    # ── inbound ───────────────────────────────────────────────────────────
    def _recv_exact(self, n: int) -> bytes:
        buf = self.rfile.read(n)
        if not buf or len(buf) < n:
            raise ConnectionError("peer closed")
        return buf

    def recv(self):
        """Read one frame. Returns (opcode, payload) or None on close/error.

        Transparently answers PING with PONG and unmasks client→server payloads
        (clients MUST mask, per the spec)."""
        try:
            b0, b1 = self._recv_exact(2)
        except (ConnectionError, OSError):
            self.open = False
            return None
        # unpack the same bit-fields we packed in _send, in reverse
        opcode = b0 & 0x0F      # low nibble of byte 0
        masked = b1 & 0x80      # top bit of byte 1 — clients MUST set this
        length = b1 & 0x7F      # low 7 bits = length, or the 126/127 sentinel
        try:
            if length == 126:
                length = struct.unpack("!H", self._recv_exact(2))[0]
            elif length == 127:
                length = struct.unpack("!Q", self._recv_exact(8))[0]
            mask = self._recv_exact(4) if masked else b"\x00\x00\x00\x00"
            data = bytearray(self._recv_exact(length))
        except (ConnectionError, OSError):
            self.open = False
            return None
        # client payloads are XOR-masked with a rotating 4-byte key; undo it
        if masked:
            for k in range(length):
                data[k] ^= mask[k & 3]
        # control frames: answer a ping with a pong, then keep waiting for real data
        if opcode == OP_PING:
            self._send(OP_PONG, bytes(data))
            return self.recv()
        if opcode == OP_CLOSE:
            self.open = False
            return None
        return opcode, bytes(data)

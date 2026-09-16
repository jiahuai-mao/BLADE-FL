from __future__ import annotations

import json
import socket
import struct
import threading
import time

import numpy as np


_HOST_CACHE = {}
_HOST_CACHE_LOCK = threading.Lock()


def send_packet(sock, header, vector=None):
    payload = b"" if vector is None else np.asarray(vector, dtype=np.float32).tobytes(order="C")
    message = dict(header)
    message["payload_nbytes"] = len(payload)
    encoded = json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")
    sock.sendall(struct.pack("!I", len(encoded)))
    sock.sendall(encoded)
    if payload:
        sock.sendall(payload)
    return 4 + len(encoded) + len(payload)


def receive_packet(sock):
    header_size = struct.unpack("!I", receive_exact(sock, 4))[0]
    if header_size > 8 * 1024 * 1024:
        raise ValueError("unreasonable protocol header size")
    encoded = receive_exact(sock, header_size)
    header = json.loads(encoded.decode("utf-8"))
    payload_size = int(header.get("payload_nbytes", 0))
    payload = receive_exact(sock, payload_size) if payload_size else b""
    vector = np.frombuffer(payload, dtype=np.float32).copy() if payload else None
    return header, vector, 4 + header_size + payload_size


def receive_exact(sock, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed with %d bytes remaining" % remaining)
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def rpc(host, port, header, vector=None, timeout=120.0, connect_retries=7):
    started = monotonic_time()
    address = resolve_host(host)
    connect_timeout = min(float(timeout), 20.0)
    for attempt in range(connect_retries + 1):
        try:
            sock = socket.create_connection((address, int(port)), timeout=connect_timeout)
            break
        except OSError:
            if attempt >= connect_retries:
                raise
            time.sleep(min(2.0 * (2**attempt), 30.0))
    with sock:
        sock.settimeout(timeout)
        request_bytes = send_packet(sock, header, vector)
        response, response_vector, response_bytes = receive_packet(sock)
    if not response.get("ok", False):
        raise RuntimeError(response.get("error", "remote operation failed"))
    return response, response_vector, {
        "request_bytes": request_bytes,
        "response_bytes": response_bytes,
        "total_bytes": request_bytes + response_bytes,
        "elapsed_sec": monotonic_time() - started,
    }


def resolve_host(host, retries=5):
    try:
        socket.inet_aton(str(host))
        return str(host)
    except OSError:
        pass
    with _HOST_CACHE_LOCK:
        cached = _HOST_CACHE.get(str(host))
        if cached is not None:
            return cached
        for attempt in range(retries + 1):
            try:
                address = socket.gethostbyname(str(host))
                _HOST_CACHE[str(host)] = address
                return address
            except socket.gaierror:
                if attempt >= retries:
                    raise
                time.sleep(0.2 * (attempt + 1))


def monotonic_time():
    import time

    return time.monotonic()

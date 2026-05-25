#!/usr/bin/env python3
"""Small OpenAI-compatible request router for pod-local vLLM replicas."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass
class Backend:
    name: str
    base_url: str
    semaphore: threading.BoundedSemaphore
    in_flight: int = 0


@dataclass
class RouterState:
    backends: list[Backend]
    api_key: str
    request_timeout: float
    acquire_timeout: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    next_backend: int = 0

    def acquire_backend(self) -> Backend | None:
        deadline = time.monotonic() + self.acquire_timeout
        while True:
            with self.lock:
                start = self.next_backend
                for offset in range(len(self.backends)):
                    index = (start + offset) % len(self.backends)
                    backend = self.backends[index]
                    if backend.semaphore.acquire(blocking=False):
                        backend.in_flight += 1
                        self.next_backend = (index + 1) % len(self.backends)
                        return backend
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def release_backend(self, backend: Backend) -> None:
        with self.lock:
            backend.in_flight -= 1
            backend.semaphore.release()


class RouterHandler(BaseHTTPRequestHandler):
    state: ClassVar[RouterState]
    server_version = "OpenAIVLLMRouter/0.1"

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send_json(200, {"ok": True})
            return
        if self.path.startswith("/v1/models"):
            self._forward_to_backend(self.state.backends[0], body=None)
            return
        self._send_json(404, {"error": {"message": f"unsupported path: {self.path}"}})

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/"):
            self._send_json(404, {"error": {"message": f"unsupported path: {self.path}"}})
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        backend = self.state.acquire_backend()
        if backend is None:
            self._send_json(
                503,
                {
                    "error": {
                        "message": "all vLLM backends are at their configured concurrency limit",
                        "type": "router_overloaded",
                    }
                },
            )
            return
        try:
            self._forward_to_backend(backend, body=body)
        finally:
            self.state.release_backend(backend)

    def _forward_to_backend(self, backend: Backend, body: bytes | None) -> None:
        url = backend.base_url + self.path.removeprefix("/v1")
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
            and key.lower() not in {"host", "content-length"}
        }
        if self.state.api_key and "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {self.state.api_key}"
        request = urllib.request.Request(
            url,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            with urllib.request.urlopen(
                request, timeout=self.state.request_timeout
            ) as response:
                response_body = response.read()
                self.send_response(response.status)
                self._copy_response_headers(response.headers.items())
                self.send_header("X-VLLM-Backend", backend.name)
                self.end_headers()
                self.wfile.write(response_body)
        except urllib.error.HTTPError as exc:
            response_body = exc.read()
            self.send_response(exc.code)
            self._copy_response_headers(exc.headers.items())
            self.send_header("X-VLLM-Backend", backend.name)
            self.end_headers()
            self.wfile.write(response_body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._send_json(
                502,
                {
                    "error": {
                        "message": f"{backend.name} request failed: {exc}",
                        "type": "backend_unavailable",
                    }
                },
            )

    def _copy_response_headers(self, headers: object) -> None:
        for key, value in headers:
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            if key.lower() in {"content-length", "content-encoding"}:
                continue
            self.send_header(key, value)

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        print(
            f"{self.log_date_time_string()} {self.address_string()} {fmt % args}",
            file=sys.stderr,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Route OpenAI-compatible requests across pod-local vLLM replicas."
    )
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=8090)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--per-backend-concurrency", type=int, default=24)
    parser.add_argument("--request-timeout", type=float, default=600)
    parser.add_argument("--acquire-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if args.per_backend_concurrency <= 0:
        raise ValueError("--per-backend-concurrency must be positive")
    return args


def build_state(args: argparse.Namespace) -> RouterState:
    backends = [
        Backend(
            name=f"vllm-{index}",
            base_url=backend.rstrip("/"),
            semaphore=threading.BoundedSemaphore(args.per_backend_concurrency),
        )
        for index, backend in enumerate(args.backend)
    ]
    return RouterState(
        backends=backends,
        api_key=args.api_key,
        request_timeout=args.request_timeout,
        acquire_timeout=args.acquire_timeout,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    RouterHandler.state = build_state(args)
    server = ThreadingHTTPServer((args.listen_host, args.listen_port), RouterHandler)
    print(
        "router_ready "
        + json.dumps(
            {
                "listen_host": args.listen_host,
                "listen_port": args.listen_port,
                "backend_count": len(args.backend),
                "per_backend_concurrency": args.per_backend_concurrency,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

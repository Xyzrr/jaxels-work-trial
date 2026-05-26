#!/usr/bin/env python3
"""Small OpenAI-compatible request router for pod-local vLLM replicas.

The current Qwen2.5 OpenHands eval presets start one vLLM server per GPU and
then expose this router as the single OpenAI-compatible base URL that OpenHands
talks to. Each OpenHands worker drives one SWE-bench task attempt, so the router
is part of the eval-serving contract: it spreads task requests across replicas
and prevents any one vLLM process from receiving more concurrent generations
than the preset says it can handle.

This file does not interpret model outputs or alter prompts. It only forwards
HTTP requests and preserves vLLM's OpenAI-style responses. That boundary matters
because eval quality should be attributed to the model/OpenHands stack, not to
hidden response transformations in the router.
"""

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
    """One pod-local vLLM server plus its concurrency accounting."""

    name: str
    base_url: str
    semaphore: threading.BoundedSemaphore
    in_flight: int = 0


@dataclass
class RouterState:
    """Mutable router state shared by all HTTP handler threads."""

    backends: list[Backend]
    api_key: str
    request_timeout: float
    acquire_timeout: float
    lock: threading.Lock = field(default_factory=threading.Lock)
    next_backend: int = 0

    def acquire_backend(self) -> Backend | None:
        """Reserve capacity on a backend, preferring round-robin distribution.

        vLLM replicas are expensive model-serving processes, and each concurrent
        request can hold GPU memory for an entire OpenHands action. The bounded
        semaphore enforces the preset's per-replica worker budget instead of
        letting a burst of tasks overload one GPU.
        """

        deadline = time.monotonic() + self.acquire_timeout
        while True:
            with self.lock:
                start = self.next_backend
                for offset in range(len(self.backends)):
                    index = (start + offset) % len(self.backends)
                    backend = self.backends[index]
                    if backend.semaphore.acquire(blocking=False):
                        backend.in_flight += 1
                        # Advance even on success so sequential requests rotate
                        # across GPUs. Busy replicas are skipped above, so
                        # concurrent bursts spill to the next backend with
                        # available model-serving capacity.
                        self.next_backend = (index + 1) % len(self.backends)
                        return backend
            if time.monotonic() >= deadline:
                return None
            time.sleep(0.05)

    def release_backend(self, backend: Backend) -> None:
        """Release one request slot after the backend response is handled."""

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
            # Model metadata should be the same for every replica because the
            # launcher starts homogeneous vLLM servers. Querying the first
            # backend is enough for readiness checks and avoids consuming a
            # generation slot on every model server.
            self._forward_to_backend(self.state.backends[0], body=None)
            return
        self._send_json(404, {"error": {"message": f"unsupported path: {self.path}"}})

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/"):
            self._send_json(
                404, {"error": {"message": f"unsupported path: {self.path}"}}
            )
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        backend = self.state.acquire_backend()
        if backend is None:
            # OpenHands treats this as an infrastructure failure for the eval
            # request. Returning 503 is more honest than queueing forever: the
            # configured worker count is asking for more concurrent generations
            # than the router can place on vLLM replicas.
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
        """Forward one OpenAI-compatible request to a selected vLLM backend."""

        url = backend.base_url + self.path.removeprefix("/v1")
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS
            and key.lower() not in {"host", "content-length"}
        }
        if self.state.api_key and "Authorization" not in headers:
            # The pod-local vLLM endpoint uses an OpenAI-compatible API shape.
            # The key is a placeholder compatibility value, not a network
            # secret, but forwarding it keeps OpenAI client libraries satisfied.
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
                # This header is for debugging eval infrastructure. It does not
                # change the OpenAI response body that OpenHands parses.
                self.send_header("X-VLLM-Backend", backend.name)
                self.end_headers()
                self.wfile.write(response_body)
        except urllib.error.HTTPError as exc:
            # Preserve backend HTTP errors and bodies. A model-serving error
            # should remain visible to the caller instead of being rewritten as
            # a generic router failure.
            response_body = exc.read()
            self.send_response(exc.code)
            self._copy_response_headers(exc.headers.items())
            self.send_header("X-VLLM-Backend", backend.name)
            self.end_headers()
            self.wfile.write(response_body)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # Network/timeouts here mean the selected vLLM replica was
            # unreachable or stalled. Surface that as a bad gateway so eval logs
            # distinguish backend health from model-generated failures.
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
        """Copy end-to-end response headers while dropping proxy-only headers."""

        for key, value in headers:
            if key.lower() in HOP_BY_HOP_HEADERS:
                continue
            if key.lower() in {"content-length", "content-encoding"}:
                # The router may alter framing by reading and rewriting the
                # body, so let BaseHTTPRequestHandler compute response framing.
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
    """Create router state from CLI args without starting the HTTP server."""

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
    # The pod launcher waits for this process by polling /v1/models. This log
    # line records the serving topology used for the eval run.
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

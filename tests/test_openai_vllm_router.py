"""Tests for the pod-local OpenAI-compatible router used by eval jobs.

The real eval stack has OpenHands workers sending chat-completion requests to
vLLM model servers, usually one vLLM process per GPU. These tests avoid real
model inference by using tiny HTTP backends that mimic the OpenAI response
shape the router must preserve. That keeps the test focused on serving
behavior: which model server receives each request and whether busy servers are
skipped before they become eval bottlenecks.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar

from scripts import openai_vllm_router as router


class FakeBackendHandler(BaseHTTPRequestHandler):
    """A minimal stand-in for one vLLM OpenAI-compatible server.

    vLLM is the inference server that hosts the model weights on a GPU. The
    router should treat vLLM as an opaque OpenAI-compatible HTTP backend: it
    forwards the request and returns the backend's response without interpreting
    generated text. The fake backend therefore only records which replica was
    selected and returns a valid chat-completion envelope.
    """

    backend_name: ClassVar[str] = ""
    sleep_seconds: ClassVar[float] = 0.0
    seen_requests: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send({"data": [{"id": self.backend_name}]})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.sleep_seconds:
            # Slow generation is the normal pressure point for an LLM server:
            # one completion can occupy GPU memory/compute long enough that the
            # router needs to send unrelated OpenHands tasks to another replica.
            time.sleep(self.sleep_seconds)
        self.seen_requests.append(self.backend_name)
        self._send(
            {
                # OpenAI chat-completion responses wrap generated assistant text
                # under choices[0].message.content. OpenHands consumes this
                # shape regardless of whether the server is OpenAI or vLLM.
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": self.backend_name,
                        }
                    }
                ]
            }
        )

    def _send(self, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_: object) -> None:
        return


def start_server(handler_cls: type[BaseHTTPRequestHandler]) -> ThreadingHTTPServer:
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TestOpenAIVLLMRouter:
    def setup_method(self) -> None:
        self.servers: list[ThreadingHTTPServer] = []

    def teardown_method(self) -> None:
        for server in getattr(self, "servers", []):
            server.shutdown()
            server.server_close()

    def _start_backend(
        self, name: str, sleep_seconds: float = 0
    ) -> tuple[ThreadingHTTPServer, type[FakeBackendHandler]]:
        # Each dynamic subclass models a separate GPU-hosted vLLM replica. The
        # per-class request log lets the assertions prove load distribution
        # without depending on real model output or real GPU hardware.
        handler = type(
            f"{name}Handler",
            (FakeBackendHandler,),
            {"backend_name": name, "sleep_seconds": sleep_seconds, "seen_requests": []},
        )
        server = start_server(handler)
        self.servers.append(server)
        return server, handler

    def _start_router(
        self, backends: list[tuple[str, str]], per_backend_concurrency: int = 24
    ) -> ThreadingHTTPServer:
        state = router.RouterState(
            backends=[
                router.Backend(
                    name=name,
                    base_url=url,
                    # This semaphore is the serving budget for one vLLM replica.
                    # In production it protects a GPU from receiving more active
                    # model generations than the eval preset says it can handle.
                    semaphore=threading.BoundedSemaphore(per_backend_concurrency),
                )
                for name, url in backends
            ],
            # The router accepts an OpenAI-compatible API key because standard
            # OpenAI clients expect one, even when the local vLLM server only
            # needs a placeholder token inside the pod.
            api_key="local-llm",
            request_timeout=5,
            acquire_timeout=5,
        )
        handler = type("TestRouterHandler", (router.RouterHandler,), {"state": state})
        server = start_server(handler)
        self.servers.append(server)
        return server

    def _post_chat(self, port: int) -> tuple[dict[str, object], object]:
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b'{"messages":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode()), response.headers

    def test_routes_sequential_requests_round_robin(self) -> None:
        backend0, handler0 = self._start_backend("backend-0")
        backend1, handler1 = self._start_backend("backend-1")
        router_server = self._start_router(
            [
                ("backend-0", f"http://127.0.0.1:{backend0.server_port}/v1"),
                ("backend-1", f"http://127.0.0.1:{backend1.server_port}/v1"),
            ]
        )

        responses = [self._post_chat(router_server.server_port)[0] for _ in range(4)]

        # Round-robin selection keeps a steady stream of OpenHands tasks from
        # piling onto the first GPU-backed model server while later replicas sit
        # idle. The generated assistant text is just the backend name so the test
        # can read the routing decision from the normal OpenAI response body.
        assert [
            response["choices"][0]["message"]["content"] for response in responses
        ] == ["backend-0", "backend-1", "backend-0", "backend-1"]
        assert len(handler0.seen_requests) == 2
        assert len(handler1.seen_requests) == 2

    def test_per_backend_concurrency_spills_to_next_backend(self) -> None:
        backend0, handler0 = self._start_backend("backend-0", sleep_seconds=0.2)
        backend1, handler1 = self._start_backend("backend-1", sleep_seconds=0.2)
        router_server = self._start_router(
            [
                ("backend-0", f"http://127.0.0.1:{backend0.server_port}/v1"),
                ("backend-1", f"http://127.0.0.1:{backend1.server_port}/v1"),
            ],
            per_backend_concurrency=1,
        )

        # With a per-backend concurrency of 1, the first slow request should
        # occupy backend-0's only generation slot. The second request must spill
        # to backend-1 instead of waiting behind backend-0, which is what lets a
        # multi-GPU eval pod use all model replicas during concurrent SWE-bench
        # task execution.
        threads = [
            threading.Thread(target=self._post_chat, args=(router_server.server_port,))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(handler0.seen_requests) == 1
        assert len(handler1.seen_requests) == 1

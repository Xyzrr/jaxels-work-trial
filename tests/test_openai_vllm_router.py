import json
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from scripts import openai_vllm_router as router


class FakeBackendHandler(BaseHTTPRequestHandler):
    backend_name = ""
    sleep_seconds = 0.0
    seen_requests = []

    def do_GET(self):
        if self.path == "/v1/models":
            self._send({"data": [{"id": self.backend_name}]})
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        self.seen_requests.append(self.backend_name)
        self._send(
            {
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

    def _send(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        return


def start_server(handler_cls):
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler_cls)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


class TestOpenAIVLLMRouter:
    def teardown_method(self):
        for server in getattr(self, "servers", []):
            server.shutdown()
            server.server_close()

    def _start_backend(self, name, sleep_seconds=0):
        handler = type(
            f"{name}Handler",
            (FakeBackendHandler,),
            {"backend_name": name, "sleep_seconds": sleep_seconds, "seen_requests": []},
        )
        server = start_server(handler)
        self.servers.append(server)
        return server, handler

    def _start_router(self, backends, per_backend_concurrency=24):
        state = router.RouterState(
            backends=[
                router.Backend(
                    name=name,
                    base_url=url,
                    semaphore=threading.BoundedSemaphore(per_backend_concurrency),
                )
                for name, url in backends
            ],
            api_key="local-llm",
            request_timeout=5,
            acquire_timeout=5,
        )
        handler = type("TestRouterHandler", (router.RouterHandler,), {"state": state})
        server = start_server(handler)
        self.servers.append(server)
        return server

    def _post_chat(self, port):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions",
            data=b'{"messages":[]}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode()), response.headers

    def test_routes_sequential_requests_round_robin(self):
        self.servers = []
        backend0, handler0 = self._start_backend("backend-0")
        backend1, handler1 = self._start_backend("backend-1")
        router_server = self._start_router(
            [
                ("backend-0", f"http://127.0.0.1:{backend0.server_port}/v1"),
                ("backend-1", f"http://127.0.0.1:{backend1.server_port}/v1"),
            ]
        )

        responses = [self._post_chat(router_server.server_port)[0] for _ in range(4)]

        assert [
            response["choices"][0]["message"]["content"] for response in responses
        ] == ["backend-0", "backend-1", "backend-0", "backend-1"]
        assert len(handler0.seen_requests) == 2
        assert len(handler1.seen_requests) == 2

    def test_per_backend_concurrency_spills_to_next_backend(self):
        self.servers = []
        backend0, handler0 = self._start_backend("backend-0", sleep_seconds=0.2)
        backend1, handler1 = self._start_backend("backend-1", sleep_seconds=0.2)
        router_server = self._start_router(
            [
                ("backend-0", f"http://127.0.0.1:{backend0.server_port}/v1"),
                ("backend-1", f"http://127.0.0.1:{backend1.server_port}/v1"),
            ],
            per_backend_concurrency=1,
        )

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

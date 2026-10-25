import http.server
import threading
import time

from kenoma.engine import RemoteEngine


class _KeepaliveOnly(http.server.BaseHTTPRequestHandler):
    """An SSE server that never sends a completion, only comments."""

    def do_POST(self) -> None:
        self.rfile.read(int(self.headers["Content-Length"]))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        try:
            while True:
                self.wfile.write(b": keepalive\n\n")
                self.wfile.flush()
                time.sleep(0.05)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, format: str, *args: object) -> None:
        pass


def test_interactive_stream_hands_back_control_and_ends_by_its_deadline():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _KeepaliveOnly)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        e = RemoteEngine(f"http://127.0.0.1:{srv.server_port}", "m", 1.0, 0.95)
        e.INTERACTIVE_DEADLINE = 0.5
        start = last = time.monotonic()
        gaps = []
        for text, _ in e.stream([("x", None)], 10, interactive=True):
            assert text == ""
            now = time.monotonic()
            gaps.append(now - last)
            last = now
        assert time.monotonic() - start < 1.5
        assert gaps and max(gaps) < 0.3
    finally:
        srv.shutdown()

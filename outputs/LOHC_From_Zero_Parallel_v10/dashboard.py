"""LLM-UCB dashboard with a read-only A baseline and live C monitoring."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import secrets
from urllib.parse import urlsplit

HERE = Path(__file__).resolve().parent
ARMS = ("native", "llm")


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


class LiveData:
    def __init__(self, run, baseline=None):
        self.directories = {"native": baseline if baseline is not None else run / "native",
                            "llm": run / "llm"}

    def status(self, arm):
        directory = self.directories[arm]
        status = read_json(directory / "status.json", {}) or {}
        if arm == "llm":
            status["stop_requested"] = (directory / "STOP").exists()
        return status

    def missing_baseline(self):
        directory = self.directories["native"]
        required = ("initial_population.csv.gz", "configuration.json", "progress_history.json")
        return not all((directory / name).is_file() for name in required) or not any(directory.glob("graph_evaluations_*.json"))

    def stop(self):
        directory = self.directories["llm"]
        state = self.status("llm")
        if not state.get("pid") or state.get("phase") in {"已完成", "已停止", "已立即停止", "运行失败"}:
            raise ValueError("C 组当前没有可停止的实验")
        (directory / "STOP").touch(exist_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory containing C results under llm/")
    parser.add_argument("--baseline", type=Path, help="Local A results directory, such as an existing run's native/ folder")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    data = LiveData(args.run.resolve(), args.baseline.resolve() if args.baseline else None)
    from rule_dashboard_data import RuleDashboard

    plotly = importlib.util.find_spec("plotly")
    if plotly is None:
        parser.error("Install the dependencies from requirements.txt.")
    javascript = Path(plotly.origin).parent / "package_data" / "plotly.min.js"
    token = secrets.token_urlsafe(24)
    page = (HERE / "rule_dashboard.html").read_text(encoding="utf-8").replace("__CONTROL_TOKEN__", token).encode("utf-8")
    dashboard = RuleDashboard(data)

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value, kind):
            self.send_response(code)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(value)))
            self.send_header("Cache-Control", "no-store" if kind != "application/javascript" else "private, max-age=3600")
            self.end_headers()
            self.wfile.write(value)

        def json(self, code, value):
            self.respond(code, json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8")

        def do_GET(self):
            request = urlsplit(self.path)
            path = request.path
            if path == "/":
                self.respond(200, page, "text/html; charset=utf-8")
            elif path == "/api/rule/status":
                self.json(200, dashboard.status())
            elif path == "/api/rule":
                self.json(200, dashboard.snapshot())
            elif path == "/plotly.js":
                self.respond(200, javascript.read_bytes(), "application/javascript")
            else:
                self.send_error(404)

        def do_POST(self):
            if self.headers.get("X-Control-Token") != token:
                self.json(403, {"error": "页面控制令牌无效，请刷新。"})
                return
            if urlsplit(self.path).path != "/api/stop/llm":
                self.send_error(404)
                return
            try:
                data.stop()
                self.json(200, {"ok": True})
            except (OSError, ValueError) as exc:
                self.json(409, {"error": str(exc)})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

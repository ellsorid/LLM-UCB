"""Local, read-only plotting with explicit per-arm stop controls."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import secrets
import subprocess
from threading import Lock
from urllib.parse import parse_qs, urlsplit

HERE = Path(__file__).resolve().parent
ARMS = ("native", "llm")
NX, NY = 15, 19
H_RANGE, D_RANGE = (0.5, 8.0), (15.0, 110.0)


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return default


def make_surface(path):
    bins = defaultdict(Counter)
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                h2, dh, sa = (float(row[key]) for key in ("h2_wt_percent", "delta_H_per_H2", "SA score"))
            except (ValueError, TypeError, KeyError):
                continue
            scaffold = row.get("scaffold_smi")
            if not scaffold or not (math.isfinite(sa) and sa <= 4 and
                                    H_RANGE[0] <= h2 <= H_RANGE[1] and D_RANGE[0] <= dh <= D_RANGE[1]):
                continue
            ix = min(NX - 1, int((h2 - H_RANGE[0]) / ((H_RANGE[1] - H_RANGE[0]) / NX)))
            iy = min(NY - 1, int((dh - D_RANGE[0]) / ((D_RANGE[1] - D_RANGE[0]) / NY)))
            bins[ix, iy][scaffold] += 1
    points = []
    for (ix, iy), count in sorted(bins.items()):
        n = sum(count.values())
        effective = math.exp(-sum((value / n) * math.log(value / n) for value in count.values()))
        points.append([ix, iy, round(effective, 5), n, len(count)])
    return points


def recent_log(path):
    try:
        with path.open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 4096))
            return stream.read().decode("utf-8", errors="replace").splitlines()[-9:]
    except OSError:
        return []


class LiveData:
    def __init__(self, run):
        self.run = run
        self.cache = {arm: {} for arm in ARMS}
        self.lock = Lock()

    def status(self, arm):
        directory = self.run / arm
        status = read_json(directory / "status.json", {}) or {}
        if (directory / "hard_stop.json").exists():
            status["phase"] = "已立即停止"
        status["stop_requested"] = (directory / "STOP").exists()
        return status

    def surface(self, arm, generation):
        directory = self.run / arm
        known = self.cache[arm]
        if generation in known:
            return
        snapshot = directory / ("latest_population.csv.gz" if generation else "initial_population.csv.gz")
        try:
            if generation and snapshot.stat().st_mtime_ns > (directory / "progress_history.json").stat().st_mtime_ns:
                return
            known[generation] = make_surface(snapshot)
        except (OSError, EOFError, ValueError):
            return
        while len(known) > 36:
            del known[min(known)]

    def snapshot(self):
        with self.lock:
            arms = {}
            for arm in ARMS:
                directory = self.run / arm
                status = self.status(arm)
                if status:
                    self.surface(arm, status.get("completed_generation", 0))
                surfaces = self.cache[arm]
                last = max(surfaces, default=None)
                arms[arm] = {
                    "status": status, "history": read_json(directory / "progress_history.json", []),
                    "surface_generation": last, "surface": surfaces.get(last, []),
                    "log": recent_log(directory / "run.log") or recent_log(self.run / f"{arm}.log"),
                    "priority": read_json(directory / "candidate_review_summary.json"),
                }
            common = set(self.cache["native"]) & set(self.cache["llm"])
            generation = max(common, default=None)
            delta = []
            if generation is not None:
                base = {(p[0], p[1]): p for p in self.cache["native"][generation]}
                for x, y, z, n, scaffolds in self.cache["llm"][generation]:
                    match = base.get((x, y))
                    if match:
                        delta.append([x, y, round(z - match[2], 5), n, match[3]])
            return {"arms": arms, "delta": delta, "delta_generation": generation,
                    "grid": {"nx": NX, "ny": NY, "h": H_RANGE, "d": D_RANGE},
                    "extended": read_json(self.run / "extended_evaluation.json"),
                    "spatial_run": (self.run / "independent_review").exists(),
                    "paired_run": (self.run / "paired_experiment.json").exists(),
                    "ac_run": (self.run / "ac_experiment.json").exists(),
                    "operator_ac": bool((read_json(self.run / "ac_experiment.json", {}) or {}).get("operator_llm")),
                    "ac_evaluation": read_json(self.run / "ac_evaluation.json"),
                    "spatial": read_json(self.run / "spatial_evaluation.json"),
                    "bayesian": read_json(self.run / "bayesian_evaluation.json")}

    def stop(self, arm, immediate):
        directory = self.run / arm
        state = self.status(arm)
        if state.get("phase") in {"已完成", "已停止", "已立即停止", "运行失败"}:
            return
        if not immediate:
            (directory / "STOP").touch(exist_ok=True)
            return
        pid = int(state.get("pid", 0))
        if pid <= 0:
            raise ValueError("工作进程尚未就绪")
        command = (
            f"$p=Get-CimInstance Win32_Process -Filter 'ProcessId={pid}'; "
            "if($p){[Console]::Write($p.CommandLine)}"
        )
        process = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command],
                                 capture_output=True, timeout=10)
        line = process.stdout.decode("utf-8", errors="replace").casefold()
        if "worker.py" not in line or f"--arm {arm}" not in line or str(directory).casefold() not in line:
            raise ValueError("无法核对实验进程身份")
        result = subprocess.run(["taskkill.exe", "/PID", str(pid), "/T", "/F"],
                                capture_output=True, timeout=10)
        if result.returncode:
            raise ValueError("未能停止该实验进程")
        (directory / "hard_stop.json").write_text(json.dumps({"pid": pid}), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=HERE / "live")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--performance", action="store_true", help="Open the performance dashboard as the home page")
    parser.add_argument("--rule-performance", action="store_true", help="Open the graph-rule comparison dashboard")
    parser.add_argument("--absolute-replay", type=Path, help="Show fixed-scale blind-pilot results in the rule dashboard")
    args = parser.parse_args()
    if args.absolute_replay and not args.rule_performance:
        parser.error('--absolute-replay requires --rule-performance')
    data = LiveData(args.run.resolve())
    projection = None
    if args.performance:
        from structural_projection import StructuralProjection
        projection = StructuralProjection(data.run)
    plotly = importlib.util.find_spec("plotly")
    javascript = Path(plotly.origin).parent / "package_data" / "plotly.min.js"
    token = secrets.token_urlsafe(24)
    page = (HERE / "dashboard.html").read_text(encoding="utf-8").replace("__CONTROL_TOKEN__", token).encode("utf-8")
    performance = (HERE / "performance.html").read_bytes()
    rule_dashboard = rule_page = None
    if args.rule_performance:
        from rule_dashboard_data import RuleDashboard
        replay = read_json(args.absolute_replay) if args.absolute_replay else None
        if args.absolute_replay and (not replay or Path(replay['source']).resolve() != data.run):
            parser.error('The replay result must belong to the specified run')
        rule_dashboard = RuleDashboard(data, replay=replay)
        rule_page = (HERE / "rule_dashboard.html").read_text(encoding="utf-8").replace("__CONTROL_TOKEN__", token).encode("utf-8")

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
                self.respond(200, rule_page or (performance if args.performance else page), "text/html; charset=utf-8")
            elif path == "/performance":
                self.respond(200, performance, "text/html; charset=utf-8")
            elif path == "/structure":
                self.respond(200, page, "text/html; charset=utf-8")
            elif path == "/api/state":
                self.json(200, data.snapshot())
            elif path == "/api/rule/status" and rule_dashboard is not None:
                self.json(200, {"status": {arm: data.status(arm) for arm in ARMS},
                                "full_m3os": rule_dashboard.full_m3os,
                                "graph_mode": rule_dashboard.graph_mode})
            elif path == "/api/performance" and projection is not None:
                self.json(200, projection.snapshot())
            elif path == "/api/rule" and rule_dashboard is not None:
                stage = parse_qs(request.query).get('generation', [''])[0]
                if stage and not (stage.isascii() and stage.isdecimal() and len(stage) <= 4):
                    self.send_error(400)
                else:
                    self.json(200, rule_dashboard.snapshot(int(stage) if stage else None))
            elif path == "/plotly.js":
                self.respond(200, javascript.read_bytes(), "application/javascript")
            else:
                self.send_error(404)

        def do_POST(self):
            path = urlsplit(self.path).path.split("/")
            if self.headers.get("X-Control-Token") != token:
                self.json(403, {"error": "页面控制令牌无效，请刷新。"})
                return
            if len(path) != 4 or path[1] != "api" or path[2] not in {"stop", "terminate"} or path[3] not in ARMS:
                self.send_error(404)
                return
            try:
                data.stop(path[3], path[2] == "terminate")
                self.json(200, {"ok": True})
            except (OSError, ValueError, subprocess.SubprocessError) as exc:
                self.json(409, {"error": str(exc)})

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Dashboard http://127.0.0.1:{server.server_port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

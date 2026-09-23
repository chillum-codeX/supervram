#!/usr/bin/env python3
"""SuperVRAM chat: a local orchestrator that lists downloaded GGUF models, starts/stops a single
llama-server instance for whichever one is selected (only one large model fits in 24 GB VRAM at a
time), and serves the chat UI. The UI itself talks directly to llama-server's own OpenAI-compatible
API (streaming) on a fixed port -- this process only manages "which model is loaded" and "how".

No third-party dependencies. Run: python3 chat/server.py [--port 8788]
"""
import http.server
import json
import os
import re
import socketserver
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path

import agent_tools

ROOT = Path(__file__).resolve().parent
SUPERVRAM_ROOT = ROOT.parent
LLAMA_SERVER = SUPERVRAM_ROOT / "third_party/llama.cpp/build-3090/bin/llama-server"
MODELS_DIR = Path(os.environ.get("SUPERVRAM_MODELS", str(Path.home() / "models")))
HISTORY_DIR = ROOT / "history"
CHAT_PORT = 8090
# Single-user UI -> one slot, so the whole budget goes to it instead of being split across
# llama-server's default 4 parallel slots (that split silently capped any one request at 8192
# tokens even though --ctx-size looked like plenty -- the coding agent's tool-call histories,
# which include whole file contents, blow past that easily).
N_PARALLEL = 1
CTX_SIZE_DEFAULT = 32768
# Qwen3.6-35B-A3B is a hybrid linear-attention architecture -- only 1-in-4 layers are full
# attention, the rest are cheap recurrent/SSM state, so its KV cache costs ~20KB/token versus
# ~200-260KB/token for a dense model. That makes a much larger context window affordable in the
# same VRAM budget, which is exactly what long coding-agent sessions need (full file contents in
# every tool result add up fast). 131072 leaves headroom under its native 262144 ceiling.
CTX_SIZE_OVERRIDES = {
    "Qwen3.6-35B-A3B-Q4_K_M.gguf": 131072,
}
LOG_PATH = "/tmp/claude-1000/chat-llama-server.log"
APP_LOG_PATH = ROOT / "logs" / "app.log"


def ctx_size_for(model_id: str) -> int:
    return CTX_SIZE_OVERRIDES.get(model_id, CTX_SIZE_DEFAULT)

# MoE fallback ladders: each entry is extra CLI args tried in order until one boots without an
# early crash (out-of-memory shows up as the process exiting within a few seconds).
MOE_FAST_LADDER = [
    ["-ngl", "999"],
    ["-ngl", "999", "--n-cpu-moe", "4"],
    ["-ngl", "999", "--n-cpu-moe", "8"],
    ["-ngl", "999", "--n-cpu-moe", "14"],
    ["-ngl", "999", "--n-cpu-moe", "20"],
    ["-ngl", "999", "--n-cpu-moe", "30"],
]
MOE_TIERED_LADDER = [
    ["-ngl", "999", "--moe-expert-storage", "cache", "--moe-expert-cache-size", "14336", "--moe-expert-direct-io"],
    ["-ngl", "999", "--moe-expert-storage", "cache", "--moe-expert-cache-size", "8192", "--moe-expert-direct-io"],
]
DENSE_LADDER = [
    ["-ngl", "999"],
    ["-ngl", "60"],
    ["-ngl", "45"],
    ["-ngl", "32"],
    ["-ngl", "20"],
    ["-ngl", "10"],
]

state_lock = threading.Lock()
gen_lock = threading.Lock()
launch_lock = threading.Lock()
current_gen = 0


def bump_gen():
    global current_gen
    with gen_lock:
        current_gen += 1
        return current_gen


def is_current(g):
    with gen_lock:
        return g == current_gen


state = {
    "loading": False,
    "ready": False,
    "error": None,
    "model_id": None,
    "model_name": None,
    "mode": None,
    "attempt_note": None,
    "started_at": None,
    "ctx_size": None,
}
proc_holder = {"proc": None}


log_write_lock = threading.Lock()


def log_app_event(kind: str, detail: dict):
    """Append one JSON-line entry to the persistent app log (client-reported chat errors, tool
    failures, etc). Separate from LOG_PATH, which is llama-server's own raw stdout/stderr --
    this one is for things the browser side sees that the model process never logs, like a
    fetch() failing or the server responding non-200."""
    entry = {"ts": time.time(), "kind": kind, **detail}
    APP_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with log_write_lock:
        with open(APP_LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")


def browse_dir(raw_path: str):
    """Directory listing for the in-app folder browser (used to pick a coding-agent workspace).
    Not sandboxed like the agent tools -- this only lists directory names to click through, it
    never reads file contents, and the workspace the agent actually operates in is whatever
    folder the person picks from here."""
    p = Path(raw_path or str(Path.home())).expanduser()
    if not p.exists():
        p = Path.home()
    if not p.is_dir():
        p = p.parent
    p = p.resolve()
    dirs = []
    try:
        for c in sorted(p.iterdir(), key=lambda x: x.name.lower()):
            if c.is_dir() and not c.name.startswith("."):
                dirs.append(c.name)
    except PermissionError:
        pass
    return {"path": str(p), "parent": str(p.parent) if p.parent != p else None, "dirs": dirs}


def is_moe(name: str) -> bool:
    return "a3b" in name.lower() or "-a3b-" in name.lower()


def list_models():
    out = []
    if not MODELS_DIR.exists():
        return out
    for f in sorted(MODELS_DIR.glob("*.gguf")):
        size_gib = f.stat().st_size / (1024 ** 3)
        moe = is_moe(f.name)
        out.append({
            "id": f.name,
            "name": f.stem,
            "path": str(f),
            "size_gib": round(size_gib, 1),
            "kind": "moe" if moe else "dense",
        })
    return out


def stop_current():
    p = proc_holder["proc"]
    if p is not None and p.poll() is None:
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
    proc_holder["proc"] = None


def health_ok():
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{CHAT_PORT}/health", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def try_launch(model_path, ctx_size, extra_args, log_f, my_gen):
    """Returns 'ready', 'crashed', 'timeout', or 'superseded' (a newer /api/select arrived --
    this attempt must stop touching shared state/the process immediately)."""
    cmd = [str(LLAMA_SERVER), "--model", model_path, "--host", "127.0.0.1", "--port", str(CHAT_PORT),
           "--ctx-size", str(ctx_size), "--parallel", str(N_PARALLEL), "--no-webui",
           "--reasoning-format", "deepseek", "--threads", "12"] + extra_args
    log_f.write(f"\n\n=== launching: {' '.join(cmd)} ===\n")
    log_f.flush()
    p = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
    proc_holder["proc"] = p
    deadline = time.time() + 90
    while time.time() < deadline:
        if not is_current(my_gen):
            return "superseded"
        if p.poll() is not None:
            return "crashed"  # exited early -- treat as a failed attempt (usually OOM)
        if health_ok():
            return "ready"
        time.sleep(1)
    try:
        p.terminate()
    except Exception:
        pass
    return "timeout"


def load_model_bg(model_id, mode, my_gen):
    # serialize concurrent /select calls; a stale (superseded) request gives up the lock quickly
    # because try_launch polls is_current() every second rather than running its full timeout.
    with launch_lock:
        if not is_current(my_gen):
            return
        stop_current()
        with state_lock:
            state.update(loading=True, ready=False, error=None, model_id=model_id, mode=mode,
                          attempt_note="starting...", started_at=time.time())
        models = {m["id"]: m for m in list_models()}
        m = models.get(model_id)
        if m is None:
            if is_current(my_gen):
                with state_lock:
                    state.update(loading=False, error=f"model not found: {model_id}")
            return
        moe = m["kind"] == "moe"
        if moe and mode == "tiered":
            ladder = MOE_TIERED_LADDER
        elif moe:
            ladder = MOE_FAST_LADDER
        else:
            ladder = DENSE_LADDER
        ctx_size = ctx_size_for(model_id)
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        result = "crashed"
        with open(LOG_PATH, "a") as log_f:
            for i, extra in enumerate(ladder):
                if not is_current(my_gen):
                    stop_current()
                    return
                with state_lock:
                    state["attempt_note"] = f"trying {' '.join(extra)} ({i+1}/{len(ladder)})"
                result = try_launch(m["path"], ctx_size, extra, log_f, my_gen)
                if result in ("ready", "superseded"):
                    break
                stop_current()
        if result == "superseded" or not is_current(my_gen):
            stop_current()
            return
        with state_lock:
            if result == "ready":
                state.update(loading=False, ready=True, error=None, model_name=m["name"],
                             ctx_size=ctx_size, attempt_note=None)
            else:
                state.update(loading=False, ready=False,
                             error="every fallback configuration failed to start -- see "
                                   f"{LOG_PATH} (likely out of VRAM even at the smallest offload tried)")


HISTORY_ID_RE = re.compile(r"^[a-zA-Z0-9_-]+$")


def history_list():
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for f in HISTORY_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        out.append({
            "id": d.get("id", f.stem),
            "title": d.get("title", "Untitled"),
            "model_name": d.get("model_name"),
            "updated_at": d.get("updated_at", 0),
            "message_count": len(d.get("messages", [])),
        })
    out.sort(key=lambda x: x["updated_at"], reverse=True)
    return out


def history_path(conv_id):
    if not HISTORY_ID_RE.match(conv_id):
        return None
    return HISTORY_DIR / f"{conv_id}.json"


def history_read(conv_id):
    p = history_path(conv_id)
    if p is None or not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def history_write(body):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    conv_id = body.get("id")
    if not conv_id or not HISTORY_ID_RE.match(conv_id):
        conv_id = uuid.uuid4().hex[:12]
    messages = body.get("messages", [])
    title = body.get("title") or "Untitled"
    title = title.strip()[:60] or "Untitled"
    data = {
        "id": conv_id,
        "title": title,
        "model_name": body.get("model_name"),
        "updated_at": time.time(),
        "messages": messages,
    }
    history_path(conv_id).write_text(json.dumps(data, indent=1))
    return conv_id


def history_delete(conv_id):
    p = history_path(conv_id)
    if p is not None and p.exists():
        p.unlink()
        return True
    return False


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT), **kw)

    def log_message(self, fmt, *args):
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urllib.parse.urlparse(self.path).path
        if path == "/api/models":
            long_ctx_id = next(iter(CTX_SIZE_OVERRIDES), None)
            return self._json({"models": list_models(), "chat_port": CHAT_PORT,
                                "long_context_model_id": long_ctx_id,
                                "long_context_size": CTX_SIZE_OVERRIDES.get(long_ctx_id)})
        if path == "/api/status":
            with state_lock:
                s = dict(state)
            s["elapsed_s"] = round(time.time() - s["started_at"], 1) if s.get("started_at") and s["loading"] else None
            return self._json(s)
        if path == "/api/history":
            return self._json({"conversations": history_list()})
        if path.startswith("/api/history/"):
            conv = history_read(path[len("/api/history/"):])
            if conv is None:
                return self._json({"error": "not found"}, 404)
            return self._json(conv)
        if path == "/api/browse":
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
            return self._json(browse_dir(qs.get("path", [str(Path.home())])[0]))
        if path == "/api/tool_schemas":
            return self._json({"tools": agent_tools.TOOL_SCHEMAS})
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path == "/api/select":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            model_id = body.get("model_id")
            mode = body.get("mode", "fast")
            if not model_id:
                return self._json({"error": "model_id required"}, 400)
            my_gen = bump_gen()
            threading.Thread(target=load_model_bg, args=(model_id, mode, my_gen), daemon=True).start()
            return self._json({"ok": True})
        if self.path == "/api/history":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            conv_id = history_write(body)
            return self._json({"id": conv_id})
        if self.path == "/api/workspace/select":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            p = Path(body.get("path", "")).expanduser()
            if not p.is_dir():
                return self._json({"error": "not a directory"}, 400)
            return self._json({"ok": True, "path": str(p.resolve())})
        if self.path == "/api/tool/execute":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            root = body.get("workspace_root")
            name = body.get("name")
            args = body.get("arguments") or {}
            if not root or name not in agent_tools.DISPATCH:
                return self._json({"error": f"unknown tool '{name}'"}, 400)
            try:
                result = agent_tools.DISPATCH[name](root, args)
            except Exception as e:
                result = {"error": f"tool '{name}' failed: {e}"}
            if result.get("error"):
                log_app_event("tool_error", {"name": name, "args": args, "error": result["error"]})
            return self._json(result)
        if self.path == "/api/tool/apply_write":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            result = agent_tools.tool_apply_write(body.get("workspace_root", ""), body.get("path", ""), body.get("content", ""))
            if result.get("error"):
                log_app_event("apply_write_error", {"path": body.get("path", ""), "error": result["error"]})
            return self._json(result)
        if self.path == "/api/log":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or b"{}")
            log_app_event(body.get("kind", "client"), {k: v for k, v in body.items() if k != "kind"})
            return self._json({"ok": True})
        self.send_response(404)
        self.end_headers()

    def do_DELETE(self):
        path = urllib.parse.urlparse(self.path).path
        if path.startswith("/api/history/"):
            ok = history_delete(path[len("/api/history/"):])
            return self._json({"ok": ok})
        self.send_response(404)
        self.end_headers()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.ThreadingTCPServer(("0.0.0.0", args.port), Handler) as httpd:
        print(f"SuperVRAM chat orchestrator on http://localhost:{args.port}  (models dir: {MODELS_DIR})")
        try:
            httpd.serve_forever()
        finally:
            stop_current()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ClearML worker for a Kaggle GPU session; started by kaggle/npuflow-agent.ipynb.

Serves the ClearML queue until it has been empty for a while or the session
time is nearly over. Expects CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY in
the environment (the notebook takes them from Kaggle Secrets).

Kaggle keeps no log of a session that is cancelled, and a session now and then
hangs before the worker registers. So every step is also written, with its time,
to a ClearML task of its own (project NPUFlow/agent, plain REST, nothing to
install first), external commands have timeouts, and the session ends itself if
no worker has registered within REGISTER_MINUTES of its start.
"""
import datetime
import json
import os
import socket
import subprocess
import sys
import threading
import time

QUEUE = os.environ.get("NPUFLOW_QUEUE", "kaggle")
MAX_HOURS = float(os.environ.get("NPUFLOW_MAX_HOURS", 11.3))     # leave before Kaggle's 12-hour limit
IDLE_MINUTES = float(os.environ.get("NPUFLOW_IDLE_MINUTES", 10))  # leave if idle this long
REGISTER_MINUTES = float(os.environ.get("NPUFLOW_REGISTER_MINUTES", 20))  # normally 7-12 min
START = float(os.environ.get("NPUFLOW_SESSION_START", time.time()))
INPUT = "/kaggle/input"
LOG_PROJECT = "NPUFlow/agent"


class RemoteLog:
    """Session log in a ClearML task of its own, through the REST API with `requests`
    (preinstalled on Kaggle). Best effort: a failure here never stops the agent."""

    def __init__(self):
        self.api = os.environ["CLEARML_API_HOST"].rstrip("/")
        self.task = None
        self.headers = None
        self.lock = threading.Lock()
        try:
            import requests
            self.http = requests.Session()
            r = self.http.post(f"{self.api}/auth.login", timeout=20,
                               auth=(os.environ["CLEARML_API_ACCESS_KEY"], os.environ["CLEARML_API_SECRET_KEY"]))
            self.headers = {"Authorization": "Bearer " + r.json()["data"]["token"]}
            found = self.call("projects.get_all", name=f"^{LOG_PROJECT}$")["projects"]
            project = found[0]["id"] if found else self.call("projects.create", name=LOG_PROJECT,
                                                               description="Kaggle agent sessions")["id"]
            name = "session " + datetime.datetime.fromtimestamp(START, datetime.timezone.utc).strftime("%Y-%m-%d %H:%M")
            self.task = self.call("tasks.create", name=name, project=project, type="application")["id"]
            self.call("tasks.started", task=self.task, force=True)
        except Exception as e:
            print("remote log unavailable:", e, flush=True)
            self.task = None

    def call(self, endpoint, **payload):
        r = self.http.post(f"{self.api}/{endpoint}", json=payload, headers=self.headers, timeout=20)
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"{endpoint}: {body.get('meta', {}).get('result_msg')}")
        return body["data"]

    def write(self, text):
        if self.task is None:
            return
        event = {"type": "log", "task": self.task, "timestamp": int(time.time() * 1000), "level": "info",
                 "worker": socket.gethostname(), "msg": text + "\n"}
        try:
            with self.lock:
                self.http.post(f"{self.api}/events.add_batch", data=json.dumps(event), headers=self.headers,
                               timeout=20)
        except Exception:
            pass

    def close(self, ok):
        if self.task is None:
            return
        try:
            if ok:
                self.call("tasks.completed", task=self.task, force=True)
            else:
                self.call("tasks.failed", task=self.task, force=True, status_reason="see log")
        except Exception:
            pass


REMOTE = None


def log(*parts):
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%H:%M:%S")
    text = f"[{stamp} +{(time.time() - START) / 60:5.1f} min] " + " ".join(str(p) for p in parts)
    print(text, flush=True)
    if REMOTE is not None:
        REMOTE.write(text)


def run(cmd, timeout, attempts=1, **kw):
    """subprocess.run with a timeout and retries; logs what it does."""
    for attempt in range(1, attempts + 1):
        log(f"run ({attempt}/{attempts}, timeout {timeout} s):", " ".join(cmd))
        try:
            proc = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, **kw)
            if proc.returncode == 0:
                return proc
            log(f"exit code {proc.returncode}:", (proc.stdout + proc.stderr)[-800:])
        except subprocess.TimeoutExpired:
            log("timed out")
    raise RuntimeError(f"failed: {' '.join(cmd)}")


registered = threading.Event()


def watchdog():
    """Ends the session if no worker has registered in time: a hung session would burn the GPU quota."""
    if not registered.wait(timeout=max(0.0, START + REGISTER_MINUTES * 60 - time.time())):
        log(f"no worker registered within {REGISTER_MINUTES:.0f} min of the session start, ending the session")
        if REMOTE is not None:
            REMOTE.close(ok=False)
        os._exit(3)


def main():
    global REMOTE
    os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    threading.Thread(target=watchdog, daemon=True).start()
    REMOTE = RemoteLog()
    log("agent.py started on", socket.gethostname(), "| remote log task:", REMOTE.task)
    run([sys.executable, "-m", "pip", "install", "-q", "clearml", "clearml-agent"], timeout=600, attempts=2)
    log("pip install done")
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], timeout=60).stdout.strip()
    log("GPU:", gpu.replace("\n", "; ") or "none", "| CPU cores:", os.cpu_count())

    # dataset locations are resolved by data.py itself (it searches /kaggle/input)
    log("inputs:", sorted(os.listdir(INPUT)) if os.path.isdir(INPUT) else "none")

    from clearml.backend_api import Session
    session = Session()
    host = socket.gethostname()

    def call(service, action, **payload):
        return session.send_request(service=service, action=action, json=payload).json()["data"]

    queue_id = next(q["id"] for q in call("queues", "get_all", name=QUEUE)["queues"] if q["name"] == QUEUE)
    log("ClearML session ok")

    def queue_length():
        return len(call("queues", "get_by_id", queue=queue_id)["queue"].get("entries", []))

    def busy():
        return any(w["id"].startswith(host) and w.get("task") for w in call("workers", "get_all")["workers"])

    env = dict(os.environ)
    env["CLEARML_AGENT_SKIP_PIP_VENV_INSTALL"] = sys.executable   # use Kaggle's Python with its torch
    env["CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL"] = "1"
    env["NPUFLOW_DEADLINE"] = str(START + (MAX_HOURS - 0.25) * 3600)  # train.py saves and stops by then

    def worker_up():
        return any(w["id"].startswith(host) for w in call("workers", "get_all")["workers"])

    log(f"queue '{QUEUE}': {queue_length()} task(s) waiting; starting clearml-agent")
    agent = subprocess.Popen(["clearml-agent", "daemon", "--queue", QUEUE, "--foreground"], env=env)
    idle_since = time.time()
    last_note = 0.0
    try:
        while agent.poll() is None:
            time.sleep(60 if registered.is_set() else 15)
            hours = (time.time() - START) / 3600
            if hours >= MAX_HOURS:
                log(f"{hours:.1f} h: session time is over, stopping the agent")
                break
            try:
                if not registered.is_set() and worker_up():
                    registered.set()
                    log("worker registered")
                active = busy() or queue_length() > 0
            except Exception as e:      # a network hiccup must not kill a running training
                log("status check failed:", e)
                active = True
            if time.time() - last_note > 1800:   # a sign of life in the remote log every 30 min
                log(f"alive, busy={active}")
                last_note = time.time()
            if active:
                idle_since = time.time()
            elif time.time() - idle_since > IDLE_MINUTES * 60:
                log(f"nothing to do for {IDLE_MINUTES:.0f} min, stopping the agent")
                break
    finally:
        if agent.poll() is None:
            agent.terminate()
            try:
                agent.wait(timeout=120)
            except subprocess.TimeoutExpired:
                agent.kill()
    log(f"agent finished after {(time.time() - START) / 3600:.2f} h")


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        log("agent failed:", repr(e))
        if REMOTE is not None:
            REMOTE.close(ok=False)
        raise
    if REMOTE is not None:
        REMOTE.close(ok=True)

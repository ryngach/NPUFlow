#!/usr/bin/env python3
"""ClearML worker for a Kaggle GPU session; started by kaggle/npuflow-agent.ipynb.

Serves the ClearML queue until it has been empty for a while or the session
time is nearly over. Expects CLEARML_API_ACCESS_KEY / CLEARML_API_SECRET_KEY in
the environment (the notebook takes them from Kaggle Secrets).
"""
import os
import socket
import subprocess
import sys
import time

QUEUE = os.environ.get("NPUFLOW_QUEUE", "kaggle")
MAX_HOURS = float(os.environ.get("NPUFLOW_MAX_HOURS", 11.3))     # leave before Kaggle's 12-hour limit
IDLE_MINUTES = float(os.environ.get("NPUFLOW_IDLE_MINUTES", 10))  # leave if idle this long
START = float(os.environ.get("NPUFLOW_SESSION_START", time.time()))
INPUT = "/kaggle/input"


def find_root(relative_marker):
    """Dataset root: the directory under /kaggle/input (up to two levels deep) that contains the marker.

    A recursive glob over the mounted datasets takes many minutes, so only a few fixed depths are tried.
    """
    if not os.path.isdir(INPUT):
        return None
    level1 = [os.path.join(INPUT, d) for d in sorted(os.listdir(INPUT))]
    level2 = [os.path.join(a, d) for a in level1 if os.path.isdir(a) for d in sorted(os.listdir(a))]
    for base in level1 + level2:
        if os.path.exists(os.path.join(base, relative_marker)):
            return base
    return None


def main():
    os.environ.setdefault("CLEARML_API_HOST", "https://api.clear.ml")
    os.environ.setdefault("CLEARML_WEB_HOST", "https://app.clear.ml")
    os.environ.setdefault("CLEARML_FILES_HOST", "https://files.clear.ml")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "clearml", "clearml-agent"], check=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() or "no GPU", "| CPU cores:", os.cpu_count())

    roots = {"NPUFLOW_CHAIRS": find_root("00001_img1.ppm"),
             "NPUFLOW_SINTEL": find_root("training/clean/alley_1/frame_0001.png"),
             "NPUFLOW_KITTI": find_root("training/flow_occ/000000_10.png")}
    for key, value in roots.items():
        if value:
            os.environ[key] = value
        print(key, "=", value or "NOT FOUND")

    from clearml.backend_api import Session
    session = Session()
    host = socket.gethostname()

    def call(service, action, **payload):
        return session.send_request(service=service, action=action, json=payload).json()["data"]

    queue_id = next(q["id"] for q in call("queues", "get_all", name=QUEUE)["queues"] if q["name"] == QUEUE)

    def queue_length():
        return len(call("queues", "get_by_id", queue=queue_id)["queue"].get("entries", []))

    def busy():
        return any(w["id"].startswith(host) and w.get("task") for w in call("workers", "get_all")["workers"])

    env = dict(os.environ)
    env["CLEARML_AGENT_SKIP_PIP_VENV_INSTALL"] = sys.executable   # use Kaggle's Python with its torch
    env["CLEARML_AGENT_SKIP_PYTHON_ENV_INSTALL"] = "1"
    env["NPUFLOW_DEADLINE"] = str(START + (MAX_HOURS - 0.25) * 3600)  # train.py saves and stops by then

    print(f"queue '{QUEUE}': {queue_length()} task(s) waiting", flush=True)
    agent = subprocess.Popen(["clearml-agent", "daemon", "--queue", QUEUE, "--foreground"], env=env)
    idle_since = time.time()
    try:
        while agent.poll() is None:
            time.sleep(60)
            hours = (time.time() - START) / 3600
            if hours >= MAX_HOURS:
                print(f"{hours:.1f} h: session time is over, stopping the agent")
                break
            try:
                active = busy() or queue_length() > 0
            except Exception as e:      # a network hiccup must not kill a running training
                print("status check failed:", e)
                active = True
            if active:
                idle_since = time.time()
            elif time.time() - idle_since > IDLE_MINUTES * 60:
                print(f"nothing to do for {IDLE_MINUTES:.0f} min, stopping the agent")
                break
    finally:
        if agent.poll() is None:
            agent.terminate()
            try:
                agent.wait(timeout=120)
            except subprocess.TimeoutExpired:
                agent.kill()
    print(f"agent finished after {(time.time() - START) / 3600:.2f} h")


if __name__ == "__main__":
    main()

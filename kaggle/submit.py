#!/usr/bin/env python3
"""Queue a training run for the Kaggle agent, or put a stopped run back in the queue.

    python kaggle/submit.py --name C3-probe --config configs/chairs_probe.yaml --variant C3_1d_ps_nopos_relu
    python kaggle/submit.py --resume <task id>
    then start the agent notebook from the Kaggle editor (Save Version); API pushes have no access to secrets

The agent clones the repository at the given branch, so push your commits first.
"""
import argparse
import subprocess

from clearml import Task

REPO = "https://github.com/ryngach/NPUFlow.git"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name")
    ap.add_argument("--config", default="configs/proxy.yaml")
    ap.add_argument("--variant", default="")
    ap.add_argument("--set", action="append", default=[])
    ap.add_argument("--max-hours", type=float, default=0,
                    help="stop cleanly (state saved) after this many hours, e.g. to fit the rest of a GPU quota")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--queue", default="kaggle")
    ap.add_argument("--project", default="NPUFlow/train")
    ap.add_argument("--resume", metavar="TASK_ID", help="re-enqueue a stopped task; it continues from its saved state")
    args = ap.parse_args()

    if args.resume:
        task = Task.get_task(task_id=args.resume)
        if str(task.status) not in ("stopped", "failed", "completed", "created"):
            ap.error(f"task is {task.status}; stop it first")
        # the agent pins the commit it ran; move the task to the branch head so that a resumed
        # run picks up fixes pushed since (artifacts such as train_state are kept)
        head = subprocess.run(["git", "ls-remote", REPO, f"refs/heads/{args.branch}"], text=True,
                              capture_output=True, check=True).stdout.split()[0]
        task.session.send_request("tasks", "edit", json={
            "task": task.id, "force": True,
            "script": {**task.data.script.to_dict(), "branch": args.branch, "version_num": head}})
        print(f"code: {args.branch} @ {head[:7]}")
        task.set_parameter("Args/max_hours", args.max_hours or "")   # the limit applies to this submission only
    else:
        if not args.name:
            ap.error("--name is required for a new run")
        task = Task.create(project_name=args.project, task_name=args.name, repo=REPO, branch=args.branch,
                           script="train.py", add_task_init_call=False)
        task.set_parameters({"Args/config": args.config, "Args/name": args.name, "Args/variant": args.variant,
                             "Args/set": repr(args.set), "Args/project": args.project,
                             "Args/max_hours": args.max_hours or ""})
        if args.variant:
            task.add_tags([args.variant])
    Task.enqueue(task, queue_name=args.queue)
    print(f"task {task.id} ({task.name}) is in queue '{args.queue}': {task.get_output_log_web_page()}")


if __name__ == "__main__":
    main()

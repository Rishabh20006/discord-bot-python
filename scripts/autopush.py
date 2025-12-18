#!/usr/bin/env python3
"""
Simple file-watcher that auto-stages, commits and pushes changes to `origin main`.

Usage:
  .venv/bin/python scripts/autopush.py

Notes:
 - Excludes `.git`, `.venv`, `__pycache__`, and `.vscode`.
 - Commits are debounced (2s) to avoid many small commits.
 - Requires Git authentication already configured in the Codespace (ssh or gh auth).
"""
import os
import sys
import time
import threading
import subprocess
from pathlib import Path

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except Exception:
    print("Missing dependency: install with 'pip install watchdog'", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
EXCLUDE = {'.git', '.venv', '__pycache__', '.vscode'}
DEBOUNCE_SECONDS = 2.0
BRANCH = os.environ.get('AUTOPUSH_BRANCH', 'main')
REMOTE = os.environ.get('AUTOPUSH_REMOTE', 'origin')


def run_cmd(cmd, cwd=ROOT):
    try:
        res = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
        return res.returncode, res.stdout.strip(), res.stderr.strip()
    except Exception as e:
        return 1, '', str(e)


class ChangeHandler(FileSystemEventHandler):
    def __init__(self, on_change):
        super().__init__()
        self.on_change = on_change

    def _should_ignore(self, path: str):
        parts = Path(path).parts
        return any(p in EXCLUDE for p in parts)

    def on_any_event(self, event):
        if event.is_directory:
            return
        if self._should_ignore(event.src_path):
            return
        self.on_change(event.src_path)


class AutoPusher:
    def __init__(self):
        self._timer = None
        self._lock = threading.Lock()

    def schedule(self, src_path):
        with self._lock:
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._commit_and_push)
            self._timer.start()

    def _commit_and_push(self):
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
        # Stage everything
        code, out, err = run_cmd(['git', 'add', '-A'])
        if code != 0:
            print('git add failed:', err)
            return

        # Check if there is anything to commit
        code, out, err = run_cmd(['git', 'status', '--porcelain'])
        if code != 0:
            print('git status failed:', err)
            return
        if not out:
            # nothing to commit
            return

        msg = f'Auto-commit: {timestamp}'
        code, out, err = run_cmd(['git', 'commit', '-m', msg])
        if code != 0:
            print('git commit failed or nothing to commit; stderr:', err)
            # continue to push if commit produced something

        print('Pushing to', f'{REMOTE}/{BRANCH}...')
        code, out, err = run_cmd(['git', 'push', REMOTE, BRANCH])
        if code != 0:
            print('git push failed:', err)
        else:
            print('Pushed at', timestamp)


def main():
    print('Starting auto-push watcher in', ROOT)

    # sanity check: ensure we're inside a git repo
    code, out, err = run_cmd(['git', 'rev-parse', '--is-inside-work-tree'])
    if code != 0:
        print('Not a git repository:', err)
        sys.exit(1)

    pusher = AutoPusher()

    def on_change(path):
        print('Change detected:', path)
        pusher.schedule(path)

    event_handler = ChangeHandler(on_change)
    observer = Observer()
    observer.schedule(event_handler, str(ROOT), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == '__main__':
    main()

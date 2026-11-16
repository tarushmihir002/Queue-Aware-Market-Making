#!/usr/bin/env python
"""
One command to host the React dashboard locally.

Builds the frontend if needed (npm install + npm run build), then starts the
FastAPI server which serves BOTH the API and the built app at http://127.0.0.1:8000.

    python scripts/serve_dashboard.py                 # build if needed, then serve
    python scripts/serve_dashboard.py --rebuild       # force a fresh frontend build
    python scripts/serve_dashboard.py --no-build      # serve whatever is already built
    python scripts/serve_dashboard.py --port 8080

For live frontend development instead, run two processes:
    uvicorn app.backend.main:app --port 8000          # terminal 1 (API)
    cd app/frontend && npm install && npm run dev      # terminal 2 (Vite @ :5173, proxies /api)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FE = ROOT / "app" / "frontend"
DIST = FE / "dist"


def sh(cmd, cwd):
    print(f"$ {' '.join(cmd)}  (in {cwd})")
    subprocess.run(cmd, cwd=cwd, check=True, shell=(sys.platform == "win32"))


def ensure_build(rebuild: bool):
    npm = shutil.which("npm")
    if npm is None:
        print("npm not found — install Node.js to build the frontend, or use --no-build.")
        sys.exit(1)
    if rebuild or not (FE / "node_modules").exists():
        sh(["npm", "install"], FE)
    if rebuild or not DIST.exists():
        sh(["npm", "run", "build"], FE)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--no-build", action="store_true")
    args = ap.parse_args()

    if not args.no_build:
        ensure_build(args.rebuild)
    if not DIST.exists():
        print("warning: frontend not built; the server will show an API-only message at /.")

    import uvicorn
    print(f"\n  Dashboard → http://{args.host}:{args.port}\n")
    uvicorn.run("app.backend.main:app", host=args.host, port=args.port, app_dir=str(ROOT))


if __name__ == "__main__":
    main()

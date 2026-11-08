"""Command line: run the pipeline, print the policy table, enforce the CI gate, serve the API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import pipeline


def _table(report: dict) -> str:
    lines = [
        "| Policy | Represented | Won | Win rate | Recovered | Handling + fees | Net recovered | vs fight everything |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for p in report["policies"]:
        lines.append(
            f"| {p['label']} | {p['fought']:,} | {p['won']:,} | {100 * p['win_rate']:.1f}% | ${p['recovered']:,.0f} | "
            f"${p['handling_cost'] + p['loss_fees']:,.0f} | **${p['net']:,.0f}** | {p['vs_fight_everything']:+,.0f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="disputedesk")
    ap.add_argument("--data-dir", default=os.environ.get("DISPUTEDESK_DATA_DIR", "data"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("all", help="generate, train, score, compare policies, analytics, seed the store")
    a.add_argument("--txns", type=int, default=300_000)
    a.add_argument("--customers", type=int, default=90_000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--gate", action="store_true", help="exit 1 unless the model policy beats fight-everything")
    sub.add_parser("policy", help="print the policy comparison table from the last run")
    sub.add_parser("gate", help="exit 1 unless the model policy beats fight-everything on the held-out set")
    s = sub.add_parser("serve", help="run the API and UI")
    s.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    s.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8833")))
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)

    if args.cmd == "all":
        report = pipeline.run(data_dir, n_txns=args.txns, n_customers=args.customers, seed=args.seed)
        print(_table(report))
        if args.gate:
            ok, msg = pipeline.gate(report)
            print(msg)
            return 0 if ok else 1
        return 0
    if args.cmd in ("policy", "gate"):
        path = data_dir / "report.json"
        if not path.exists():
            print(f"no report at {path}; run `disputedesk all` first", file=sys.stderr)
            return 2
        report = json.loads(path.read_text())
        if args.cmd == "policy":
            print(_table(report))
            return 0
        ok, msg = pipeline.gate(report)
        print(msg)
        return 0 if ok else 1
    if args.cmd == "serve":
        import uvicorn

        os.environ["DISPUTEDESK_DATA_DIR"] = str(data_dir)
        uvicorn.run("disputedesk.app:main_app", factory=True, host=args.host, port=args.port, log_level="warning")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

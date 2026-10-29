"""Freeze, then execute a bounded training/validation-only convergence diagnostic."""

import argparse
import json
from pathlib import Path

from neural_recommender.convergence import freeze, run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("freeze", "run"))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "freeze":
        report = freeze(args.project, args.protocol)
        print(json.dumps({"frozen": str(args.protocol), "created_at": report["created_at"]}))
    else:
        if args.output is None:
            parser.error("run requires --output")
        report = run(args.project, args.protocol, args.output)
        print(json.dumps({"output": str(args.output), "status": report["status"]}))


if __name__ == "__main__":
    main()

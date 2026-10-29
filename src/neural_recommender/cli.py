"""Train and inspect the same neural recommender served by the API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data import fetch_dataset
from .training import TrainConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch = commands.add_parser("download")
    fetch.add_argument("--data-dir", type=Path, default=Path("data"))
    run = commands.add_parser("train")
    run.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    run.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    run.add_argument("--report", type=Path, default=Path("reports/benchmark.json"))
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--epochs", type=int, default=5)
    run.add_argument("--max-train-users", type=int, default=2500)
    run.add_argument("--evaluation-users", type=int, default=250)
    optimized = commands.add_parser("optimize")
    optimized.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    optimized.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    optimized.add_argument("--report", type=Path, default=Path("reports/optimized_benchmark.json"))
    optimized.add_argument("--protocol", type=Path, default=Path("reports/upgrade_protocol.json"))
    optimized.add_argument(
        "--pilot-metadata", type=Path, default=Path("reports/initial_pilot_metadata.json")
    )
    optimized.add_argument("--epochs", type=int, default=5)
    optimized.add_argument("--max-train-users", type=int, default=2500)
    optimized.add_argument("--evaluation-users", type=int, default=200)
    refinement = commands.add_parser("refine", help="Fit ALS and history-conditional selectors")
    refinement.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    refinement.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    refinement.add_argument("--report", type=Path, default=Path("reports/refined_benchmark.json"))
    refinement.add_argument(
        "--protocol", type=Path, default=Path("reports/refinement_protocol.json")
    )
    refinement.add_argument(
        "--upgrade-protocol", type=Path, default=Path("reports/upgrade_protocol.json")
    )
    refinement.add_argument(
        "--pilot-metadata", type=Path, default=Path("reports/initial_pilot_metadata.json")
    )
    rollback = commands.add_parser(
        "rollback", help="Restore the previously validated serving generation"
    )
    refinement.add_argument("--reserve-users", type=int, default=200)
    rollback.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    audit = commands.add_parser(
        "audit-serving", help="Reproduce frozen metrics and serving rankings"
    )
    audit.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    audit.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    audit.add_argument("--frozen-report", type=Path, default=Path("reports/refined_benchmark.json"))
    audit.add_argument("--output", type=Path, default=Path("reports/serving_audit.json"))
    leakage = commands.add_parser(
        "audit-leakage", help="Reconstruct recorded split/selection and fitting isolation"
    )
    leakage.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    leakage.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    leakage.add_argument("--reports", type=Path, default=Path("reports"))
    leakage.add_argument("--output", type=Path, default=Path("reports/leakage_audit.json"))
    recommend = commands.add_parser("recommend", help="Rank films with optional genre diversity")
    recommend.add_argument("--movie-ids", type=int, nargs="+", required=True)
    recommend.add_argument("--mode", default="accuracy")
    recommend.add_argument("--k", type=int, default=10)
    recommend.add_argument("--relevance-weight", type=float, default=1.0)
    recommend.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    recommend.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    diversity = commands.add_parser(
        "evaluate-diversity", help="Describe a frozen genre-diversity tradeoff grid without tuning"
    )
    diversity.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    diversity.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    diversity.add_argument(
        "--frozen-report", type=Path, default=Path("reports/refined_benchmark.json")
    )
    diversity.add_argument("--protocol", type=Path, default=Path("reports/diversity_protocol.json"))
    diversity.add_argument("--output", type=Path, default=Path("reports/diversity_tradeoffs.json"))
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8002)
    serve.add_argument("--data", type=Path, default=Path("data/ml-1m.zip"))
    serve.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    if args.command == "download":
        print(fetch_dataset(args.data_dir))
    elif args.command == "train":
        report = train(
            args.data,
            args.artifacts,
            args.report,
            TrainConfig(
                seed=args.seed,
                epochs=args.epochs,
                max_train_users=args.max_train_users,
                evaluation_users=args.evaluation_users,
            ),
        )
        print(json.dumps({key: value for key, value in report.items() if key != "test"}, indent=2))
    elif args.command == "optimize":
        from .optimization import OptimizationConfig, optimize

        report = optimize(
            args.data,
            args.artifacts,
            args.report,
            args.protocol,
            args.pilot_metadata,
            OptimizationConfig(
                epochs=args.epochs,
                train_users=args.max_train_users,
                evaluation_users=args.evaluation_users,
            ),
        )
        print(
            json.dumps(
                {"selected": report["selected"], "elapsed_seconds": report["elapsed_seconds"]},
                indent=2,
            )
        )
    elif args.command == "refine":
        from .refinement import refine

        report = refine(
            args.data,
            args.artifacts,
            args.protocol,
            args.report,
            args.upgrade_protocol,
            args.pilot_metadata,
            args.reserve_users,
        )
        print(
            json.dumps(
                {
                    "selected_als": report["selected_als"],
                    "history_selector": report["history_selector"],
                    "elapsed_seconds": report["elapsed_seconds"],
                },
                indent=2,
            )
        )
    elif args.command == "rollback":
        from .bundles import rollback_bundle

        print(rollback_bundle(args.artifacts))
    elif args.command == "audit-serving":
        from .audit import audit_serving

        print(
            json.dumps(
                audit_serving(args.data, args.artifacts, args.frozen_report, args.output), indent=2
            )
        )
    elif args.command == "audit-leakage":
        from .leakage_audit import audit_leakage

        print(
            json.dumps(
                audit_leakage(args.data, args.artifacts, args.reports, args.output), indent=2
            )
        )
    elif args.command == "recommend":
        from .service import RecommenderService

        service = RecommenderService(args.data, args.artifacts)
        print(
            json.dumps(
                service.recommend(args.movie_ids, args.k, args.mode, args.relevance_weight),
                indent=2,
            )
        )
    elif args.command == "evaluate-diversity":
        from .diversity_evaluation import evaluate_diversity

        report = evaluate_diversity(
            args.data, args.artifacts, args.frozen_report, args.protocol, args.output
        )
        print(json.dumps(report["results"], indent=2))
    else:
        import logging

        import uvicorn

        from .api import create_app

        logging.basicConfig(level=logging.INFO)
        uvicorn.run(
            create_app(args.data, args.artifacts),
            host="127.0.0.1",
            port=args.port,
            limit_concurrency=16,
            access_log=False,
        )


if __name__ == "__main__":
    main()

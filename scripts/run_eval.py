"""CLI wrapper for the evaluation pipeline.

Usage:
    python scripts/run_eval.py                  # full dataset
    python scripts/run_eval.py --limit 20       # quick sanity check
    python scripts/run_eval.py --dataset path/to/custom.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure src/ is on the path when run as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analyzer.config import get_settings
from analyzer.eval.runner import run_eval
from analyzer.llm.client import LLMClient


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the LLM test failure analyzer evaluation pipeline."
    )
    parser.add_argument(
        "--dataset",
        default=None,
        metavar="PATH",
        help="Path to labeled_failures.json (default: value from settings)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Evaluate only the first N cases (useful for quick checks)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Confidence threshold for routing (default: from settings)",
    )
    return parser.parse_args()


def _print_report(report: object) -> None:
    from analyzer.models.eval import EvalReport

    assert isinstance(report, EvalReport)

    print("\n" + "=" * 60)
    print("EVALUATION REPORT")
    print("=" * 60)
    print(f"  Total cases  : {report.n_total}")
    print(f"  Correct      : {report.n_correct}  ({report.accuracy:.1%})")
    print(f"  Accuracy     : {report.accuracy:.4f}")
    print(f"  Macro P      : {report.macro_precision:.4f}")
    print(f"  Macro R      : {report.macro_recall:.4f}")
    print(f"  Macro F1     : {report.macro_f1:.4f}")
    print(f"  Avg latency  : {report.avg_latency_ms:.0f}ms")
    print(f"  Fallback rate: {report.fallback_rate:.1%}")
    print()
    print(f"{'Category':<22} {'P':>6} {'R':>6} {'F1':>6} {'Support':>8}")
    print("-" * 52)
    for m in sorted(report.per_category, key=lambda x: x.category):
        print(
            f"  {m.category:<20} {m.precision:>6.3f} {m.recall:>6.3f}"
            f" {m.f1:>6.3f} {m.support:>8}"
        )
    print("=" * 60 + "\n")


async def main() -> None:
    args = _parse_args()
    settings = get_settings()

    dataset_path = args.dataset or settings.eval_dataset_path
    threshold = args.threshold or settings.confidence_threshold

    if not Path(dataset_path).exists():
        print(f"ERROR: dataset not found at {dataset_path!r}", file=sys.stderr)
        print(
            "Tip: run `python scripts/generate_eval_dataset.py` to create it.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Dataset : {dataset_path}")
    if args.limit:
        print(f"Limit   : {args.limit} cases")
    print(f"Model   : {settings.model_id}")
    print(f"Threshold: {threshold}")
    print("Running eval… (this makes real API calls)\n")

    client = LLMClient(settings)
    report = await run_eval(
        dataset_path=dataset_path,
        client=client,
        confidence_threshold=threshold,
        limit=args.limit,
    )
    _print_report(report)


if __name__ == "__main__":
    asyncio.run(main())

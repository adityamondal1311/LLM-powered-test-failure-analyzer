from __future__ import annotations

from collections import defaultdict

from analyzer.models.eval import CategoryMetrics, CaseResult, EvalReport
from analyzer.models.pipeline import FailureCategory


def compute_report(case_results: list[CaseResult]) -> EvalReport:
    """Compute precision/recall/F1 per category and overall accuracy."""
    n_total = len(case_results)
    n_correct = sum(1 for r in case_results if r.correct)

    # Build confusion matrix counts
    tp: dict[str, int] = defaultdict(int)
    fp: dict[str, int] = defaultdict(int)
    fn: dict[str, int] = defaultdict(int)

    for r in case_results:
        pred = r.predicted.value
        truth = r.ground_truth.value
        if pred == truth:
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1

    per_category: list[CategoryMetrics] = []
    for cat in FailureCategory:
        c = cat.value
        support = tp[c] + fn[c]
        if support == 0:
            continue
        precision = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        recall = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_category.append(
            CategoryMetrics(
                category=cat,
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1=round(f1, 4),
                support=support,
            )
        )

    n_cats = len(per_category)
    macro_p = sum(c.precision for c in per_category) / n_cats if n_cats else 0.0
    macro_r = sum(c.recall for c in per_category) / n_cats if n_cats else 0.0
    macro_f1 = sum(c.f1 for c in per_category) / n_cats if n_cats else 0.0

    avg_latency = sum(r.latency_ms for r in case_results) / n_total if n_total else 0.0
    fallback_rate = (
        sum(1 for r in case_results if r.fallback_used) / n_total if n_total else 0.0
    )

    return EvalReport(
        n_total=n_total,
        n_correct=n_correct,
        accuracy=round(n_correct / n_total, 4) if n_total else 0.0,
        macro_precision=round(macro_p, 4),
        macro_recall=round(macro_r, 4),
        macro_f1=round(macro_f1, 4),
        per_category=per_category,
        case_results=case_results,
        avg_latency_ms=round(avg_latency, 2),
        fallback_rate=round(fallback_rate, 4),
    )

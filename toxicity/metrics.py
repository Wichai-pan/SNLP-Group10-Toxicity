from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    f1_macro: float
    f1_toxic: float


def _require_sklearn() -> None:
    try:
        import sklearn  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "scikit-learn is required for metrics. Install it via requirements.txt."
        ) from e


def compute_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Metrics:
    _require_sklearn()
    from sklearn.metrics import accuracy_score, f1_score

    return Metrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1_macro=float(f1_score(y_true, y_pred, average="macro")),
        f1_toxic=float(f1_score(y_true, y_pred, average="binary", pos_label=1)),
    )


def compute_metrics_by_lang(
    y_true: Sequence[int], y_pred: Sequence[int], langs: Sequence[str]
) -> Mapping[str, Metrics]:
    if not (len(y_true) == len(y_pred) == len(langs)):
        raise ValueError("y_true, y_pred, langs must have the same length.")

    bucket_true: dict[str, list[int]] = defaultdict(list)
    bucket_pred: dict[str, list[int]] = defaultdict(list)
    for yt, yp, lang in zip(y_true, y_pred, langs):
        bucket_true[lang].append(int(yt))
        bucket_pred[lang].append(int(yp))

    return {lang: compute_metrics(bucket_true[lang], bucket_pred[lang]) for lang in bucket_true}


def format_metrics(name: str, m: Metrics) -> str:
    return (
        f"{name:>8} | acc={m.accuracy:.4f}  f1_macro={m.f1_macro:.4f}  f1_toxic={m.f1_toxic:.4f}"
    )


def print_report(
    *,
    title: str,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    langs: Sequence[str],
    lang_order: Iterable[str] = ("eng", "ger", "fin", "unk"),
) -> None:
    overall = compute_metrics(y_true, y_pred)
    by_lang = compute_metrics_by_lang(y_true, y_pred, langs)

    print(f"\n== {title} ==")
    print(format_metrics("overall", overall))
    print("-- by_lang --")
    printed = set()
    for lang in lang_order:
        if lang in by_lang:
            print(format_metrics(lang, by_lang[lang]))
            printed.add(lang)
    for lang in sorted(set(by_lang.keys()) - printed):
        print(format_metrics(lang, by_lang[lang]))


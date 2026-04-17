from __future__ import annotations

import argparse
import csv
from pathlib import Path

from toxicity.data import iter_ids, iter_labels, iter_langs, iter_texts, read_tsv
from toxicity.metrics import print_report
from toxicity.utils import ensure_dir, seed_everything


def build_vectorizer_args(analyzer: str, max_features: int, min_df: int) -> dict:
    if analyzer == "char_wb":
        return {
            "analyzer": "char_wb",
            "ngram_range": (3, 5),
            "min_df": min_df,
            "max_features": max_features,
        }
    if analyzer == "word":
        return {
            "analyzer": "word",
            "ngram_range": (1, 2),
            "min_df": min_df,
            "max_features": max_features,
        }
    raise ValueError(f"Unknown analyzer: {analyzer}")


def write_predictions_tsv(ids: list[str], preds: list[int], out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["id", "label"])
        for example_id, pred in zip(ids, preds):
            w.writerow([example_id, int(pred)])


def main() -> None:
    ap = argparse.ArgumentParser(description="TF-IDF + Logistic Regression baseline.")
    ap.add_argument("--train_tsv", type=str, default="train.tsv")
    ap.add_argument("--dev_tsv", type=str, default="dev.tsv")
    ap.add_argument("--test_tsv", type=str, default="test.tsv")

    ap.add_argument("--analyzer", type=str, choices=["char_wb", "word"], default="char_wb")
    ap.add_argument("--max_features", type=int, default=200_000)
    ap.add_argument("--min_df", type=int, default=2)
    ap.add_argument("--class_weight", type=str, choices=["none", "balanced"], default="none")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--model_out", type=str, default="artifacts/baseline.joblib")

    ap.add_argument("--predict_test", action="store_true")
    ap.add_argument("--out_tsv", type=str, default="artifacts/pred_baseline_test.tsv")

    args = ap.parse_args()
    seed_everything(args.seed)
    ensure_dir("artifacts")

    train = read_tsv(args.train_tsv, has_label=True)
    dev = read_tsv(args.dev_tsv, has_label=True)

    _require = "scikit-learn"
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"{_require} is required. Install it via requirements.txt.") from e

    vectorizer = TfidfVectorizer(**build_vectorizer_args(args.analyzer, args.max_features, args.min_df))
    class_weight = None if args.class_weight == "none" else "balanced"
    clf = LogisticRegression(
        solver="saga",
        max_iter=2000,
        n_jobs=-1,
        class_weight=class_weight,
    )
    pipe = Pipeline([("tfidf", vectorizer), ("clf", clf)])

    x_train = iter_texts(train)
    y_train = iter_labels(train)
    pipe.fit(x_train, y_train)

    x_dev = iter_texts(dev)
    y_dev = iter_labels(dev)
    langs_dev = iter_langs(dev)
    pred_dev = pipe.predict(x_dev).astype(int).tolist()

    print_report(title="DEV (baseline)", y_true=y_dev, y_pred=pred_dev, langs=langs_dev)

    try:
        import joblib
    except Exception as e:  # pragma: no cover
        raise RuntimeError("joblib is required to save the baseline model.") from e
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, args.model_out)
    print(f"\nSaved baseline model to: {args.model_out}")

    if args.predict_test:
        test = read_tsv(args.test_tsv, has_label=False)
        x_test = iter_texts(test)
        pred_test = pipe.predict(x_test).astype(int).tolist()
        write_predictions_tsv(iter_ids(test), pred_test, args.out_tsv)
        print(f"Wrote test predictions to: {args.out_tsv}")


if __name__ == "__main__":
    main()


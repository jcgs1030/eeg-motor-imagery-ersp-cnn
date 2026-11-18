"""
diagnostic_ea_invariance.py
----------------------------
Diagnostic only — does NOT touch the production pipeline or any committed
result file.

Finding under test: Experiment 4 (`train_ea_csp_lda.py`) reports results that
are identical to Experiment 3 (`train_csp_lda.py`) to every printed digit, on
all 9 subjects and all 5 metrics. `EXPERIMENTS.md` originally read that as a
substantive negative result about the offline->online shift. It is not: the
Euclidean Alignment step cannot change anything downstream of MNE's `CSP`,
because `CSP` already whitens by the composite covariance of the training
set, which makes its filters equivariant to any invertible linear spatial
transform applied to the data.

Concretely, with `W` the filters CSP learns on raw epochs and
`R = R_train^(-1/2)` the EA whitener, CSP trained on `R X` learns `W R^(-1)`,
so the log-variance features `log var(W R^(-1) R X) = log var(W X)` are
unchanged. The test split is whitened by its own `R_test^(-1/2)`, which is a
different matrix — yet the features still come out identical, because the
whitening CSP applies internally absorbs it as well.

This script measures that directly: it computes the CSP log-variance features
and the LDA predictions with and without EA, for the requested subjects, and
reports the maximum absolute feature difference and whether the predicted
labels are identical.

Usage:
    python src/analysis/diagnostics/diagnostic_ea_invariance.py
    python src/analysis/diagnostics/diagnostic_ea_invariance.py --subjects 1 4
"""

import argparse

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

from src.config import SUBJECTS
from src.p3_classification.classical.train_ea_csp_lda import (
    euclidean_alignment,
    extract_csp_features,
    load_epochs,
)


def compare_subject(subject: int) -> dict:
    """CSP+LDA with and without Euclidean Alignment, same subject, same seed."""
    X_train_raw, y_train = load_epochs(subject, "T")
    X_test_raw,  y_test  = load_epochs(subject, "E")

    # ── Without EA (Experiment 3) ──
    f_train, f_test, _ = extract_csp_features(X_train_raw, y_train, X_test_raw)
    pred_plain = LinearDiscriminantAnalysis(solver="svd").fit(
        f_train, y_train).predict(f_test)

    # ── With EA, applied independently per split (Experiment 4) ──
    X_train_ea, _ = euclidean_alignment(X_train_raw)
    X_test_ea,  _ = euclidean_alignment(X_test_raw)
    g_train, g_test, _ = extract_csp_features(X_train_ea, y_train, X_test_ea)
    pred_ea = LinearDiscriminantAnalysis(solver="svd").fit(
        g_train, y_train).predict(g_test)

    return {
        "subject": subject,
        "acc_plain": accuracy_score(y_test, pred_plain),
        "acc_ea": accuracy_score(y_test, pred_ea),
        "train_feat_maxdiff": float(np.abs(f_train - g_train).max()),
        "test_feat_maxdiff": float(np.abs(f_test - g_test).max()),
        "preds_identical": bool(np.array_equal(pred_plain, pred_ea)),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Diagnostic: is Euclidean Alignment a no-op under CSP?"
    )
    parser.add_argument("--subjects", type=int, nargs="+", default=SUBJECTS,
                        help="Subjects to test (default: all 9)")
    args = parser.parse_args()

    print("\n══════════════════════════════════════════════")
    print("  Diagnostic — EA invariance under MNE's CSP")
    print("══════════════════════════════════════════════\n")
    print(f"  {'Subj':>5} {'CSP+LDA':>9} {'EA+CSP+LDA':>11} "
          f"{'max|Δfeat| train':>17} {'max|Δfeat| test':>16} {'same preds':>11}")

    rows = []
    for subject in args.subjects:
        try:
            r = compare_subject(subject)
        except FileNotFoundError as e:
            print(f"  S{subject:02d}: skipped — {e}")
            continue
        rows.append(r)
        print(f"  S{r['subject']:02d}   {r['acc_plain']:>8.2%} "
              f"{r['acc_ea']:>10.2%} {r['train_feat_maxdiff']:>17.2e} "
              f"{r['test_feat_maxdiff']:>16.2e} {str(r['preds_identical']):>11}")

    if not rows:
        print("\n  No epoch files found. Run preprocessing.py first.")
        return

    all_identical = all(r["preds_identical"] for r in rows)
    worst = max(max(r["train_feat_maxdiff"], r["test_feat_maxdiff"]) for r in rows)
    print(f"\n  Predictions identical on {sum(r['preds_identical'] for r in rows)}"
          f"/{len(rows)} subjects.")
    print(f"  Largest feature difference anywhere: {worst:.2e} "
          f"(float64 epsilon scale).")
    if all_identical:
        print("\n  Conclusion: EA is algebraically a no-op here. Experiment 4 is")
        print("  not evidence about the offline->online shift — it cannot differ")
        print("  from Experiment 3 by construction. To give EA any chance of")
        print("  mattering it has to feed a classifier that does NOT whiten")
        print("  internally (see EXPERIMENTS.md, Experiment 4).")


if __name__ == "__main__":
    main()

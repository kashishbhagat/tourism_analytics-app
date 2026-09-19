"""
End-to-end pipeline: raw files -> cleaned master -> EDA -> SQL -> models.

Run everything:
    python run_pipeline.py

Run selected stages (useful while iterating on one model):
    python run_pipeline.py --stages clean eda
    python run_pipeline.py --skip-eda

Outputs
    data/processed/master_dataset.parquet   analysis-ready dataset
    data/processed/tourism.db               SQLite database for the SQL layer
    reports/cleaning_log.md                 auditable record of every clean step
    reports/eda_report.md + figures/        EDA with findings
    models/*.joblib                         trained models
    models/metrics.json                     all evaluation metrics
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime

import pandas as pd

from src.config import MASTER_CSV, MASTER_PARQUET, METRICS_PATH, SQLITE_DB
from src.cleaning import run_cleaning
from src.data_loader import load_raw_tables, summarise_tables
from src.features import add_basic_features

warnings.filterwarnings("ignore", category=UserWarning)

ALL_STAGES = ["clean", "eda", "sql", "regression", "classification", "recommendation"]


def header(text: str) -> None:
    print(f"\n{'=' * 72}\n{text}\n{'=' * 72}")


def stage(text: str) -> None:
    print(f"\n[{datetime.now():%H:%M:%S}] {text}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", choices=ALL_STAGES, default=ALL_STAGES,
                    help="Which stages to run (default: all).")
    ap.add_argument("--skip-eda", action="store_true", help="Skip figure generation.")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    stages = [s for s in args.stages if not (args.skip_eda and s == "eda")]
    verbose = not args.quiet
    t_start = time.time()
    metrics: dict = {"generated_at": datetime.now().isoformat(timespec="seconds")}

    header("TOURISM EXPERIENCE ANALYTICS -- PIPELINE")

    # ---------------------------------------------------------------- load
    stage("Loading raw tables")
    try:
        tables = load_raw_tables()
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}\n")
        print("Hint: run `python scripts/generate_sample_data.py` to create a "
              "sample dataset, or place the real files in data/raw/.")
        return 1

    if verbose:
        print(summarise_tables(tables).to_string(index=False))

    # --------------------------------------------------------------- clean
    if "clean" in stages or not MASTER_PARQUET.exists():
        stage("Cleaning and integrating")
        master, log = run_cleaning(tables)
        log.save()
        master = add_basic_features(master)

        master.to_parquet(MASTER_PARQUET, index=False)
        master.to_csv(MASTER_CSV, index=False)
        print(f"    {len(log)} cleaning steps logged -> reports/cleaning_log.md")
        print(f"    master dataset: {master.shape[0]:,} rows x {master.shape[1]} cols")
        metrics["cleaning"] = {
            "steps": len(log),
            "rows": int(len(master)),
            "columns": int(master.shape[1]),
        }
    else:
        stage("Loading cached master dataset")
        master = pd.read_parquet(MASTER_PARQUET)

    # ----------------------------------------------------------------- eda
    if "eda" in stages:
        stage("Exploratory data analysis")
        from src.eda import run_eda
        eda_result = run_eda(master, verbose=verbose)
        metrics["eda"] = eda_result["overview"]
        metrics["eda"]["findings"] = eda_result["findings"]

    # ----------------------------------------------------------------- sql
    if "sql" in stages:
        stage("Building SQL database and running analytical queries")
        from src.database import build_database, run_all
        build_database(master, tables)
        print(f"    database: {SQLITE_DB}")
        results = run_all(verbose=verbose)

        dq = results.get("data_quality_check")
        if dq is not None:
            violations = int(dq["violations"].sum())
            status = "PASS" if violations == 0 else f"FAIL ({violations} violations)"
            print(f"    data quality assertions: {status}")
            metrics["data_quality"] = {
                "status": status,
                "checks": dq.to_dict(orient="records"),
            }

    # -------------------------------------------------------------- models
    if "regression" in stages:
        stage("Task 1 -- Regression: predicting attraction ratings")
        from src.models import regression
        metrics["regression"] = regression.train(master, verbose=verbose)

    if "classification" in stages:
        stage("Task 2 -- Classification: predicting visit mode")
        from src.models import classification
        metrics["classification"] = classification.train(master, verbose=verbose)

    if "recommendation" in stages:
        stage("Task 3 -- Recommendation: personalised attractions")
        from src.models import recommender
        metrics["recommendation"] = recommender.train(master, verbose=verbose)

    # ------------------------------------------------------------- persist
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")

    header("PIPELINE COMPLETE")
    print(f"Elapsed: {time.time() - t_start:.1f}s")
    print(f"Metrics: {METRICS_PATH}")

    if "regression" in metrics:
        r = metrics["regression"]
        print(f"  Regression     {r['best_model']:<22} test R2  = {r['test_r2']:.4f}")
    if "classification" in metrics:
        c = metrics["classification"]
        print(f"  Classification {c['best_model']:<22} macro F1 = {c['test_f1_macro']:.4f}")
    if "recommendation" in metrics:
        rc = metrics["recommendation"]
        best = rc["comparison"][0] if rc["comparison"] else {}
        print(f"  Recommendation {rc['best_method']:<22} NDCG@10  = "
              f"{best.get('ndcg@10', float('nan')):.4f}")

    print("\nNext: streamlit run app/streamlit_app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())

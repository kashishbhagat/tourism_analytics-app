"""
Raw data loading.

The source dataset ships as a set of per-table files (or one multi-sheet
workbook) whose exact filenames differ between copies. `load_raw_tables`
resolves each logical table by fuzzy-matching filenames and sheet names
against the aliases declared in config.RAW_TABLES, so the pipeline works
whether the user dropped in `Transaction.xlsx`, `transaction data.csv`
or a single `Tourism.xlsx` with nine sheets.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from .config import RAW_DIR, RAW_TABLES

READERS = {
    ".csv": lambda p: pd.read_csv(p),
    ".tsv": lambda p: pd.read_csv(p, sep="\t"),
    ".xlsx": lambda p: pd.read_excel(p),
    ".xls": lambda p: pd.read_excel(p),
    ".xlsm": lambda p: pd.read_excel(p),
}


def _normalise(name: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace for matching."""
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def _score(candidate: str, aliases: list[str]) -> int:
    """
    Rank how well a filename/sheet name matches a table's aliases.
    Exact match beats whole-word containment, which beats substring.
    """
    cand = _normalise(candidate)
    best = 0
    for alias in aliases:
        a = _normalise(alias)
        if cand == a:
            best = max(best, 100)
        elif re.search(rf"\b{re.escape(a)}\b", cand):
            best = max(best, 60 + len(a))
        elif a in cand:
            best = max(best, 30 + len(a))
    return best


def _collect_candidates(raw_dir: Path) -> dict[str, pd.DataFrame]:
    """Map every readable file (and every sheet of every workbook) to a frame."""
    candidates: dict[str, pd.DataFrame] = {}

    for path in sorted(raw_dir.iterdir()):
        if path.name.startswith((".", "~$")) or not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in READERS:
            continue

        if suffix in (".xlsx", ".xls", ".xlsm"):
            book = pd.read_excel(path, sheet_name=None)
            if len(book) == 1:
                # Single-sheet workbook: the filename is the meaningful label.
                candidates[path.stem] = next(iter(book.values()))
            else:
                for sheet, frame in book.items():
                    candidates[sheet] = frame
        else:
            candidates[path.stem] = READERS[suffix](path)

    return candidates


def load_raw_tables(raw_dir: Path | str = RAW_DIR) -> dict[str, pd.DataFrame]:
    """
    Return {logical_table_name: DataFrame} for all nine source tables.

    Raises FileNotFoundError listing exactly which tables could not be
    resolved, rather than failing later with an opaque KeyError.
    """
    raw_dir = Path(raw_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {raw_dir}")

    candidates = _collect_candidates(raw_dir)
    if not candidates:
        raise FileNotFoundError(
            f"No readable data files in {raw_dir}. Place the dataset files there, "
            "or run `python scripts/generate_sample_data.py` to create a sample."
        )

    tables: dict[str, pd.DataFrame] = {}
    used: set[str] = set()
    missing: list[str] = []

    # Greedy best-match assignment, strongest match first, one source per table.
    scored = []
    for logical, spec in RAW_TABLES.items():
        for label in candidates:
            s = _score(label, spec["aliases"])
            if s:
                scored.append((s, logical, label))
    scored.sort(reverse=True)

    for _, logical, label in scored:
        if logical in tables or label in used:
            continue
        tables[logical] = candidates[label].copy()
        used.add(label)

    # Fallback: match on column signature for anything still unresolved.
    for logical, spec in RAW_TABLES.items():
        if logical in tables:
            continue
        want = {c.lower() for c in spec["columns"]}
        for label, frame in candidates.items():
            if label in used:
                continue
            have = {str(c).lower() for c in frame.columns}
            if len(want & have) >= max(2, len(want) - 1):
                tables[logical] = frame.copy()
                used.add(label)
                break
        if logical not in tables:
            missing.append(logical)

    if missing:
        raise FileNotFoundError(
            "Could not resolve these tables in "
            f"{raw_dir}: {', '.join(missing)}.\n"
            f"Files found: {', '.join(sorted(candidates))}.\n"
            "Rename the files to match the table names "
            f"({', '.join(RAW_TABLES)}) or add an alias in src/config.py."
        )

    return tables


def summarise_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compact shape/null overview, used by the pipeline log and the app."""
    rows = []
    for name, df in tables.items():
        rows.append(
            {
                "table": name,
                "rows": len(df),
                "columns": df.shape[1],
                "null_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("rows", ascending=False)


if __name__ == "__main__":
    t = load_raw_tables()
    print(summarise_tables(t).to_string(index=False))

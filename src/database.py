"""
SQL layer.

Loads the cleaned tables into a SQLite database and runs the named queries in
sql/analysis.sql. SQLite is used because it needs no server and keeps the
project reproducible from a clone; the queries are standard SQL and port to
Postgres or MySQL with only the window-function dialect unchanged.

Indexes are created on the join and group-by keys -- without them the
cross-type affinity self-join scans the fact table twice and dominates runtime.
"""
from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import pandas as pd

from .config import SQL_DIR, SQLITE_DB

QUERY_FILE = SQL_DIR / "analysis.sql"

INDEXES = [
    ("master", "UserId"),
    ("master", "AttractionId"),
    ("master", "AttractionType"),
    ("master", "Continent"),
    ("master", "VisitMode"),
    ("master", "VisitYear"),
]


def build_database(
    master: pd.DataFrame,
    tables: dict[str, pd.DataFrame] | None = None,
    db_path: Path = SQLITE_DB,
) -> Path:
    """Write the master table (and optionally the dimensions) to SQLite."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    with sqlite3.connect(db_path) as con:
        # Nullable Int64 / string dtypes need converting to plain objects
        # before sqlite3 will accept them.
        out = master.copy()
        for c in out.columns:
            if str(out[c].dtype) in ("Int64", "Float64", "string"):
                out[c] = out[c].astype(object).where(out[c].notna(), None)
        out.to_sql("master", con, index=False)

        if tables:
            for name, df in tables.items():
                d = df.copy()
                for c in d.columns:
                    if str(d[c].dtype) in ("Int64", "Float64", "string"):
                        d[c] = d[c].astype(object).where(d[c].notna(), None)
                d.to_sql(name, con, index=False)

        for table, col in INDEXES:
            con.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_{col} ON {table}("{col}")')
        con.commit()

    return db_path


def parse_queries(path: Path = QUERY_FILE) -> dict[str, str]:
    """Split analysis.sql into {name: sql} using the `-- name:` tags."""
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")

    queries: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []

    for line in text.splitlines():
        m = re.match(r"^\s*--\s*name:\s*(\w+)\s*$", line)
        if m:
            if current and buffer:
                queries[current] = "\n".join(buffer).strip()
            current = m.group(1)
            buffer = []
        elif current:
            buffer.append(line)

    if current and buffer:
        queries[current] = "\n".join(buffer).strip()
    return queries


def run_query(name_or_sql: str, db_path: Path = SQLITE_DB) -> pd.DataFrame:
    """Run a named query from analysis.sql, or raw SQL if no such name exists."""
    queries = parse_queries()
    sql = queries.get(name_or_sql, name_or_sql)
    with sqlite3.connect(db_path) as con:
        return pd.read_sql_query(sql, con)


def run_all(db_path: Path = SQLITE_DB, verbose: bool = True) -> dict[str, pd.DataFrame]:
    """Execute every named query; used by the pipeline as a smoke test."""
    results = {}
    for name, sql in parse_queries().items():
        try:
            with sqlite3.connect(db_path) as con:
                results[name] = pd.read_sql_query(sql, con)
            if verbose:
                print(f"    query {name:28s} -> {len(results[name]):>4} rows")
        except Exception as exc:  # surface a bad query rather than failing silently
            print(f"    query {name:28s} -> FAILED: {exc}")
    return results


if __name__ == "__main__":
    print(run_query("top_attractions").to_string(index=False))

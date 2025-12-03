"""
Generic DuckDB helpers extracted from the project specific database module.

The goal is to provide lightweight wrappers for connecting to local files or
MotherDuck remotes together with convenience query helpers.  Nothing here
depends on repository state, so the module can be vendored anywhere a small
DuckDB toolkit is needed.
"""

from __future__ import annotations

from .utils import get_env_var
from pathlib import Path
from typing import Any, List, Optional, Union
from uuid import uuid4
import duckdb
import polars as pl

DataLike = Union[pl.DataFrame, dict[str, Any]]


def _escape(value: str) -> str:
    """Escape single quotes for DuckDB string literals."""
    return value.replace("'", "''")


def _connect_local(path: str, key: Optional[str] = None) -> duckdb.DuckDBPyConnection:
    """
    Create a DuckDB connection to a local database with optional encryption key.
    """
    path_obj = Path(path).expanduser().resolve()
    if key and not path_obj.exists():
        raise FileNotFoundError(f"Database file does not exist: {path_obj}")

    if not key:
        return duckdb.connect(str(path_obj))

    con = duckdb.connect()
    db_literal = _escape(str(path_obj))
    key_literal = _escape(key)
    con.execute(
        f"ATTACH '{db_literal}' AS encrypted_db (ENCRYPTION_KEY '{key_literal}')"
    )
    con.execute("USE encrypted_db")
    return con


def _connect_motherduck(resolved: str) -> duckdb.DuckDBPyConnection:
    """Connect to MotherDuck using the ``md:`` URI syntax."""
    get_env_var("MOTHERDUCK_TOKEN")  # ensures credentials are available
    con = duckdb.connect()
    try:
        try:
            con.execute("LOAD 'motherduck'")
        except duckdb.BinderException:
            con.execute("INSTALL motherduck")
            con.execute("LOAD 'motherduck'")
        con.execute("ATTACH 'md:'")
        con.execute(f"USE {resolved[3:]}")
        return con
    except Exception:
        con.close()
        raise


def _connect_duckdb(
    path: Optional[Union[str, Path]], key: Optional[str] = None
) -> duckdb.DuckDBPyConnection:
    """
    Normalize the provided path and connect to DuckDB or MotherDuck as needed.
    """
    if path is None:
        raise ValueError("Provide path when connecting to DuckDB.")

    if isinstance(path, Path):
        expanded_path = path.expanduser()
        resolved = str(expanded_path)
        if not resolved:
            raise ValueError("Database path cannot be empty.")
        return _connect_local(resolved, key)

    as_str = str(path).strip()
    if not as_str:
        raise ValueError("Database path cannot be empty.")

    if as_str.lower().startswith("md:"):
        if key:
            raise ValueError(
                "Encryption keys are only supported for local DuckDB files."
            )
        return _connect_motherduck(as_str)

    resolved = str(Path(as_str).expanduser())
    return _connect_local(resolved, key)


def query(
    sql: str, path: Optional[Union[str, Path]] = None, key: Optional[str] = None
) -> List[tuple]:
    """Execute a SQL query and return a list of tuples."""
    con = _connect_duckdb(path, key)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def query_df(
    sql: str, path: Optional[Union[str, Path]] = None, key: Optional[str] = None
) -> pl.DataFrame:
    """Execute a SQL query and return the results as a Polars DataFrame."""
    con = _connect_duckdb(path, key)
    try:
        return con.sql(sql).pl()
    finally:
        con.close()


def table(
    table_name: str, path: Optional[Union[str, Path]] = None, key: Optional[str] = None
) -> pl.DataFrame:
    """Fetch every row from a table as a Polars DataFrame."""
    return query_df(f"SELECT * FROM {table_name}", path, key)


def _to_polars(data: DataLike) -> pl.DataFrame:
    """Normalize insert input to a Polars DataFrame."""
    # 1. Convert input to a Polars DataFrame
    if isinstance(data, pl.DataFrame):
        df = data

    elif isinstance(data, dict):
        if not data:
            raise ValueError("Empty data dictionary.")

        # Normalize values to lists and enforce consistent row count
        normalized: dict[str, list[Any]] = {}
        for key, value in data.items():
            if isinstance(value, (list, tuple)):
                normalized[key] = list(value)
            else:
                normalized[key] = [value]

        lengths = {len(v) for v in normalized.values()}
        if len(lengths) != 1:
            raise ValueError("All columns must share the same number of rows.")

        df = pl.DataFrame(normalized)

    else:
        raise TypeError("data must be a Polars DataFrame or a dict.")

    return df


def insert(
    table_name: str,
    data: DataLike,
    path: Optional[Union[str, Path]] = None,
    key: Optional[str] = None,
    strict: bool = True,
) -> int:
    """
    Insert rows into an existing DuckDB table using a DataFrame or mapping.

    When ``strict`` is True, validate the input columns against the table schema
    before inserting.
    """

    expected_columns = table(table_name, path, key).columns
    expected_order = list(expected_columns)
    df = _to_polars(data)

    if df.height == 0:
        raise ValueError("No rows to insert.")

    cols = set(df.columns)
    expected = set(expected_order)
    insert_columns: list[str]

    if strict:
        missing = expected - cols
        extra = cols - expected
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if extra:
            raise ValueError(f"Unexpected columns: {sorted(extra)}")
        insert_columns = expected_order
    else:
        # Only insert columns that exist in both the table and the provided data.
        insert_columns = [col for col in expected_order if col in cols]
        if not insert_columns:
            raise ValueError("No columns provided for insert.")

    df = df.select(insert_columns)

    con = _connect_duckdb(path, key)
    tmp_view = f"_vincentpy_insert_{uuid4().hex}"
    try:
        con.register(tmp_view, df)
        try:
            cols_sql = ", ".join(f'"{col}"' for col in insert_columns)
            con.execute(
                f'INSERT INTO "{table_name}" ({cols_sql}) '
                f"SELECT {cols_sql} FROM {tmp_view}"
            )
        finally:
            con.unregister(tmp_view)
    finally:
        con.close()

    return df.height


def clone(
    remote: Union[str, Path],
    local: Union[str, Path],
    key: Optional[str] = None,
    overwrite: bool = False,
) -> Path:
    """
    Clone a DuckDB database to a local path and return the resolved destination.
    """
    local_path = Path(local).expanduser().resolve()
    if local_path.exists():
        if local_path.is_dir():
            raise IsADirectoryError(f"Destination is a directory: {local_path}")
        if not overwrite:
            raise FileExistsError(
                f"Destination already exists: {local_path}. Pass overwrite=True to replace it."
            )
        local_path.unlink()

    local_path.parent.mkdir(parents=True, exist_ok=True)
    created_file = not local_path.exists()

    con = _connect_duckdb(remote, key)
    try:
        local_literal = _escape(str(local_path))
        con.execute(f"ATTACH '{local_literal}' AS local_db")
        con.execute("COPY FROM DATABASE main TO local_db")
    except Exception:
        if created_file and local_path.exists():
            local_path.unlink()
        raise
    finally:
        con.close()

    return local_path


__all__ = [
    "query",
    "query_df",
    "table",
    "insert",
    "clone",
]

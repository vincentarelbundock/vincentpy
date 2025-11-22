"""
Minimal example showing how to query and insert rows in a DuckDB file.
"""

from pathlib import Path
from vincentpy import duckdb
import polars as pl
import random
import string

# --- Bootstrap TPC-H demo database ------------------------------------------
DB = Path("examples/example.duckdb")
if DB.exists():
    DB.unlink()
bootstrap_sql = """
INSTALL tpch;
LOAD tpch;
CALL dbgen(sf=0.01);
"""
duckdb.query(bootstrap_sql, path=DB)

# --- Basic inspection -------------------------------------------------------
print("Tables:", duckdb.query("SHOW TABLES", path=DB))

customers = duckdb.query_df("SELECT * FROM customer LIMIT 2", path=DB)
print(customers)

nation_head = duckdb.table("nation", path=DB).head()
print(nation_head)

# --- Insert sample rows -----------------------------------------------------
count_before = duckdb.table("nation", path=DB).shape[0]

sample = duckdb.table("nation", path=DB).head(3)
token = "".join(random.choices(string.ascii_lowercase, k=8))
sample = sample.with_columns(pl.lit(token).alias("n_comment"))
duckdb.insert(
    "nation",
    sample,
    path=DB,
)

sample = {
    "n_nationkey": 0,
    "n_name": "Hello",
    "n_regionkey": 1,
    "n_comment": "World",
}
duckdb.insert(
    "nation",
    sample,
    path=DB,
)

count_after = duckdb.table("nation", path=DB).shape[0]
assert count_after - count_before == 4, "Insertion failed"

# --- Verify insertion -------------------------------------------------------
new = duckdb.table("nation", path=DB).tail(10)
print(new)

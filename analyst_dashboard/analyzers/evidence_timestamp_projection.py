"""Read-only timestamp accessor for DB-API mapping rows.

Both psycopg2 RealDictRow and sqlite3.Row support item access and keys(),
but only the former consistently offers dict.get(). Never substitute SEC
publication time for a missing ARX observation timestamp.
"""
from typing import Any


def evidence_row_value(row: Any, key: str) -> Any:
    """Return the literal stored value, or None if the column is absent."""
    try:
        return row[key] if key in row.keys() else None
    except (KeyError, TypeError, AttributeError, IndexError):
        return None

import hashlib
import json
import os

import pandas as pd


TEXT_COLUMNS = ("Item", "Value", "Unit", "Category")


def measurement_cache_enabled():
    value = os.environ.get("STRUCTEHR_MEASUREMENT_CACHE", "1").strip().lower()
    return value not in {"0", "false", "no", "off"}


def stable_cache_key(*parts):
    payload = json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _coerce_text_value(value):
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def normalize_measurement_table_for_cache(table):
    if table is None or table.empty:
        return table

    table = table.copy()
    for column in TEXT_COLUMNS:
        if column not in table.columns:
            table[column] = ""
        table[column] = table[column].map(_coerce_text_value)
    return table


def get_or_build_measurement_table(cache_dir, cache_key, build_fn):
    if not measurement_cache_enabled():
        return build_fn()

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.parquet")
    if os.path.exists(cache_path):
        try:
            return pd.read_parquet(cache_path)
        except Exception:
            pass

    table = normalize_measurement_table_for_cache(build_fn())
    tmp_path = f"{cache_path}.{os.getpid()}.tmp.parquet"
    table.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, cache_path)
    return table

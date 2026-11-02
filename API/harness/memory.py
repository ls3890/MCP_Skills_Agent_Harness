"""Responsibility: user preferences in memory.json (fixed keys, atomic writes)."""
import json

from shared.jobs_domain import PREFERENCE_KEYS, apply_pref, data_path, empty_prefs, write_json_atomic

FILE = "memory.json"


def log(msg: str) -> None:
    print(f"[memory.py] {msg}")


def empty() -> dict:
    return empty_prefs()


def load() -> dict:
    path = data_path(FILE)
    prefs = empty()
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        prefs.update({k: v for k, v in stored.items() if k in PREFERENCE_KEYS})
    return prefs


def save(key: str, value) -> object:
    prefs = load()
    stored = apply_pref(prefs, key, value)
    write_json_atomic(data_path(FILE), prefs)
    log(f"saved {key}={stored}")
    return stored

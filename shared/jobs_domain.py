"""Domain code: applications storage and rules. Shared by every agent version."""
import json
import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

DATA_DIR = Path(__file__).parent / "data"
TZ = ZoneInfo("Asia/Jerusalem")

STATUSES = ["saved", "applied", "phone_screen", "interview", "offer", "rejected", "withdrawn"]
CLOSED_STATUSES = ("rejected", "withdrawn")
PREFERENCE_KEYS = ["role", "location", "work_mode", "excluded_company", "min_salary"]


class DomainError(Exception):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def data_path(name: str) -> Path:
    return DATA_DIR / name


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def today() -> str:
    return datetime.now(TZ).strftime("%Y-%m-%d")


def reset_data() -> None:
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def write_json_atomic(path: Path, obj) -> None:
    write_text_atomic(path, json.dumps(obj, ensure_ascii=False, indent=2))


# ---------- preferences: same rules for every agent version, storage is up to the version ----------

def empty_prefs() -> dict:
    return {"role": None, "location": None, "work_mode": None, "excluded_company": [], "min_salary": None}


def validate_pref(key: str, value) -> object:
    """Returns the normalized value or raises DomainError."""
    if key not in PREFERENCE_KEYS:
        raise DomainError("invalid_key", key)
    value = str(value).strip()
    if not value:
        raise DomainError("invalid_input", "empty value")
    if key == "min_salary":
        if not value.isdigit():
            raise DomainError("invalid_input", "min_salary must be a whole number")
        return int(value)
    return value


def apply_pref(prefs: dict, key: str, value) -> object:
    """Validates, applies to prefs in place, returns the stored value."""
    value = validate_pref(key, value)
    if key == "excluded_company":
        if value.lower() not in (c.lower() for c in prefs[key]):
            prefs[key].append(value)
    else:
        prefs[key] = value
    return prefs[key]


def _load() -> dict:
    path = data_path("applications.json")
    if not path.exists():
        return {"next_id": 1, "applications": []}
    return json.loads(path.read_text(encoding="utf-8"))


def _save(store: dict) -> None:
    write_json_atomic(data_path("applications.json"), store)


def all_applications() -> list[dict]:
    return _load()["applications"]


def add_application(company: str, title: str, status: str = "applied") -> dict:
    company, title = (company or "").strip(), (title or "").strip()
    if not company or not title:
        raise DomainError("invalid_input", "company and title are required")
    if status not in STATUSES:
        raise DomainError("invalid_status", status)
    store = _load()
    for app in store["applications"]:
        if app["company"].lower() == company.lower() and app["title"].lower() == title.lower():
            raise DomainError("duplicate", f"id={app['id']}")
    ts = now()
    app = {"id": store["next_id"], "company": company, "title": title, "status": status,
           "created_at": ts, "status_history": [{"status": status, "at": ts}]}
    store["applications"].append(app)
    store["next_id"] += 1
    _save(store)
    return app


def update_status(app_id: int, status: str, confirm_reopen: bool = False) -> tuple[dict, str]:
    if status not in STATUSES:
        raise DomainError("invalid_status", status)
    store = _load()
    app = next((a for a in store["applications"] if a["id"] == app_id), None)
    if app is None:
        raise DomainError("not_found", f"id={app_id}")
    old = app["status"]
    if old in CLOSED_STATUSES and status not in CLOSED_STATUSES and not confirm_reopen:
        raise DomainError("confirm_reopen", f"id={app_id}|from={old}")
    app["status"] = status
    app["status_history"].append({"status": status, "at": now()})
    _save(store)
    return app, old


def find_applications(company: str = "", title: str = "", status: str = "") -> list[dict]:
    if status and status not in STATUSES:
        raise DomainError("invalid_status", status)
    return [a for a in all_applications()
            if company.lower() in a["company"].lower()
            and title.lower() in a["title"].lower()
            and (not status or a["status"] == status)]

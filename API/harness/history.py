"""Responsibility: append-only conversation log in history.jsonl."""
import json
import os

from shared.jobs_domain import data_path, now

FILE = "history.jsonl"
# Message fields kept in the log; tool_calls/tool_call_id pair each tool result with its call.
MESSAGE_FIELDS = ("role", "content", "tool_calls", "tool_call_id")


def log(msg: str) -> None:
    print(f"[history.py] {msg}")


def _append(record: dict) -> None:
    path = data_path(FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:  # append mode only: lines are never rewritten
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def append_message(message: dict) -> None:
    _append({"type": "message", "at": now(), **{k: message[k] for k in MESSAGE_FIELDS if k in message}})
    log(f"appended {message['role']} message")


def append_summary(text: str) -> None:
    _append({"type": "summary", "at": now(), "text": text})
    log("appended summary record")


def read_all() -> list[dict]:
    path = data_path(FILE)
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_view() -> tuple[str | None, list[dict]]:
    """Latest summary (or None) plus the messages recorded after it."""
    records = read_all()
    start, summary = 0, None
    for i, rec in enumerate(records):
        if rec["type"] == "summary":
            start, summary = i + 1, rec["text"]
    messages = [{k: r[k] for k in MESSAGE_FIELDS if k in r} for r in records[start:] if r["type"] == "message"]
    log(f"view: summary={'yes' if summary else 'no'}, messages={len(messages)}")
    return summary, messages

"""Responsibility: build the request context (system messages + last N turns)."""
import json

from API.harness import caching

TURNS_KEPT = 6


def log(msg: str) -> None:
    print(f"[context.py] {msg}")


def is_plain_user(message: dict) -> bool:
    return message["role"] == "user" and isinstance(message["content"], str)


def last_turns(messages: list[dict], n: int = TURNS_KEPT) -> list[dict]:
    """Keeps the last n turns. A turn starts at a plain user text message, so the cut
    can never start at an orphan tool result."""
    starts = [i for i, m in enumerate(messages) if is_plain_user(m)]
    if not starts:
        return []
    return messages[starts[-n] if len(starts) >= n else starts[0]:]


def dynamic_block(today: str, prefs: dict, summary: str | None) -> dict:
    text = f"Today's date: {today}. Timezone: Asia/Jerusalem.\n\n" \
           f"Saved preferences:\n{json.dumps(prefs, ensure_ascii=False)}"
    if summary:
        text += f"\n\nSummary of earlier conversation:\n{summary}"
    return {"role": "system", "content": text}  # after the static prefix: changes between turns


def build(system_prompt: str, today: str, prefs: dict, summary: str | None,
          history_messages: list[dict]) -> tuple[list[dict], list[dict]]:
    """Returns (system messages, full message list for the request)."""
    system = [caching.static_system_block(system_prompt), dynamic_block(today, prefs, summary)]
    turns = last_turns(history_messages)
    log(f"context: {len(turns)} messages from {len(history_messages)} in view, summary={'yes' if summary else 'no'}")
    return system, system + turns

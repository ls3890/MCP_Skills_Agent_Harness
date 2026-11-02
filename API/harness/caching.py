"""Responsibility: keep the request prefix cache-friendly and report whether caching happened.

Groq caches prompt prefixes automatically on models that support it; there are no
cache_control markers. The prefix (tools + static system message) must stay byte-identical
across requests, and everything that changes goes after it."""
import copy


def log(msg: str) -> None:
    print(f"[caching.py] {msg}")


def static_system_block(text: str) -> dict:
    return {"role": "system", "content": text}


def stable_tools(tools: list[dict]) -> list[dict]:
    """Same tools, same order, every request."""
    return copy.deepcopy(tools)


def has_message_markers(messages: list[dict]) -> bool:
    return any("cache_control" in m for m in messages)


def check_usage(usage: dict, model: str) -> bool:
    """Warns when no prompt tokens were served from cache. Uses response.usage only."""
    if usage["cache_read"] == 0:
        log(f"WARNING: no cached prompt tokens for {model}: first request, prefix too short, "
            f"or the model has no prompt caching")
        return False
    log(f"cache ok: read={usage['cache_read']}")
    return True

"""Responsibility: token usage and cost per model call, read from response.usage only."""

# USD per 1M tokens, Groq on-demand prices as remembered on 2026-10-26: NOT VERIFIED.
# Check against https://groq.com/pricing before trusting the cost numbers.
# Groq does not charge for cache writes; cache_write is reported as 0.
PRICES = {
    "openai/gpt-oss-120b": {"input": 0.15, "cache_write": 0.15, "cache_read": 0.075, "output": 0.60},
    "llama-3.3-70b-versatile": {"input": 0.59, "cache_write": 0.59, "cache_read": 0.59, "output": 0.79},
}


def log(msg: str) -> None:
    print(f"[tokens.py] {msg}")


def cost(usage: dict, model: str) -> float:
    p = PRICES.get(model)
    if p is None:
        log(f"no price for {model}; cost recorded as 0")
        return 0.0
    return round((usage["input"] * p["input"] + usage["cache_write"] * p["cache_write"]
                  + usage["cache_read"] * p["cache_read"] + usage["output"] * p["output"]) / 1_000_000, 6)


def record(response, model: str, elapsed: float) -> dict:
    u = response.usage
    details = getattr(u, "prompt_tokens_details", None)
    cached = (getattr(details, "cached_tokens", 0) or 0) if details else 0
    usage = {
        "input": u.prompt_tokens - cached,  # uncached part only
        "cache_read": cached,
        "cache_write": 0,
        "output": u.completion_tokens,
        "elapsed_s": round(elapsed, 3),
    }
    usage["cost_usd"] = cost(usage, model)
    log(f"in={usage['input']} cache_read={usage['cache_read']} out={usage['output']} "
        f"cost=${usage['cost_usd']} time={usage['elapsed_s']}s")
    return usage


def total_input(usage: dict) -> int:
    return usage["input"] + usage["cache_read"] + usage["cache_write"]

"""Runs scenario.json through the Claude Agent SDK, runs checks C1-C8, writes results/sdk.json."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from SDK import agent as agent_module  # noqa: E402
from SDK import tools  # noqa: E402
from shared import jobs_domain  # noqa: E402

ACME = "acme"
GLOBEX = "globex"
PER_CALL_NOTE = ("NOT VERIFIED: token counts are taken only from ResultMessage.usage, which is a per-turn total; "
                 "model_calls counts distinct AssistantMessage.message_id values")


def log(msg: str) -> None:
    print(f"[main.py] {msg}")


def _acme() -> dict | None:
    return next((a for a in jobs_domain.all_applications() if ACME in a["company"].lower()), None)


# ---------- checks: each returns (name, passed, detail) ----------

def c1_acme_applied():
    app = _acme()
    ok = app is not None and app["status"] == "applied"
    return "C1 applications.json has Acme with status applied", ok, f"acme={app}"


def c2_acme_phone_screen():
    app = _acme()
    ok = (app is not None and app["status"] == "phone_screen"
          and [h["status"] for h in app["status_history"]][-1] == "phone_screen")
    return "C2 Acme status is phone_screen", ok, f"acme={app}"


def c3_session2_loaded(loaded_count: int):
    return "C3 session 2 loads applications from disk", loaded_count > 0, f"loaded={loaded_count}"


def c4_recall(reply: str, prior_turns_in_memory: int, resume: str | None, session_id_on_disk: str | None):
    app = _acme()
    recalled = app is not None and app["title"].lower() in reply.lower()
    resumed = resume is not None and resume == session_id_on_disk and prior_turns_in_memory == 0
    return "C4 session 2 recalls the Acme role via resume=session_id from disk", recalled and resumed, \
        f"title={app and app['title']}|resume={resume}|on_disk={session_id_on_disk}|" \
        f"prior_turns={prior_turns_in_memory}|reply={reply[:120]!r}"


def c5_preference_persisted(context: dict | None):
    on_disk = any(GLOBEX in c.lower() for c in tools.load_prefs()["excluded_company"])
    memory_files = [str(f.get("path", f)) for f in (context or {}).get("memoryFiles", [])]
    expected = tools.memory_path().resolve()
    in_context = any(Path(p).resolve() == expected for p in memory_files if p)
    return "C5 preference in CLAUDE.md and CLAUDE.md loaded in session 2 context", on_disk and in_context, \
        f"on_disk={on_disk}|in_context={in_context}|memory_files={memory_files}"


def c6_conflict(apps_before: int, reply: str):
    apps = jobs_domain.all_applications()
    no_new = len(apps) == apps_before and not any(GLOBEX in a["company"].lower() for a in apps)
    flagged = GLOBEX in reply.lower() and "?" in reply
    return "C6 conflict: no new application, reply flags it and asks", no_new and flagged, \
        f"apps {apps_before}->{len(apps)}|flagged={flagged}"


def c7_error_recovery(tool_calls: list[dict]):
    updates = [c for c in tool_calls if c["name"] == "update_status"]
    first_err = next((i for i, c in enumerate(updates) if (c["result"] or "").startswith("ERR|not_found")), None)
    recovered = first_err is not None and updates[first_err]["is_error"] and any(
        (c["result"] or "").startswith("OK|updated") for c in updates[first_err + 1:])
    return "C7 ERR|not_found then recovery in the same turn", recovered, \
        f"update_status results={[c['result'] for c in updates]}"


def c8_list_lines(tool_calls: list[dict]):
    sizes = [len((c["result"] or "").splitlines()) for c in tool_calls if c["name"] == "list_applications"]
    return "C8 every list_applications result <= 20 lines", all(s <= 20 for s in sizes), f"line_counts={sizes}"


# ---------- run ----------

def line_counts(folder: Path) -> dict:
    return {p.name: len(p.read_text(encoding="utf-8").splitlines()) for p in sorted(folder.glob("*.py"))}


def turn_summary(n: int, session: int, turn: dict) -> dict:
    return {"turn": n, "session": session, "user": turn["user"], "reply": turn["reply"],
            "resume": turn["resume"], "session_id": turn["session_id"], "subtype": turn["subtype"],
            "model_calls": turn["model_calls"], "num_turns": turn["num_turns"],
            "tools_called": [c["name"] for c in turn["tool_calls"]],
            "tool_results": [(c["result"] or "").splitlines()[0:1] for c in turn["tool_calls"]],
            **turn["usage"], "total_cost_usd": turn["total_cost_usd"], "elapsed_s": turn["elapsed_s"],
            "duration_ms": turn["duration_ms"], "model_usage": turn["model_usage"]}


def main() -> None:
    if "ANTHROPIC_API_KEY" in os.environ:  # S2: run on the Claude Code login only; the value is never read
        log("S2 FAIL: ANTHROPIC_API_KEY is set in the environment. Not running.")
        sys.exit(2)

    jobs_domain.reset_data()
    log(f"reset {jobs_domain.DATA_DIR}")
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))

    checks, turns, all_tool_calls, context_seen, n = [], [], [], None, 0
    for s_index, session in enumerate(scenario["sessions"], start=1):
        agent = agent_module.Agent()  # new instance: state comes from disk only
        log(f"session {s_index}: {len(session['turns'])} turns")
        if s_index == 2:
            checks.append(c3_session2_loaded(agent.loaded_applications))
        for user_text in session["turns"]:
            n += 1
            prior_turns = len(agent.turns)
            apps_before = len(jobs_domain.all_applications())
            log(f"turn {n}: {user_text}")
            turn = agent.ask(user_text, inspect_context=(n == 6))
            log(f"reply: {turn['reply']}")
            all_tool_calls += turn["tool_calls"]
            turns.append(turn_summary(n, s_index, turn))
            if n == 2:
                checks.append(c1_acme_applied())
            if n == 5:
                checks += [c2_acme_phone_screen(), c7_error_recovery(turn["tool_calls"])]
            if n == 6:
                context_seen = turn["context"]
                checks += [c4_recall(turn["reply"], prior_turns, turn["resume"], agent.loaded_session_id),
                           c5_preference_persisted(context_seen)]
            if n == 8:
                checks.append(c6_conflict(apps_before, turn["reply"]))
    checks.append(c8_list_lines(all_tool_calls))

    for name, passed, detail in checks:
        log(f"{'PASS' if passed else 'FAIL'} {name} ({detail})")
    results = {
        "model": agent_module.MODEL, "effort": agent_module.EFFORT, "max_turns": agent_module.MAX_TURNS,
        "per_call_breakdown": PER_CALL_NOTE,
        "turns": turns,
        "totals": {k: round(sum(t[k] or 0 for t in turns), 6)
                   for k in ("input", "cache_read", "cache_write", "output", "total_cost_usd", "elapsed_s")},
        "checks": [{"name": c[0], "passed": c[1], "detail": c[2]} for c in checks],
        "session2_context": context_seen and {
            "categories": context_seen.get("categories"), "memoryFiles": context_seen.get("memoryFiles"),
            "totalTokens": context_seen.get("totalTokens")},
        "sdk_line_counts": line_counts(ROOT / "SDK"),
        "api_harness_line_counts": line_counts(ROOT / "API" / "harness"),
    }
    out = ROOT / "results" / "sdk.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=lambda o: getattr(o, "__dict__", str(o))),
                   encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

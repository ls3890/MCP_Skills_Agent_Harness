"""Runs scenario.json against the live Groq API, runs checks C1-C8, writes results/api.json."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from API import agent as agent_module  # noqa: E402
from shared import jobs_domain  # noqa: E402

ACME = "acme"
GLOBEX = "globex"


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


def c4_recall(reply: str, prior_turns_in_memory: int):
    app = _acme()
    ok = app is not None and app["title"].lower() in reply.lower() and prior_turns_in_memory == 0
    return "C4 session 2 recalls the Acme role with no in-memory history", ok, \
        f"title={app and app['title']}|prior_turns={prior_turns_in_memory}|reply={reply[:120]!r}"


def c5_preference_persisted(session2_system: list[dict] | None):
    path = jobs_domain.data_path("memory.json")
    prefs = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    on_disk = any(GLOBEX in c.lower() for c in prefs.get("excluded_company", []))
    in_context = bool(session2_system) and GLOBEX in session2_system[-1]["content"].lower()
    return "C5 preference persisted and present in session 2 context", on_disk and in_context, \
        f"on_disk={on_disk}|in_context={in_context}"


def c6_conflict(apps_before: int, reply: str):
    apps = jobs_domain.all_applications()
    no_new = len(apps) == apps_before and not any(GLOBEX in a["company"].lower() for a in apps)
    flagged = GLOBEX in reply.lower() and "?" in reply
    return "C6 conflict: no new application, reply flags it and asks", no_new and flagged, \
        f"apps {apps_before}->{len(apps)}|flagged={flagged}"


def c7_error_recovery(tool_calls: list[dict]):
    updates = [c for c in tool_calls if c["name"] == "update_status"]
    first_err = next((i for i, c in enumerate(updates) if c["result"].startswith("ERR|not_found")), None)
    recovered = first_err is not None and any(
        c["result"].startswith("OK|updated") for c in updates[first_err + 1:])
    return "C7 ERR|not_found then recovery in the same turn", recovered, \
        f"update_status results={[c['result'] for c in updates]}"


def c8_list_lines(tool_calls: list[dict]):
    sizes = [len(c["result"].splitlines()) for c in tool_calls if c["name"] == "list_applications"]
    return "C8 every list_applications result <= 20 lines", all(s <= 20 for s in sizes), f"line_counts={sizes}"


# ---------- run ----------

def harness_line_counts() -> dict:
    return {p.name: len(p.read_text(encoding="utf-8").splitlines())
            for p in sorted((ROOT / "API" / "harness").glob("*.py"))}


def turn_summary(n: int, session: int, turn: dict) -> dict:
    return {"turn": n, "session": session, "user": turn["user"], "reply": turn["reply"],
            "model_calls": turn["model_calls"], "tools_called": [c["name"] for c in turn["tool_calls"]],
            "compacted": turn["compacted"], "usage": turn["usage"],
            "totals": {k: round(sum(u[k] for u in turn["usage"]), 6)
                       for k in ("input", "cache_read", "cache_write", "output", "cost_usd", "elapsed_s")}}


def failed_turn(user_text: str, error: Exception) -> dict:
    return {"user": user_text, "reply": f"[API error] {type(error).__name__}: {error}", "model_calls": 0,
            "tool_calls": [], "capped": False, "compacted": False, "usage": []}


def main() -> None:
    import groq
    import truststore
    from dotenv import load_dotenv

    # Use the Windows certificate store so TLS works behind an inspecting proxy/antivirus.
    truststore.inject_into_ssl()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        sys.exit("[main.py] GROQ_API_KEY is not set. Put it in .env at the project root.")

    jobs_domain.reset_data()
    log(f"reset {jobs_domain.DATA_DIR}")
    scenario = json.loads((ROOT / "scenario.json").read_text(encoding="utf-8"))
    client = groq.Groq()
    log(f"model={agent_module.MODEL}")

    checks, turns, all_tool_calls, n = [], [], [], 0
    for s_index, session in enumerate(scenario["sessions"], start=1):
        agent = agent_module.Agent(client)  # session 2 starts from disk only
        log(f"session {s_index}: {len(session['turns'])} turns")
        if s_index == 2:
            checks.append(c3_session2_loaded(agent.loaded_applications))
        for user_text in session["turns"]:
            n += 1
            prior_turns = len(agent.turns)
            apps_before = len(jobs_domain.all_applications())
            log(f"turn {n}: {user_text}")
            try:
                turn = agent.ask(user_text)
            except Exception as e:  # API errors: record the turn and keep going
                log(f"turn {n} failed: {type(e).__name__}: {e}")
                turn = failed_turn(user_text, e)
            print(turn["reply"])
            all_tool_calls += turn["tool_calls"]
            turns.append(turn_summary(n, s_index, turn))
            if n == 2:
                checks.append(c1_acme_applied())
            if n == 5:
                checks += [c2_acme_phone_screen(), c7_error_recovery(turn["tool_calls"])]
            if n == 6:
                checks += [c4_recall(turn["reply"], prior_turns), c5_preference_persisted(agent.last_system)]
            if n == 8:
                checks.append(c6_conflict(apps_before, turn["reply"]))
    checks.append(c8_list_lines(all_tool_calls))

    for name, passed, detail in checks:
        log(f"{'PASS' if passed else 'FAIL'} {name} ({detail})")
    results = {
        "model": agent_module.MODEL, "effort": agent_module.EFFORT, "turns": turns,
        "checks": [{"name": c[0], "passed": c[1], "detail": c[2]} for c in checks],
        "harness_line_counts": harness_line_counts(),
    }
    out = ROOT / "results" / "api.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"wrote {out}")


if __name__ == "__main__":
    main()

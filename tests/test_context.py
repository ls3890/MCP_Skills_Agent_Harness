import pytest

from fake_client import FakeClient, text_response, tool_response
from API import agent as agent_module
from API import main
from API.harness import caching, compaction, context, history, tokens
from shared import jobs_domain


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_domain, "DATA_DIR", tmp_path)


def turn(i, with_tool=True):
    msgs = [{"role": "user", "content": f"q{i}"}]
    if with_tool:
        msgs += [{"role": "assistant", "content": "", "tool_calls": [
                     {"id": f"t{i}", "type": "function", "function": {"name": "x", "arguments": "{}"}}]},
                 {"role": "tool", "tool_call_id": f"t{i}", "content": "OK|"}]
    msgs.append({"role": "assistant", "content": f"a{i}"})
    return msgs


def test_cut_never_starts_with_tool_result():
    messages = [m for i in range(10) for m in turn(i, with_tool=i % 2 == 0)]
    for start in range(len(messages)):  # every possible window of a long history
        kept = context.last_turns(messages[start:])
        if kept:
            assert context.is_plain_user(kept[0])
    kept = context.last_turns(messages)
    assert sum(context.is_plain_user(m) for m in kept) == 6 and kept[0]["content"] == "q4"


def test_static_first_dynamic_second():
    system, messages = context.build("STATIC", "2026-10-26", {"role": "DE"}, "earlier summary", turn(1))
    assert messages[:2] == system and system[0] == {"role": "system", "content": "STATIC"}
    assert "2026-10-26" in system[1]["content"] and "earlier summary" in system[1]["content"]
    assert messages[2] == {"role": "user", "content": "q1"}


def test_prefix_is_stable_and_has_no_markers():
    client = FakeClient([text_response("hi"), text_response("hi again", cached=900)])
    agent = agent_module.Agent(client)
    agent.ask("hello")
    agent.ask("hello again")
    first, second = client.calls
    assert first["tools"] == second["tools"] and len(first["tools"]) == 5
    assert first["messages"][0] == second["messages"][0]  # static system message unchanged
    assert not caching.has_message_markers(second["messages"])
    assert all("cache_control" not in t for t in second["tools"])


def test_cache_warning_when_nothing_cached(capsys):
    assert caching.check_usage({"cache_read": 0, "cache_write": 0}, "m") is False
    assert "WARNING" in capsys.readouterr().out
    assert caching.check_usage({"cache_read": 800, "cache_write": 0}, "m") is True


def test_compaction_threshold():
    assert not compaction.should_compact(13_999, 20_000, 0.7)
    assert not compaction.should_compact(14_000, 20_000, 0.7)
    assert compaction.should_compact(14_001, 20_000, 0.7)


def test_compaction_after_turn_appends_summary():
    # turn input = 15000 > 0.7 * 20000, so exactly one summary call follows the turn
    client = FakeClient([
        tool_response(("t1", "list_applications", {}), input=15_000),
        text_response("done", input=15_000),
        text_response("SUMMARY TEXT", input=500),
    ])
    turn = agent_module.Agent(client).ask("show my applications")
    summary_call = client.calls[2]
    assert "tools" not in summary_call and summary_call["max_completion_tokens"] <= 2000
    assert turn["compacted"] and turn["model_calls"] == 3
    records = history.read_all()
    assert [r["type"] for r in records] == ["message"] * 4 + ["summary"]
    assert history.load_view() == ("SUMMARY TEXT", [])


def test_no_compaction_below_threshold():
    client = FakeClient([text_response("hi", input=1_000)])
    turn = agent_module.Agent(client).ask("hi")
    assert not turn["compacted"] and len(client.calls) == 1


def test_cost_calculation():
    model = "openai/gpt-oss-120b"
    p = tokens.PRICES[model]
    usage = {"input": 1_000_000, "cache_write": 0, "cache_read": 1_000_000, "output": 1_000_000}
    assert tokens.cost(usage, model) == pytest.approx(p["input"] + p["cache_read"] + p["output"])
    assert tokens.cost({"input": 2000, "cache_write": 0, "cache_read": 0, "output": 500}, model) == \
        pytest.approx((2000 * p["input"] + 500 * p["output"]) / 1_000_000)
    assert tokens.cost(usage, "unknown-model") == 0.0


def test_usage_read_from_response_only():
    resp = text_response("x", input=16, output=7, cached=5)
    u = tokens.record(resp, "openai/gpt-oss-120b", 0.5)
    assert (u["input"], u["output"], u["cache_read"], u["cache_write"]) == (11, 7, 5, 0)


def test_session2_context_comes_from_disk():
    client = FakeClient([tool_response(("t1", "save_preference", {"key": "excluded_company", "value": "Globex"})),
                         text_response("saved"), text_response("you excluded Globex")])
    agent_module.Agent(client).ask("exclude Globex")
    second = agent_module.Agent(client)
    assert second.turns == []
    second.ask("what did I exclude?")
    sent = client.calls[2]["messages"]
    assert "Globex" in sent[1]["content"]
    assert sent[2] == {"role": "user", "content": "exclude Globex"}  # history reloaded from history.jsonl


# ---------- checks must be able to fail ----------

def test_checks_fail_when_behavior_is_broken():
    assert main.c1_acme_applied()[1] is False  # no Acme row
    jobs_domain.add_application("Acme Analytics", "Senior Data Engineer")
    assert main.c1_acme_applied()[1] is True
    assert main.c2_acme_phone_screen()[1] is False
    assert main.c3_session2_loaded(0)[1] is False
    assert main.c4_recall("Senior Data Engineer", prior_turns_in_memory=2)[1] is False
    assert main.c4_recall("no idea", prior_turns_in_memory=0)[1] is False
    assert main.c5_preference_persisted(None)[1] is False
    jobs_domain.add_application("Globex", "Data Engineer")
    assert main.c6_conflict(1, "Added Globex?")[1] is False  # a row was created
    assert main.c7_error_recovery([{"name": "update_status", "result": "ERR|not_found|id=42"}])[1] is False
    big = "\n".join(["OK|count=30"] + ["x"] * 20)
    assert main.c8_list_lines([{"name": "list_applications", "result": big}])[1] is False


def test_checks_pass_on_correct_behavior():
    jobs_domain.add_application("Acme Analytics", "Senior Data Engineer")
    jobs_domain.update_status(1, "phone_screen")
    assert main.c2_acme_phone_screen()[1]
    assert main.c4_recall("You applied for Senior Data Engineer.", 0)[1]
    assert main.c6_conflict(1, "Globex is on your excluded list. Add it anyway?")[1]
    calls = [{"name": "update_status", "result": "ERR|not_found|id=42"},
             {"name": "list_applications", "result": "OK|count=1|shown=1\n1|Acme|SDE|applied"},
             {"name": "update_status", "result": "OK|updated|id=1|applied->phone_screen"}]
    assert main.c7_error_recovery(calls)[1] and main.c8_list_lines(calls)[1]
    from shared.jobs_domain import data_path
    import json
    data_path("memory.json").write_text(json.dumps({"excluded_company": ["Globex"]}), encoding="utf-8")
    assert main.c5_preference_persisted([{"role": "system", "content": "prefs: Globex"}])[1]

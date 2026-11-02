import re
from pathlib import Path

import pytest

from fake_client import ROOT, FakeSDKClient
from SDK import agent as sdk_agent
from SDK import main as sdk_main
from SDK import tools as sdk_tools
from shared import jobs_domain

SDK_FILES = sorted((ROOT / "SDK").glob("*.py"))
REPLACES = {"loop.py", "tools.py", "history.py", "memory.py", "context.py", "compaction.py", "caching.py", "tokens.py"}


@pytest.fixture(autouse=True)
def fake_sdk(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_domain, "DATA_DIR", tmp_path)
    monkeypatch.setattr(sdk_agent, "ClaudeSDKClient", FakeSDKClient)
    FakeSDKClient.instances, FakeSDKClient.scripts = [], []
    FakeSDKClient.context = {"memoryFiles": [], "categories": []}
    return tmp_path


def source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------- options ----------

def test_options_wiring():
    opts = sdk_agent.build_options(resume=None)
    assert opts.strict_mcp_config is True
    assert opts.allowed_tools and all(t.startswith("mcp__jobs__") for t in opts.allowed_tools)
    assert sorted(opts.allowed_tools) == sorted(f"mcp__jobs__{n}" for n in
                                                ["add_application", "update_status", "list_applications",
                                                 "save_preference", "search_jobs"])
    assert list(opts.mcp_servers) == ["jobs"]
    assert opts.tools == []  # no built-in tools
    assert opts.setting_sources == ["project"]
    assert Path(opts.cwd) == jobs_domain.DATA_DIR
    assert opts.model == sdk_agent.MODEL and opts.effort == sdk_agent.EFFORT and opts.max_turns == sdk_agent.MAX_TURNS


def test_system_prompt_is_set_with_today_and_has_no_preferences_placeholder():
    assert "{preferences}" not in sdk_agent.SYSTEM_PROMPT
    assert re.findall(r"\{(\w*)\}", sdk_agent.SYSTEM_PROMPT) == ["today"]
    prompt = sdk_agent.build_options(None).system_prompt
    assert isinstance(prompt, str) and jobs_domain.today() in prompt and "{" not in prompt


def test_tool_names_in_prompt_match_server_tools():
    for name in sdk_tools.HANDLERS:
        assert f"- {name}:" in sdk_agent.SYSTEM_PROMPT


# ---------- source rules ----------

def test_no_api_key_read_or_passed():
    for path in SDK_FILES:
        text = source(path)
        assert not re.search(r"\bapi_key\b", text), path  # no api_key= passed to anything
        assert not re.search(r"(getenv|environ\.get|environ\[)\s*\(?\s*['\"]ANTHROPIC_API_KEY", text), path
        # The only mention allowed is the presence guard in main.py, which never reads the value.
        mentions = [line for line in text.splitlines() if "ANTHROPIC_API_KEY" in line]
        assert all('"ANTHROPIC_API_KEY" in os.environ' in m or "log(" in m for m in mentions), mentions


def test_sdk_does_not_reimplement_the_harness():
    for path in SDK_FILES:
        text = source(path)
        assert "memory.json" not in text, path
        assert "cache_control" not in text, path
        assert "stop_reason" not in text, path
        assert "API.harness" not in text and "from API" not in text, path
        assert not re.search(r"^\s*while\b", text, re.M), path  # the SDK runs the loop


def test_replaces_markers_use_known_names():
    found = set()
    for path in SDK_FILES:
        for name in re.findall(r"# replaces: (\S+)", source(path)):
            assert name in REPLACES, (path, name)
            found.add(name)
    assert {"loop.py", "tools.py", "history.py", "memory.py", "context.py", "compaction.py",
            "caching.py", "tokens.py"} <= found


def test_every_print_goes_through_a_prefixed_log():
    for path in SDK_FILES:
        prints = [line.strip() for line in source(path).splitlines() if "print(" in line]
        assert prints == [f'print(f"[{path.name}] {{msg}}")'], (path, prints)


# ---------- a two-session run through the fake client ----------

def test_two_sessions_resume_from_disk_and_record_turn_data(capsys):
    FakeSDKClient.scripts = [
        [("tool", "add_application", {"company": "Acme Analytics", "title": "Senior Data Engineer"}),
         ("text", "Recorded.")],
        [("tool", "update_status", {"id": 42, "status": "phone_screen"}),
         ("tool", "list_applications", {"company": "Acme"}),
         ("tool", "update_status", {"id": 1, "status": "phone_screen"}),
         ("text", "Updated id 1.")],
        [("text", "You applied for Senior Data Engineer at Acme Analytics; status phone_screen.")],
    ]
    first = sdk_agent.Agent()
    t1 = first.ask("I applied to Acme")
    assert t1["resume"] is None and sdk_agent.load_session_id() == "sess-1"
    t2 = first.ask("Update id 42")
    assert t2["resume"] == "sess-1"
    assert sdk_main.c7_error_recovery(t2["tool_calls"])[1]
    assert t2["model_calls"] == 4 and [c["name"] for c in t2["tool_calls"]] == \
        ["update_status", "list_applications", "update_status"]
    assert t2["usage"] == {"input": 100, "cache_read": 0, "cache_write": 0, "output": 20}
    assert t2["total_cost_usd"] == 0.01 and t2["elapsed_s"] >= 0

    second = sdk_agent.Agent()  # new instance, nothing carried over in memory
    assert second.turns == [] and second.loaded_applications == 1 and second.loaded_session_id == "sess-1"
    t3 = second.ask("Which role at Acme?")
    assert FakeSDKClient.instances[-1].options.resume == "sess-1"
    assert sdk_main.c4_recall(t3["reply"], 0, t3["resume"], second.loaded_session_id)[1]
    assert capsys.readouterr().out.count("[agent.py] turn done") == 3


def test_c5_needs_claude_md_on_disk_and_in_loaded_context():
    sdk_tools.save_pref("excluded_company", "Globex")
    in_ctx = {"memoryFiles": [{"path": str(sdk_tools.memory_path())}]}
    assert sdk_main.c5_preference_persisted(in_ctx)[1]
    assert sdk_main.c5_preference_persisted({"memoryFiles": []})[1] is False
    assert sdk_main.c5_preference_persisted(None)[1] is False


# ---------- the checks can fail ----------

def test_checks_fail_when_behavior_is_broken():
    assert sdk_main.c1_acme_applied()[1] is False  # no Acme row
    jobs_domain.add_application("Acme Analytics", "Senior Data Engineer")
    assert sdk_main.c1_acme_applied()[1] is True
    assert sdk_main.c2_acme_phone_screen()[1] is False
    assert sdk_main.c3_session2_loaded(0)[1] is False
    assert sdk_main.c4_recall("Senior Data Engineer", 2, "s1", "s1")[1] is False      # in-memory history
    assert sdk_main.c4_recall("Senior Data Engineer", 0, None, "s1")[1] is False      # not resumed
    assert sdk_main.c4_recall("no idea", 0, "s1", "s1")[1] is False                   # not recalled
    assert sdk_main.c5_preference_persisted({"memoryFiles": [{"path": str(sdk_tools.memory_path())}]})[1] is False
    jobs_domain.add_application("Globex", "Data Engineer")
    assert sdk_main.c6_conflict(1, "Added Globex?")[1] is False  # a row was created
    err = {"name": "update_status", "result": "ERR|not_found|id=42", "is_error": True}
    assert sdk_main.c7_error_recovery([err])[1] is False
    assert sdk_main.c7_error_recovery([{**err, "is_error": False},
                                       {"name": "update_status", "result": "OK|updated|id=1", "is_error": False}])[1] is False
    big = "\n".join(["OK|count=30"] + ["x"] * 20)
    assert sdk_main.c8_list_lines([{"name": "list_applications", "result": big}])[1] is False


def test_checks_pass_on_correct_behavior():
    jobs_domain.add_application("Acme Analytics", "Senior Data Engineer")
    jobs_domain.update_status(1, "phone_screen")
    assert sdk_main.c2_acme_phone_screen()[1]
    assert sdk_main.c3_session2_loaded(1)[1]
    assert sdk_main.c6_conflict(1, "Globex is on your excluded list. Add it anyway?")[1]
    calls = [{"name": "update_status", "result": "ERR|not_found|id=42", "is_error": True},
             {"name": "list_applications", "result": "OK|count=1|shown=1\n1|Acme|SDE|applied", "is_error": False},
             {"name": "update_status", "result": "OK|updated|id=1|applied->phone_screen", "is_error": False}]
    assert sdk_main.c7_error_recovery(calls)[1] and sdk_main.c8_list_lines(calls)[1]

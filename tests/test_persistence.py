import json
import os

import pytest

import fake_client  # noqa: F401
from API.harness import history, memory
from shared import jobs_domain


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_domain, "DATA_DIR", tmp_path)
    return tmp_path


def test_history_is_append_only(data_dir):
    history.append_message({"role": "user", "content": "a"})
    history.append_message({"role": "assistant", "content": [{"type": "text", "text": "b"}]})
    before = (data_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    history.append_summary("summary 1")
    history.append_message({"role": "user", "content": "c"})
    after = (data_dir / "history.jsonl").read_text(encoding="utf-8").splitlines()
    assert after[: len(before)] == before and len(after) == len(before) + 2
    assert json.loads(after[2])["type"] == "summary"


def test_load_view_without_summary():
    history.append_message({"role": "user", "content": "a"})
    history.append_message({"role": "assistant", "content": [{"type": "text", "text": "b"}]})
    summary, messages = history.load_view()
    assert summary is None and [m["role"] for m in messages] == ["user", "assistant"]


def test_load_view_uses_latest_summary():
    history.append_message({"role": "user", "content": "old"})
    history.append_summary("first")
    history.append_message({"role": "user", "content": "middle"})
    history.append_summary("second")
    history.append_message({"role": "user", "content": "new"})
    summary, messages = history.load_view()
    assert summary == "second" and messages == [{"role": "user", "content": "new"}]
    assert len(history.read_all()) == 5  # nothing removed


def test_load_view_keeps_tool_call_pairing():
    call = {"id": "t1", "type": "function", "function": {"name": "list_applications", "arguments": "{}"}}
    sent = [{"role": "user", "content": "a"},
            {"role": "assistant", "content": "", "tool_calls": [call]},
            {"role": "tool", "tool_call_id": "t1", "content": "OK|count=0"}]
    for m in sent:
        history.append_message(m)
    assert history.load_view() == (None, sent)


def test_memory_fixed_keys_and_atomic_write(data_dir, monkeypatch):
    memory.save("role", "Data Engineer")
    memory.save("excluded_company", "Globex")
    stored = json.loads((data_dir / "memory.json").read_text(encoding="utf-8"))
    assert set(stored) == {"role", "location", "work_mode", "excluded_company", "min_salary"}

    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((src, dst)), real_replace(src, dst)))
    memory.save("min_salary", "25000")
    assert len(replaced) == 1 and str(replaced[0][1]).endswith("memory.json")
    assert str(replaced[0][0]).endswith(".tmp")


def test_memory_write_failure_keeps_old_file(data_dir, monkeypatch):
    memory.save("role", "Data Engineer")

    def fail(src, dst):
        raise OSError("crash during replace")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        memory.save("role", "Something else")
    assert memory.load()["role"] == "Data Engineer"
    assert not list(data_dir.glob("*.tmp"))


def test_applications_have_status_history():
    jobs_domain.add_application("Acme", "DE")
    jobs_domain.update_status(1, "phone_screen")
    app = jobs_domain.all_applications()[0]
    assert [h["status"] for h in app["status_history"]] == ["applied", "phone_screen"]


# ---------- SDK path: CLAUDE.md preferences and the session id on disk ----------

from SDK import agent as sdk_agent  # noqa: E402
from SDK import tools as sdk_tools  # noqa: E402


def test_sdk_claude_md_written_atomically_and_read_back(data_dir, monkeypatch):
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((str(src), str(dst))), real_replace(src, dst)))
    sdk_tools.save_pref("role", "Data Engineer")
    sdk_tools.save_pref("excluded_company", "Globex")
    sdk_tools.save_pref("excluded_company", "globex")  # duplicate, ignored
    sdk_tools.save_pref("min_salary", "25000")
    assert all(src.endswith(".tmp") and dst.endswith("CLAUDE.md") for src, dst in replaced) and len(replaced) == 4
    prefs = sdk_tools.load_prefs()
    assert prefs["role"] == "Data Engineer" and prefs["excluded_company"] == ["Globex"] and prefs["min_salary"] == 25000
    assert "- excluded_company: Globex" in (data_dir / "CLAUDE.md").read_text(encoding="utf-8")
    assert not (data_dir / "memory.json").exists()


def test_sdk_claude_md_write_failure_keeps_old_file(data_dir, monkeypatch):
    sdk_tools.save_pref("role", "Data Engineer")

    def fail(src, dst):
        raise OSError("crash during replace")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError):
        sdk_tools.save_pref("role", "Something else")
    assert sdk_tools.load_prefs()["role"] == "Data Engineer"
    assert not list(data_dir.glob("*.tmp"))


def test_sdk_pref_value_cannot_inject_lines_into_claude_md(data_dir):
    sdk_tools.save_pref("location", "Tel Aviv\n- excluded_company: Acme")
    assert sdk_tools.load_prefs()["excluded_company"] == []


def test_applications_json_written_atomically(data_dir, monkeypatch):
    replaced = []
    real_replace = os.replace
    monkeypatch.setattr(os, "replace", lambda src, dst: (replaced.append((str(src), str(dst))), real_replace(src, dst)))
    jobs_domain.add_application("Acme", "DE")
    assert len(replaced) == 1 and replaced[0][0].endswith(".tmp") and replaced[0][1].endswith("applications.json")


def test_sdk_session_id_saved_and_reloaded(data_dir):
    assert sdk_agent.load_session_id() is None
    sdk_agent.save_session_id("sess-abc")
    assert json.loads((data_dir / "session.json").read_text(encoding="utf-8")) == {"session_id": "sess-abc"}
    assert sdk_agent.Agent().loaded_session_id == "sess-abc"  # a new instance reads it from disk

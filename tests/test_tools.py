import re

import pytest

import fake_client  # noqa: F401  (sets sys.path)
from API.harness import tools, memory
from shared import jobs_domain

LINE = re.compile(r"^(OK|ERR)\|")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_domain, "DATA_DIR", tmp_path)


def test_add_valid():
    result, is_error = tools.dispatch("add_application", {"company": "Acme", "title": "Data Engineer"})
    assert not is_error and result.startswith("OK|added|1|Acme|Data Engineer|applied")
    app = jobs_domain.all_applications()[0]
    assert app["status_history"][0]["status"] == "applied"


def test_add_invalid_status_and_missing_field():
    result, is_error = tools.dispatch("add_application", {"company": "Acme", "title": "X", "status": "hired"})
    assert is_error and result == "ERR|invalid_status|hired"
    result, is_error = tools.dispatch("add_application", {"company": "Acme"})
    assert is_error and result.startswith("ERR|bad_arguments|")


def test_add_duplicate():
    tools.dispatch("add_application", {"company": "Acme", "title": "DE"})
    result, is_error = tools.dispatch("add_application", {"company": "acme", "title": "de"})
    assert is_error and result == "ERR|duplicate|id=1"


def test_add_excluded_company_is_conflict():
    memory.save("excluded_company", "Globex")
    result, is_error = tools.dispatch("add_application", {"company": "Globex", "title": "DE"})
    assert is_error and result.startswith("ERR|conflict|")
    assert jobs_domain.all_applications() == []


def test_update_unknown_id():
    result, is_error = tools.dispatch("update_status", {"id": 99, "status": "offer"})
    assert is_error and result == "ERR|not_found|id=99"


def test_update_and_reopen_needs_confirmation():
    tools.dispatch("add_application", {"company": "Acme", "title": "DE"})
    assert tools.dispatch("update_status", {"id": 1, "status": "rejected"})[0] == "OK|updated|id=1|applied->rejected"
    result, is_error = tools.dispatch("update_status", {"id": 1, "status": "interview"})
    assert is_error and result.startswith("ERR|confirm_reopen|id=1|from=rejected")
    result, is_error = tools.dispatch("update_status", {"id": 1, "status": "interview", "confirm_reopen": True})
    assert not is_error
    history = [h["status"] for h in jobs_domain.all_applications()[0]["status_history"]]
    assert history == ["applied", "rejected", "interview"]


def test_list_is_capped_and_truncated():
    for i in range(30):
        tools.dispatch("add_application", {"company": "C" * 40 + str(i), "title": "T" * 60 + str(i)})
    result, is_error = tools.dispatch("list_applications", {})
    lines = result.splitlines()
    assert not is_error and len(lines) <= 20
    assert lines[0].startswith("OK|count=30|shown=19")
    _id, company, title, _status = lines[1].split("|")
    assert len(company) <= 25 and len(title) <= 40


def test_save_preference_rejects_unknown_key():
    result, is_error = tools.dispatch("save_preference", {"key": "favorite_color", "value": "blue"})
    assert is_error and result == "ERR|invalid_key|favorite_color"


def test_search_jobs_compact_format():
    result, is_error = tools.dispatch("search_jobs", {"query": "data", "location": "Tel Aviv"})
    assert not is_error
    assert result.startswith("OK|count=")
    assert "{" not in result  # no JSON
    assert "IGNORE" not in result  # posting descriptions are not passed to the model


def test_dispatch_unknown_tool_and_internal_error(monkeypatch):
    assert tools.dispatch("nope", {}) == ("ERR|unknown_tool|nope", True)

    def boom(**_):
        raise RuntimeError("disk on fire")

    monkeypatch.setitem(tools.HANDLERS, "list_applications", boom)
    assert tools.dispatch("list_applications", {}) == ("ERR|internal|RuntimeError", True)


def test_every_result_has_ok_or_err_prefix():
    calls = [("add_application", {"company": "A", "title": "B"}), ("list_applications", {"status": "bad"}),
             ("update_status", {"id": "x", "status": "applied"}), ("save_preference", {"key": "role", "value": "DE"}),
             ("search_jobs", {"query": "zzz"})]
    for name, args in calls:
        result, _ = tools.dispatch(name, args)
        assert LINE.match(result), (name, result)


# ---------- SDK path (SDK/tools.py): same contract through the @tool handlers ----------

import asyncio  # noqa: E402

from SDK import tools as sdk_tools  # noqa: E402


def call(name, args):
    out = asyncio.run(sdk_tools.HANDLERS[name](args))
    return out["content"][0]["text"], out["is_error"]


def test_sdk_add_valid():
    result, is_error = call("add_application", {"company": "Acme", "title": "Data Engineer"})
    assert not is_error and result.startswith("OK|added|1|Acme|Data Engineer|applied")
    assert jobs_domain.all_applications()[0]["status_history"][0]["status"] == "applied"


def test_sdk_add_invalid_input():
    result, is_error = call("add_application", {"company": "Acme", "title": "X", "status": "hired"})
    assert is_error and result == "ERR|invalid_status|hired"
    result, is_error = call("add_application", {"company": "Acme"})
    assert is_error and result.startswith("ERR|bad_arguments|")
    result, is_error = call("add_application", {"company": " ", "title": "X"})
    assert is_error and result.startswith("ERR|invalid_input|")


def test_sdk_add_duplicate():
    call("add_application", {"company": "Acme", "title": "DE"})
    assert call("add_application", {"company": "acme", "title": "de"}) == ("ERR|duplicate|id=1", True)


def test_sdk_add_excluded_company_from_claude_md_is_conflict():
    call("save_preference", {"key": "excluded_company", "value": "Globex"})
    result, is_error = call("add_application", {"company": "Globex", "title": "DE"})
    assert is_error and result.startswith("ERR|conflict|")
    assert jobs_domain.all_applications() == []


def test_sdk_update_unknown_id():
    assert call("update_status", {"id": 42, "status": "phone_screen"}) == ("ERR|not_found|id=42", True)


def test_sdk_list_is_capped_and_truncated():
    for i in range(30):
        call("add_application", {"company": "C" * 40 + str(i), "title": "T" * 60 + str(i)})
    result, is_error = call("list_applications", {})
    lines = result.splitlines()
    assert not is_error and len(lines) <= 20 and lines[0].startswith("OK|count=30|shown=19")
    _id, company, title, _status = lines[1].split("|")
    assert len(company) <= 25 and len(title) <= 40


def test_sdk_save_preference_rejects_unknown_key():
    assert call("save_preference", {"key": "favorite_color", "value": "blue"}) == ("ERR|invalid_key|favorite_color", True)


def test_sdk_internal_error_becomes_err_internal(monkeypatch):
    def boom(*_, **__):
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(jobs_domain, "find_applications", boom)
    assert call("list_applications", {}) == ("ERR|internal|RuntimeError", True)


def test_sdk_every_result_is_compact_ok_or_err():
    calls = [("add_application", {"company": "A", "title": "B"}), ("list_applications", {"status": "bad"}),
             ("update_status", {"id": "x", "status": "applied"}), ("save_preference", {"key": "role", "value": "DE"}),
             ("search_jobs", {"query": "data", "location": "Tel Aviv"}), ("search_jobs", {"query": "zzz"})]
    for name, args in calls:
        result, is_error = call(name, args)
        assert LINE.match(result), (name, result)
        assert is_error == result.startswith("ERR|")
        assert "{" not in result and "IGNORE" not in result  # no JSON, no posting descriptions

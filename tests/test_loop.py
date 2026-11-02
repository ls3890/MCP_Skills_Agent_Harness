import json

import pytest

from fake_client import FakeClient, text_response, tool_response
from API.harness import history, loop
from shared import jobs_domain


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(jobs_domain, "DATA_DIR", tmp_path)


def run(client, max_iterations=5):
    messages = [{"role": "user", "content": "hi"}]
    result = loop.run_turn(client, {"model": "m", "max_completion_tokens": 10}, messages, max_iterations,
                           on_message=history.append_message, on_response=lambda r, t: None)
    return result, messages


def test_ends_normally():
    client = FakeClient([text_response("hello")])
    result, messages = run(client)
    assert result["reply"] == "hello" and result["iterations"] == 1 and not result["capped"]
    json.dumps(messages)  # stored as plain dicts


def test_two_tool_calls_in_one_response():
    client = FakeClient([
        tool_response(("t1", "add_application", {"company": "Acme", "title": "DE"}),
                      ("t2", "list_applications", {})),
        text_response("done"),
    ])
    result, messages = run(client)
    assert [c["name"] for c in result["tool_calls"]] == ["add_application", "list_applications"]
    # both results directly follow the assistant message, before the next model call
    assert [m["role"] for m in messages] == ["user", "assistant", "tool", "tool", "assistant"]
    assert [m["tool_call_id"] for m in messages[2:4]] == ["t1", "t2"]
    assert [m["role"] for m in client.calls[1]["messages"]] == ["user", "assistant", "tool", "tool"]


def test_error_result_is_flagged():
    client = FakeClient([tool_response(("t1", "update_status", {"id": 7, "status": "offer"})),
                         text_response("not found")])
    result, messages = run(client)
    assert result["tool_calls"][0]["is_error"] is True
    assert messages[2]["content"] == "ERR|not_found|id=7"


def test_bad_json_arguments_become_error():
    client = FakeClient([tool_response(("t1", "list_applications", "{not json")), text_response("sorry")])
    result, messages = run(client)
    assert messages[2]["content"].startswith("ERR|bad_json") and result["tool_calls"][0]["is_error"]


def test_iteration_cap(capsys):
    client = FakeClient([tool_response((f"t{i}", "list_applications", {})) for i in range(3)])
    result, _ = run(client, max_iterations=3)
    assert result["capped"] and result["reply"] == loop.CAP_MESSAGE
    assert len(client.calls) == 3
    assert "[loop.py] MAX_ITERATIONS (3) reached" in capsys.readouterr().out


def test_messages_written_immediately():
    seen = []
    client = FakeClient([tool_response(("t1", "list_applications", {})), text_response("ok")])
    original = client.chat.completions.create

    def create(**kwargs):
        seen.append(len(history.read_all()))  # records on disk before each call
        return original(**kwargs)

    client.chat.completions.create = create
    run(client)
    # call 1: nothing yet (the user message is written by agent.py); call 2: assistant + tool result on disk
    assert seen == [0, 2]

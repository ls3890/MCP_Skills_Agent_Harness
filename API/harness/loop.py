"""Responsibility: the tool-use loop for one user turn."""
import json
import time

from API.harness import tools

CAP_MESSAGE = "I stopped: this request needed too many tool calls. Please narrow it down."


def log(msg: str) -> None:
    print(f"[loop.py] {msg}")


def to_dict(message) -> dict:
    """SDK assistant message -> plain serializable dict with only the fields the API accepts back."""
    out = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        out["tool_calls"] = [{"id": tc.id, "type": "function",
                              "function": {"name": tc.function.name, "arguments": tc.function.arguments or "{}"}}
                             for tc in message.tool_calls]
    return out


def _run_tool(call: dict) -> tuple[str, bool, dict]:
    try:
        args = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError:
        return "ERR|bad_json|arguments were not valid JSON", True, {}
    if not isinstance(args, dict):
        return "ERR|bad_json|arguments must be an object", True, {}
    result, is_error = tools.dispatch(call["function"]["name"], args)
    return result, is_error, args


def run_turn(client, request: dict, messages: list[dict], max_iterations: int,
             on_message, on_response) -> dict:
    """request: model/max tokens/tools/extras. messages: system + context + the new user
    message; extended in place. on_message(msg) persists each message as soon as it exists."""
    tool_calls = []
    for iteration in range(1, max_iterations + 1):
        started = time.perf_counter()
        response = client.chat.completions.create(messages=messages, **request)
        on_response(response, time.perf_counter() - started)
        choice = response.choices[0]
        assistant = to_dict(choice.message)
        messages.append(assistant)
        on_message(assistant)
        log(f"iteration {iteration}: finish_reason={choice.finish_reason}")

        if not assistant.get("tool_calls"):
            reply = assistant["content"].strip()
            if choice.finish_reason == "length" and not reply:
                reply = "My reply was cut off. Please ask again."
            return {"reply": reply, "iterations": iteration, "tool_calls": tool_calls, "capped": False}

        # One tool message per call, all appended together right after the assistant message.
        for call in assistant["tool_calls"]:
            result, is_error, args = _run_tool(call)
            tool_calls.append({"name": call["function"]["name"], "input": args,
                               "result": result, "is_error": is_error})
            tool_message = {"role": "tool", "tool_call_id": call["id"], "content": result}
            messages.append(tool_message)
            on_message(tool_message)

    log(f"MAX_ITERATIONS ({max_iterations}) reached; stopping this turn")
    return {"reply": CAP_MESSAGE, "iterations": max_iterations, "tool_calls": tool_calls, "capped": True}

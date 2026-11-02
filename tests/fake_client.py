"""Scripted stand-ins: groq.Groq() for API/, ClaudeSDKClient for SDK/. Also puts the project root on sys.path."""
import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def usage(input=100, output=20, cached=0):
    return SimpleNamespace(prompt_tokens=input, completion_tokens=output,
                           prompt_tokens_details=SimpleNamespace(cached_tokens=cached))


def _response(message, finish_reason, u):
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)], usage=usage(**u))


def text_response(text, **u):
    return _response(SimpleNamespace(role="assistant", content=text, tool_calls=None), "stop", u)


def tool_response(*calls, **u):
    """calls: (id, name, input) tuples, all in one assistant response. input may be a raw string."""
    tool_calls = [SimpleNamespace(id=i, type="function", function=SimpleNamespace(
        name=n, arguments=inp if isinstance(inp, str) else json.dumps(inp))) for i, n, inp in calls]
    return _response(SimpleNamespace(role="assistant", content=None, tool_calls=tool_calls), "tool_calls", u)


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("FakeClient ran out of scripted responses")
        return self.responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=_Completions(responses))

    @property
    def calls(self):
        return self.chat.completions.calls


# ---------- SDK path: fake ClaudeSDKClient, plus a stand-in claude_agent_sdk module if it isn't installed ----------

def _install_sdk_stand_in():
    """Mirrors only the documented names SDK/ imports. Used when claude-agent-sdk isn't installed."""
    import types
    from dataclasses import dataclass, field
    from typing import Any

    sdk = types.ModuleType("claude_agent_sdk")
    sdk.IS_STAND_IN = True

    @dataclass
    class SdkMcpTool:
        name: str
        description: str
        input_schema: Any
        handler: Any

    def tool(name, description, input_schema, annotations=None):
        return lambda handler: SdkMcpTool(name, description, input_schema, handler)

    def create_sdk_mcp_server(name, version="1.0.0", tools=None):
        return {"type": "sdk", "name": name, "instance": None, "tools": tools}

    @dataclass
    class ClaudeAgentOptions:
        tools: Any = None
        allowed_tools: list = field(default_factory=list)
        system_prompt: Any = None
        mcp_servers: Any = field(default_factory=dict)
        strict_mcp_config: bool = False
        resume: Any = None
        max_turns: Any = None
        model: Any = None
        cwd: Any = None
        setting_sources: Any = None
        effort: Any = None
        env: dict = field(default_factory=dict)

    @dataclass
    class TextBlock:
        text: str

    @dataclass
    class ToolUseBlock:
        id: str
        name: str
        input: dict

    @dataclass
    class ToolResultBlock:
        tool_use_id: str
        content: Any = None
        is_error: Any = None

    @dataclass
    class AssistantMessage:
        content: list
        model: str
        message_id: Any = None

    @dataclass
    class UserMessage:
        content: Any

    @dataclass
    class ResultMessage:
        subtype: str
        duration_ms: int
        duration_api_ms: int
        is_error: bool
        num_turns: int
        session_id: str
        total_cost_usd: Any = None
        usage: Any = None
        result: Any = None
        model_usage: Any = None

    class ClaudeSDKClient:  # tests replace this with FakeSDKClient
        def __init__(self, options=None):
            raise RuntimeError("stand-in SDK: no live client")

    for obj in (SdkMcpTool, tool, create_sdk_mcp_server, ClaudeAgentOptions, TextBlock, ToolUseBlock,
                ToolResultBlock, AssistantMessage, UserMessage, ResultMessage, ClaudeSDKClient):
        setattr(sdk, obj.__name__, obj)
    sys.modules["claude_agent_sdk"] = sdk


try:
    import claude_agent_sdk  # noqa: F401
except ImportError:
    _install_sdk_stand_in()


def sdk_usage(input=100, output=20, cache_read=0, cache_write=0):
    return {"input_tokens": input, "output_tokens": output,
            "cache_read_input_tokens": cache_read, "cache_creation_input_tokens": cache_write}


class FakeSDKClient:
    """Scripted stand-in for ClaudeSDKClient. Each script is one turn: a list of steps,
    ("tool", name, args) runs the real SDK/tools handler, ("text", str) is the reply.
    Records every options object and prompt. Never touches the network."""

    instances: list = []
    scripts: list = []
    context: dict = {"memoryFiles": [], "categories": []}

    def __init__(self, options=None):
        self.options = options
        self.prompts = []
        FakeSDKClient.instances.append(self)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def query(self, prompt):
        self.prompts.append(prompt)

    async def receive_response(self):
        from claude_agent_sdk import (AssistantMessage, ResultMessage, TextBlock, ToolResultBlock,
                                      ToolUseBlock, UserMessage)
        from SDK import tools

        steps = FakeSDKClient.scripts.pop(0)
        reply = ""
        for i, step in enumerate(steps):
            if step[0] == "tool":
                _, name, args = step
                out = await tools.HANDLERS[name](args)
                yield AssistantMessage(content=[ToolUseBlock(id=f"t{i}", name=f"mcp__jobs__{name}", input=args)],
                                       model="fake", message_id=f"m{i}")
                yield UserMessage(content=[ToolResultBlock(tool_use_id=f"t{i}", content=out["content"],
                                                           is_error=out["is_error"])])
            else:
                reply = step[1]
                yield AssistantMessage(content=[TextBlock(text=reply)], model="fake", message_id=f"m{i}")
        yield ResultMessage(subtype="success", duration_ms=5, duration_api_ms=4, is_error=False,
                            num_turns=len(steps), session_id=self.options.resume or "sess-1",
                            total_cost_usd=0.01, usage=sdk_usage(), result=reply, model_usage={})

    async def get_context_usage(self):
        return FakeSDKClient.context

"""P2 runtime 测试：RunResult 信封解析 + agentao 不在 PATH 的兜底 + OpenAIRuntime agent loop（不打真实 LLM）。"""

import json
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

from guanlan.runtime import (
    AgentTaskRequest,
    OpenAIRuntime,
    _parse_envelope,
    _read_file_tool,
    _safe_join,
    _write_file_tool,
    run_agent_task,
)


def test_parse_envelope_ok():
    r = _parse_envelope(0, '{"status": "ok", "final_text": "答案"}', "")
    assert r.ok and r.final_text == "答案" and r.error_type is None


def test_parse_envelope_error_type():
    r = _parse_envelope(
        3, '{"status": "error", "error": {"type": "permission_required"}}', ""
    )
    assert not r.ok and r.error_type == "permission_required"


def test_parse_envelope_nonzero_without_error_block():
    r = _parse_envelope(1, '{"status": "ok", "final_text": "x"}', "")
    assert not r.ok and r.error_type == "runtime_error"


def test_parse_envelope_unparsable_stdout():
    r = _parse_envelope(0, "not json", "boom")
    assert not r.ok and r.error_type == "runtime_error" and "boom" in r.final_text


def test_parse_envelope_falls_back_to_error_message():
    """final_text 缺失时用 error.message 作诊断（如 invalid_spec / permission_denied）。"""
    r = _parse_envelope(
        3,
        '{"status": "error", "error": {"type": "invalid_spec", "message": "skill not found"}}',
        "",
    )
    assert not r.ok and r.error_type == "invalid_spec"
    assert "skill not found" in r.final_text


def test_status_ok_but_llm_api_error_is_failure():
    """status=ok + 退出码 0，但 final_text 含 `[LLM API error:]` → 仍判失败（不当成功 no-op）。"""
    r = _parse_envelope(
        0, '{"status": "ok", "final_text": "[LLM API error: 401 unauthorized]"}', ""
    )
    assert not r.ok
    assert r.error_type == "runtime_error"
    assert "LLM API error" in r.final_text


def test_missing_agentao_executable_is_runtime_error(tmp_path: Path, monkeypatch):
    """agentao 不在 PATH → subprocess 抛 OSError → 归一为 runtime_error，不抛 traceback。"""

    def boom(*args, **kwargs):
        raise FileNotFoundError("agentao")

    monkeypatch.setattr(subprocess, "run", boom)

    # skills=() 时不触发 skill 兜底；直接走到 subprocess.run 的 OSError 分支。
    r = run_agent_task("q", working_directory=tmp_path, skills=())
    assert not r.ok
    assert r.error_type == "runtime_error"
    assert "PATH" in r.final_text


# ── OpenAIRuntime 测试 ──────────────────────────────────────────────────────


def test_safe_join_rejects_traversal(tmp_path: Path):
    """_safe_join 拒绝越界路径（../、绝对路径）。"""
    assert _safe_join(tmp_path, "raw/test.md") == (tmp_path / "raw" / "test.md").resolve()
    assert _safe_join(tmp_path, "../etc/passwd") is None
    assert _safe_join(tmp_path, "/etc/passwd") is None


def test_read_file_tool(tmp_path: Path):
    """_read_file_tool 读文件，越界返回错误串。"""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "x.md").write_text("hello")
    assert _read_file_tool("raw/x.md", working_directory=tmp_path) == "hello"
    assert "不存在" in _read_file_tool("raw/missing.md", working_directory=tmp_path)
    assert "越界" in _read_file_tool("../etc/passwd", working_directory=tmp_path)


def test_write_file_tool(tmp_path: Path):
    """_write_file_tool 写文件、建父目录；越界拒绝。"""
    _write_file_tool("wiki/sources/test.md", "content", working_directory=tmp_path)
    assert (tmp_path / "wiki" / "sources" / "test.md").read_text() == "content"
    assert "越界" in _write_file_tool("../etc/x", "x", working_directory=tmp_path)


def test_openai_runtime_no_api_key(monkeypatch, tmp_path: Path):
    """无 OPENAI_API_KEY → 优雅降级为 runtime_error。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    rt = OpenAIRuntime(model="gpt-4o-mini")
    r = rt.run(AgentTaskRequest(prompt="x", working_directory=tmp_path, skills=()))
    assert not r.ok
    assert r.error_type == "runtime_error"
    assert "OPENAI_API_KEY" in r.final_text


def test_openai_runtime_no_openai_package(monkeypatch, tmp_path: Path):
    """openai 包未装 → 优雅降级为 runtime_error。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")
    monkeypatch.setattr("builtins.__import__", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError(name)))
    rt = OpenAIRuntime(model="gpt-4o-mini")
    r = rt.run(AgentTaskRequest(prompt="x", working_directory=tmp_path, skills=()))
    assert not r.ok
    assert r.error_type == "runtime_error"
    assert "openai" in r.final_text.lower()


def _mock_tool_call(tc_id: str, fn_name: str, args: dict):
    """构造一个 mock ChatCompletionMessageToolCall。"""
    tc = MagicMock()
    tc.id = tc_id
    tc.function.name = fn_name
    tc.function.arguments = json.dumps(args)
    return tc


def _mock_msg(*, content=None, tool_calls=None):
    """构造一个 mock ChatCompletionMessage。"""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = tool_calls
    msg.model_dump = lambda exclude_none=False: {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
            for tc in (tool_calls or [])
        ] or None,
    }
    return msg


def test_openai_runtime_agent_loop_read_then_done(monkeypatch, tmp_path: Path):
    """完整 agent loop：第一轮调 read_file，第二轮返回最终文本。"""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "test.md").write_text("# Test\nHello world")
    (tmp_path / "wiki").mkdir()

    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message = _mock_msg(tool_calls=[
        _mock_tool_call("tc1", "read_file", {"path": "raw/test.md"})
    ])

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message = _mock_msg(content="已读取，内容是 Hello world。")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("openai.OpenAI", return_value=mock_client):
        rt = OpenAIRuntime(model="gpt-4o-mini")
        r = rt.run(AgentTaskRequest(
            prompt="读取 raw/test.md 并总结",
            working_directory=tmpdir if (tmpdir := tmp_path) else tmp_path,
            permission_mode="read-only",
            skills=(),
        ))

    assert r.ok
    assert "Hello world" in r.final_text
    assert r.raw["model"] == "gpt-4o-mini"
    # 两轮 API 调用
    assert len(mock_client.chat.completions.create.call_args_list) == 2


def test_openai_runtime_write_tool_in_write_mode(monkeypatch, tmp_path: Path):
    """workspace-write 模式暴露 write_file，LLM 能写 wiki/ 页面。"""
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "src.md").write_text("# Src\nContent")
    (tmp_path / "wiki").mkdir()
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message = _mock_msg(tool_calls=[
        _mock_tool_call("tc1", "write_file", {"path": "wiki/sources/src.md", "content": "# 摘要"})
    ])

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message = _mock_msg(content="已写入 wiki/sources/src.md。")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("openai.OpenAI", return_value=mock_client):
        rt = OpenAIRuntime(model="gpt-4o-mini")
        r = rt.run(AgentTaskRequest(
            prompt="ingest raw/src.md",
            working_directory=tmp_path,
            permission_mode="workspace-write",
            skills=(),
        ))

    assert r.ok
    assert (tmp_path / "wiki" / "sources" / "src.md").read_text() == "# 摘要"


def test_openai_runtime_readonly_blocks_write_tool(monkeypatch, tmp_path: Path):
    """read-only 模式不暴露 write_file——LLM 调它会被判为未知工具。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-fake")

    resp1 = MagicMock()
    resp1.choices = [MagicMock()]
    resp1.choices[0].message = _mock_msg(tool_calls=[
        _mock_tool_call("tc1", "write_file", {"path": "wiki/x.md", "content": "hack"})
    ])

    resp2 = MagicMock()
    resp2.choices = [MagicMock()]
    resp2.choices[0].message = _mock_msg(content="完成。")

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = [resp1, resp2]

    with patch("openai.OpenAI", return_value=mock_client):
        rt = OpenAIRuntime(model="gpt-4o-mini")
        r = rt.run(AgentTaskRequest(
            prompt="write to wiki",
            working_directory=tmp_path,
            permission_mode="read-only",
            skills=(),
        ))

    # agent loop 不会因未知工具崩溃，write_file 被判未知 → 返回错误串但 loop 继续
    assert r.ok  # loop 正常完成
    # 文件不应被写（write_file 未注册到 read-only 的工具集）
    assert not (tmp_path / "wiki" / "x.md").exists()


def test_guanlan_runtime_env_selects_openai(monkeypatch, tmp_path: Path):
    """GUANLAN_RUNTIME=openai 时 run_agent_task 走 OpenAIRuntime。"""
    monkeypatch.setenv("GUANLAN_RUNTIME", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    r = run_agent_task("test", working_directory=tmp_path, skills=())
    assert not r.ok
    assert r.error_type == "runtime_error"
    assert "OPENAI_API_KEY" in r.final_text

"""Docker command isolation and workspace path boundaries."""

import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from corecoder.agent import Agent
from corecoder.config import Config
from corecoder.demo import ScriptedLLM
from corecoder.sandbox import (
    CommandResult,
    DockerCommandExecutor,
    SandboxViolation,
    WorkspacePathPolicy,
)
from corecoder.tools import build_tools
from corecoder.tools.bash import BashTool
from corecoder.tools.edit import EditFileTool
from corecoder.tools.glob_tool import GlobTool
from corecoder.tools.grep import GrepTool
from corecoder.tools.read import ReadFileTool
from corecoder.tools.write import WriteFileTool


def test_workspace_policy_accepts_inside_and_rejects_outside(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    policy = WorkspacePathPolicy(workspace)

    assert policy.resolve("src/example.py") == workspace / "src" / "example.py"
    with pytest.raises(SandboxViolation, match="outside workspace"):
        policy.resolve(tmp_path / "secret.txt")


def test_workspace_policy_rejects_symlink_escape(tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    link = workspace / "escape"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    policy = WorkspacePathPolicy(workspace)
    with pytest.raises(SandboxViolation, match="outside workspace"):
        policy.resolve(link / "secret.txt")


def test_file_tools_cannot_leave_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    policy = WorkspacePathPolicy(workspace)

    assert "outside workspace" in ReadFileTool(policy).execute(str(outside))
    assert "outside workspace" in WriteFileTool(policy).execute(str(outside), "changed")
    assert "outside workspace" in EditFileTool(policy).execute(str(outside), "secret", "changed")
    assert "outside workspace" in GrepTool(policy).execute("secret", str(outside))
    assert "outside workspace" in GlobTool(policy).execute("*.txt", str(tmp_path))
    assert outside.read_text(encoding="utf-8") == "secret"


def test_glob_pattern_cannot_traverse_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    result = GlobTool(WorkspacePathPolicy(workspace)).execute("../*.txt")
    assert "may not escape" in result


def test_docker_executor_applies_security_boundary(tmp_path):
    executor = DockerCommandExecutor(tmp_path, image="corecoder-test:1")
    completed = SimpleNamespace(stdout="ok\n", stderr="", returncode=0)

    with mock.patch("corecoder.sandbox.subprocess.run", return_value=completed) as run:
        result = executor.run("python -V", str(tmp_path), timeout=12)

    assert result == CommandResult("ok\n", "", 0)
    argv = run.call_args.args[0]
    assert argv[:2] == ["docker", "run"]
    assert "--read-only" in argv
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--security-opt") + 1] == "no-new-privileges"
    assert argv[argv.index("--pull") + 1] == "never"
    assert argv[argv.index("--workdir") + 1] == "/workspace"
    mount = argv[argv.index("--mount") + 1]
    assert mount == f"type=bind,src={tmp_path},dst=/workspace"
    assert not mount.endswith(",rw")
    assert argv[-3:] == ["corecoder-test:1", "-lc", "python -V"]
    assert run.call_args.kwargs["shell"] is False
    assert run.call_args.kwargs["timeout"] == 12


def test_docker_executor_maps_nested_workdir(tmp_path):
    nested = tmp_path / "src" / "pkg"
    nested.mkdir(parents=True)
    executor = DockerCommandExecutor(tmp_path)
    assert str(executor._container_path(nested)) == "/workspace/src/pkg"


def test_docker_executor_rejects_workdir_outside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    executor = DockerCommandExecutor(workspace)
    with pytest.raises(SandboxViolation, match="working directory"):
        executor._container_path(tmp_path)


def test_docker_timeout_forces_named_container_removal(tmp_path):
    executor = DockerCommandExecutor(tmp_path)
    timeout = subprocess.TimeoutExpired(cmd="docker", timeout=1)
    removed = SimpleNamespace(stdout="", stderr="", returncode=0)

    with (
        mock.patch("corecoder.sandbox.subprocess.run", side_effect=[timeout, removed]) as run,
        pytest.raises(subprocess.TimeoutExpired),
    ):
        executor.run("sleep 10", str(tmp_path), timeout=1)

    cleanup_argv = run.call_args_list[1].args[0]
    assert cleanup_argv[:3] == ["docker", "rm", "-f"]
    assert cleanup_argv[3].startswith("corecoder-")


def test_bash_uses_injected_executor(tmp_path):
    class FakeExecutor:
        mode = "docker"
        workspace_root = Path(tmp_path)

        def run(self, command, cwd, timeout):
            assert command == "echo hello"
            assert cwd == str(tmp_path)
            assert timeout == 9
            return CommandResult("hello\n", "", 0)

    result = BashTool(FakeExecutor()).execute("echo hello", timeout=9)
    assert result == "hello"


def test_bash_maps_container_cd_without_leaking_between_instances(tmp_path):
    nested = tmp_path / "src"
    nested.mkdir()

    class FakeExecutor:
        mode = "docker"
        workspace_root = Path(tmp_path)
        container_workspace = Path("/workspace")

        def __init__(self):
            self.cwds = []

        def run(self, command, cwd, timeout):
            self.cwds.append(cwd)
            return CommandResult("", "", 0)

    first_executor = FakeExecutor()
    first = BashTool(first_executor)
    first.execute("cd /workspace/src")
    first.execute("pwd")
    assert first_executor.cwds == [str(tmp_path), str(nested)]

    second_executor = FakeExecutor()
    BashTool(second_executor).execute("pwd")
    assert second_executor.cwds == [str(tmp_path)]


def test_build_tools_returns_fresh_sandbox_bound_instances(tmp_path):
    policy = WorkspacePathPolicy(tmp_path)
    executor = DockerCommandExecutor(tmp_path)
    first = build_tools(executor=executor, path_policy=policy)
    second = build_tools(executor=executor, path_policy=policy)

    assert first is not second
    assert first[0] is not second[0]
    assert first[0].executor is executor
    assert first[1].path_policy is policy


def test_default_agents_do_not_share_tool_instances():
    first = Agent(llm=ScriptedLLM([]))
    second = Agent(llm=ScriptedLLM([]))
    assert first.tools is not second.tools
    assert first.tools[0] is not second.tools[0]


def test_sandbox_config_from_env(monkeypatch):
    monkeypatch.setenv("CORECODER_SANDBOX", "docker")
    monkeypatch.setenv("CORECODER_SANDBOX_IMAGE", "project-tools:7")
    monkeypatch.setenv("CORECODER_SANDBOX_NETWORK", "none")
    monkeypatch.setenv("CORECODER_SANDBOX_MEMORY", "768m")
    monkeypatch.setenv("CORECODER_SANDBOX_CPUS", "1.5")
    monkeypatch.setenv("CORECODER_SANDBOX_PIDS", "64")

    config = Config.from_env()
    assert config.sandbox == "docker"
    assert config.sandbox_image == "project-tools:7"
    assert config.sandbox_memory == "768m"
    assert config.sandbox_cpus == 1.5
    assert config.sandbox_pids == 64


def test_sandbox_cli_flags_parse(monkeypatch):
    from corecoder.cli import _parse_args

    monkeypatch.setattr("sys.argv", [
        "corecoder", "--sandbox", "docker", "--sandbox-image", "project-tools:8",
        "--sandbox-network", "none", "--sandbox-memory", "512m",
        "--sandbox-cpus", "2", "--sandbox-pids", "32",
    ])
    args = _parse_args()
    assert args.sandbox == "docker"
    assert args.sandbox_image == "project-tools:8"
    assert args.sandbox_memory == "512m"
    assert args.sandbox_cpus == 2
    assert args.sandbox_pids == 32

"""A timed-out capture must never be reported as completed full-suite evidence."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def capture_module(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "docs/evidence/issue-2336-png-failures"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("png_coverage_capture", directory / "capture.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def source(tmp_path):
    repo = tmp_path / "repo"
    (repo / "pa_core").mkdir(parents=True)
    (repo / "pa_core/cli.py").write_text("# original product\n")
    (repo / "tests").mkdir()
    (repo / "pyproject.toml").write_text("# capture configuration\n")
    (repo / "tests/test_example.py").write_text("def test_example():\n    assert True\n")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "base",
        ],
        cwd=repo,
        check=True,
    )
    return repo


@pytest.mark.parametrize("untracked", [False, True])
def test_capture_rejects_changed_product_before_creating_output(capture_module, source, untracked):
    changed = source / ("pa_core/new.py" if untracked else "pa_core/cli.py")
    changed.write_text("# changed\n")
    output = source.parent / "evidence"
    with pytest.raises(ValueError, match="Product source differs"):
        capture_module.capture(source, output, "HEAD", 10)
    assert not output.exists()


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "pytest.toml",
        ".pytest.toml",
        "pytest.ini",
        ".pytest.ini",
        "tox.ini",
        "setup.cfg",
        ".coveragerc",
    ],
)
def test_capture_rejects_configuration_drift_before_running_pytest(
    capture_module, source, monkeypatch, name
):
    (source / name).write_text("# changed coverage or pytest configuration\n")
    output = source.parent / "evidence"
    real_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "timeout":
            pytest.fail("Configuration drift reached pytest")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(capture_module.subprocess, "run", run)
    with pytest.raises(ValueError, match="Test or coverage configuration differs"):
        capture_module.capture(source, output, "HEAD", 10)
    assert not output.exists()


def test_capture_rejects_gitignored_configuration(capture_module, source, monkeypatch):
    (source / ".git/info/exclude").write_text(".coveragerc\n")
    (source / ".coveragerc").write_text("[run]\nomit = pa_core/cli.py\n")
    output = source.parent / "evidence"
    real_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "timeout":
            pytest.fail("Gitignored coverage configuration reached pytest")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(capture_module.subprocess, "run", run)
    with pytest.raises(ValueError, match="Test or coverage configuration differs"):
        capture_module.capture(source, output, "HEAD", 10)
    assert not output.exists()


def test_capture_rejects_deleted_configuration(capture_module, source):
    (source / "pyproject.toml").unlink()
    output = source.parent / "evidence"
    with pytest.raises(ValueError, match="Test or coverage configuration differs"):
        capture_module.capture(source, output, "HEAD", 10)
    assert not output.exists()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("PYTEST_ADDOPTS", "-x -k test_example"),
        ("COVERAGE_RCFILE", "alternative.coveragerc"),
        ("COVERAGE_PROCESS_START", "alternative.coveragerc"),
    ],
)
def test_capture_rejects_environment_overrides_before_running_pytest(
    capture_module, source, monkeypatch, name, value
):
    """Inherited options can make a partial or differently measured suite look complete."""
    monkeypatch.setenv(name, value)
    output = source.parent / "evidence"
    real_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] == "timeout":
            pytest.fail(f"Inherited {name} reached pytest")
        return real_run(argv, **kwargs)

    monkeypatch.setattr(capture_module.subprocess, "run", run)
    with pytest.raises(
        ValueError, match=f"Environment override changes test or coverage settings: {name}"
    ):
        capture_module.capture(source, output, "HEAD", 10)
    assert not output.exists()


@pytest.mark.parametrize("returncode", [0, 1, 124, 137])
def test_capture_records_inputs_and_rejects_timeout(
    capture_module, source, monkeypatch, returncode
):
    output = source.parent / "evidence"
    (source / "tests/test_example.py").write_text("def test_example():\n    assert False\n")
    (source / "tests/test_added.py").write_text("def test_added():\n    assert True\n")
    real_run = subprocess.run

    def run(argv, **kwargs):
        if argv[0] != "timeout":
            return real_run(argv, **kwargs)
        assert argv[:4] == ["timeout", "--signal=INT", "--kill-after=10s", "10s"]
        assert argv[4:7] == [sys.executable, "-m", "pytest"]
        assert argv[-1] == "tests/"
        assert argv.count("-m") == 1  # Python module flag only; pytest uses repository defaults.
        assert kwargs["env"]["COVERAGE_FILE"] == str(output / ".coverage")
        assert json.loads((output / "command.json").read_text())["completed"] is False
        assert b"assert False" in (output / "tests.patch").read_bytes()
        assert b"+++ b/tests/test_added.py" in (output / "tests.patch").read_bytes()
        assert (
            "tests/test_added.py"
            in json.loads((output / "command.json").read_text())["test_sha256"]
        )
        kwargs["stdout"].write(b"retained stdout and stderr\n")
        summary = dict(covered_lines=3, num_statements=4, missing_lines=1, percent_covered=75.0)
        (output / "coverage.json").write_text(
            json.dumps(
                {
                    "totals": summary,
                    "files": {"pa_core/cli.py": {"summary": summary, "missing_lines": [4]}},
                }
            )
        )
        (output / "completion.json").write_text(
            json.dumps(
                {
                    "selected": ["tests/test_example.py::test_example"],
                    "finished": ["tests/test_example.py::test_example"],
                    "deselected": 0,
                    "exit": returncode,
                }
            )
        )
        failed = int(returncode == 1)
        (output / "full-suite.xml").write_text(
            f'<testsuite tests="1" failures="{failed}" errors="0" skipped="0">'
            '<testcase classname="tests.test_example" name="test_example">'
            + ('<failure message="AssertionError"/>' if failed else "")
            + "</testcase></testsuite>"
        )
        return subprocess.CompletedProcess(argv, returncode)

    monkeypatch.setattr(capture_module.subprocess, "run", run)
    assert capture_module.capture(source, output, "HEAD", 10) == returncode
    metadata = json.loads((output / "command.json").read_text())
    assert metadata["exit"] == returncode
    assert metadata["completed"] is (returncode in (0, 1))
    assert (output / "summary.json").exists() is (returncode in (0, 1))
    assert (output / "full-suite.txt").read_text() == "retained stdout and stderr\n"
    if returncode not in (0, 1):
        assert "Incomplete pytest run" in metadata["incomplete_reason"]
    else:
        assert json.loads((output / "summary.json").read_text())["counts"]["failed"] == int(
            returncode == 1
        )
    with pytest.raises(FileExistsError):
        capture_module.capture(source, output, "HEAD", 10)


def test_capture_bounds_a_real_stalled_process(capture_module, source, monkeypatch):
    if shutil.which("timeout") is None:
        pytest.skip("Capture driver requires GNU timeout")
    interpreter = source.parent / "stalled-python"
    interpreter.write_text(
        "#!" + sys.executable + "\n"
        "import time\n"
        "print('stalled pytest witness', flush=True)\n"
        "time.sleep(60)\n"
    )
    interpreter.chmod(0o755)
    monkeypatch.setattr(sys, "executable", str(interpreter))
    output = source.parent / "evidence"
    # A renderer may ignore SIGINT and require timeout's subsequent SIGKILL.
    assert capture_module.capture(source, output, "HEAD", 1) in (124, 137)
    metadata = json.loads((output / "command.json").read_text())
    assert metadata["completed"] is False
    assert "Incomplete pytest run" in metadata["incomplete_reason"]
    assert "stalled pytest witness" in (output / "full-suite.txt").read_text()
    assert not (output / "summary.json").exists()


@pytest.mark.parametrize("stop_early", [False, True])
def test_capture_requires_all_selected_nodes_to_finish(
    capture_module, source, monkeypatch, stop_early
):
    """Exit 1 plus consistent JUnit/coverage can still describe a truncated suite."""
    monkeypatch.delenv("PYTEST_ADDOPTS", raising=False)
    monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    (source / "pa_core/__init__.py").write_text("")
    (source / "pa_core/cli.py").write_text("value = 1\n")
    (source / "dashboard").mkdir()
    (source / "dashboard/__init__.py").write_text("")
    (source / "tests/test_example.py").write_text(
        "from pa_core.cli import value\n"
        "def test_first(request):\n"
        + ("    del request.session.items[1:]\n" if stop_early else "")
        + "    assert value == 2\n"
        "def test_second():\n"
        "    assert value == 1\n"
        "def test_third():\n"
        "    assert value == 1\n"
    )
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "capture inputs",
        ],
        cwd=source,
        check=True,
    )
    output = source.parent / "evidence"
    assert capture_module.capture(source, output, "HEAD", 30) == 1
    metadata = json.loads((output / "command.json").read_text())
    assert metadata["completed"] is (not stop_early)
    assert (output / "summary.json").exists() is (not stop_early)
    completion = json.loads((output / "completion.json").read_text())
    assert len(completion["selected"]) == 3
    if stop_early:
        assert completion["finished"] == ["tests/test_example.py::test_first"]
        assert "Selected tests did not finish" in metadata["incomplete_reason"]
    else:
        assert completion["finished"] == completion["selected"]
        assert metadata["test_selection"] == {"selected": 3, "deselected": 0, "finished": 3}

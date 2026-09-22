"""
test_providers.py — credence/providers.py had no test file at all.

It is the provider-agnostic client factory the eval scripts use, and it is part
of the shipped package. Nothing exercised it, which is how it kept a module-level
`import requests` for a dependency that pyproject.toml did not declare anywhere:
`import credence.providers` raised ModuleNotFoundError on a clean install, and
the only thing that noticed was a legacy suite whose results were being thrown
away. In CI that single undeclared import blanket-failed 14 checks across two
suites, because both suites wrap their bodies in `except Exception` and mark
every check in the suite failed.

The lesson is the same one the enforcement bugs taught: a module nobody imports
in a test is a module nobody has checked. `anthropic` was already handled the
right way — imported inside `make_client`, declared in the `api` extra — so this
pins that `requests` now behaves the same way.

Coverage:
  V1 Importing the module does not require `requests`          [regression]
  V2 A missing `requests` is named, with the extra to install
  V3 make_client refuses to guess when no key is available
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

# Repo root. The subprocesses run `python -c`, which puts the cwd on sys.path,
# so `import credence` resolves there without installing the package.
ROOT = Path(__file__).resolve().parent.parent.parent

_BLOCK_REQUESTS = textwrap.dedent(
    """
    import sys


    class _BlockRequests:
        \"\"\"Refuses `requests` the way an install without it would.\"\"\"

        def find_spec(self, name, path=None, target=None):
            if name == "requests" or name.startswith("requests."):
                raise ImportError("No module named 'requests'")
            return None


    sys.meta_path.insert(0, _BlockRequests())
    """
)


def _run(script: str) -> subprocess.CompletedProcess:
    """Run a snippet in a fresh interpreter, with no API keys in the env."""
    env = {
        "PATH": "/usr/bin:/bin:/usr/local/bin",
        "HOME": str(Path.home()),
    }
    return subprocess.run(
        [sys.executable, "-c", _BLOCK_REQUESTS + textwrap.dedent(script)],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        env=env,
        timeout=60,
    )


# ── V1 ───────────────────────────────────────────────────────────────────────

def test_importing_providers_does_not_require_requests():
    """Regression: the module-level import made the package unusable.

    `requests` is needed only by the hf and groq clients, and only when one of
    them actually makes a call. Importing the module is not using a provider, so
    importing it must not require the dependency — the same rule `anthropic`
    already followed here.
    """
    proc = _run(
        """
        import credence.providers as p

        print("imported")
        print(type(p.HFClient("dummy")).__name__)
        """
    )
    assert proc.returncode == 0, (
        f"import credence.providers failed without requests — it is imported at "
        f"module scope again\nstdout: {proc.stdout}\nstderr: {proc.stderr[-600:]}"
    )
    assert "imported" in proc.stdout
    assert "_OAIClient" in proc.stdout


# ── V2 ───────────────────────────────────────────────────────────────────────

def test_missing_requests_names_the_extra_to_install():
    """The failure has to say what to do.

    Declaring a dependency and reporting a missing one are different jobs. The
    undeclared import failed with a bare ModuleNotFoundError naming no package
    and no remedy.
    """
    proc = _run(
        """
        from credence.providers import HFClient

        try:
            HFClient("dummy").create("m", [{"role": "user", "content": "hi"}])
        except ImportError as exc:
            print("IMPORTERROR:", exc)
        try:
            from credence.providers import GroqClient
            GroqClient("dummy").create("m", [{"role": "user", "content": "hi"}])
        except ImportError as exc:
            print("IMPORTERROR:", exc)
        """
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    assert proc.stdout.count("IMPORTERROR:") == 2, (
        f"an hf or groq call did not fail with ImportError naming the missing "
        f"dependency\nstdout: {proc.stdout}\nstderr: {proc.stderr[-600:]}"
    )
    assert "requests" in proc.stdout
    assert "api" in proc.stdout, (
        f"the message does not name the extra that provides `requests`: "
        f"{proc.stdout}"
    )


# ── V3 ───────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("provider,expected", [
    ("anthropic", "No API key found"),
    ("groq", "GROQ_API_KEY"),
])
def test_make_client_refuses_to_guess_without_a_key(provider, expected):
    """No key must be an error, not a client that fails later.

    A factory that returns a client with an empty token turns a configuration
    mistake into a confusing 401 at call time. This pins the message instead.
    """
    proc = _run(
        f"""
        from credence.providers import make_client

        try:
            make_client({provider!r})
            print("NOERROR")
        except EnvironmentError as exc:
            print("ENVERROR:", exc)
        """
    )
    assert proc.returncode == 0, proc.stderr[-600:]
    assert "NOERROR" not in proc.stdout, (
        f"make_client({provider!r}) returned a client with no key available"
    )
    assert expected in proc.stdout, f"stdout: {proc.stdout}"

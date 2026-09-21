"""
install_gate.py — Install the credence-gate Rust binary.

Run after `pip install credence-enforce` to build and install the native
PreToolUse hook. Requires Rust (cargo) to be installed.

Usage:
    python -m credence.install_gate

What it does:
    1. Finds the credence_gate/ source directory (bundled with the package)
    2. Runs `cargo build --release`
    3. Copies the binary to ~/.local/bin/credence-gate (or ~/bin/ as fallback)
    4. Prints the settings.json snippet to add to .claude/settings.json

If Rust is not installed:
    https://rustup.rs — installs in 60 seconds, no system privileges needed.
"""

import shutil
import subprocess
import sys
from pathlib import Path

from credence.matching import enforced_tool_matcher


def _cargo_available() -> bool:
    return shutil.which("cargo") is not None


def _find_gate_src() -> Path | None:
    # When installed as a package, credence_gate/ is alongside credence/
    candidates = [
        Path(__file__).parent.parent / "credence_gate",   # editable install / source
        Path(sys.prefix) / "credence_gate",               # installed wheel (future)
    ]
    for c in candidates:
        if (c / "Cargo.toml").exists():
            return c
    return None


def _install_dir() -> Path:
    candidates = [
        Path.home() / ".local" / "bin",
        Path.home() / "bin",
        Path.home() / ".cargo" / "bin",
    ]
    for d in candidates:
        if d.exists():
            return d
    # Create ~/.local/bin if nothing exists
    d = Path.home() / ".local" / "bin"
    d.mkdir(parents=True, exist_ok=True)
    return d


def main() -> int:
    if not _cargo_available():
        print("❌  cargo not found. Install Rust from https://rustup.rs then re-run.")
        print("    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh")
        return 1

    gate_src = _find_gate_src()
    if gate_src is None:
        print("❌  credence_gate/ source directory not found.")
        print("    Clone the repo: git clone https://github.com/mrfentmen/credence-enforce")
        return 1

    print(f"Building credence-gate from {gate_src} ...")
    result = subprocess.run(
        ["cargo", "build", "--release"],
        cwd=gate_src,
    )
    if result.returncode != 0:
        print("❌  cargo build failed. See output above.")
        return result.returncode

    binary_name = "credence-gate.exe" if sys.platform == "win32" else "credence-gate"
    built = gate_src / "target" / "release" / binary_name
    if not built.exists():
        print(f"❌  Expected binary not found at {built}")
        return 1

    dest_dir  = _install_dir()
    dest      = dest_dir / binary_name
    shutil.copy2(built, dest)
    dest.chmod(0o755)

    print(f"✅  credence-gate installed → {dest}")
    print()
    print("Add to .claude/settings.json:")
    # Derived from ENFORCED_TOOLS so the installed config cannot disagree with
    # the list the gate actually enforces. Hand-written, it omitted MultiEdit —
    # a file-writing tool that then bypassed the gate.
    print(f"""  {{
    "hooks": {{
      "UserPromptSubmit": [{{
        "hooks": [{{"type": "command", "command": "python3 -m credence.observer"}}]
      }}],
      "PreToolUse": [{{
        "matcher": "{enforced_tool_matcher()}",
        "hooks": [{{"type": "command", "command": "credence-gate"}}]
      }}]
    }}
  }}

  UserPromptSubmit: passive observer — registers uncertain values automatically.
  PreToolUse: enforcement gate — blocks writes containing unverified values.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())

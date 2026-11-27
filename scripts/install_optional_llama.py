"""Optional Llama 3.2 setup via Ollama, on Windows, macOS, or Linux.
Author: Debanjali Ghosh
Run from the project root: python scripts/install_optional_llama.py
Requires Ollama to be installed separately. Does not modify the Qwen default.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys

MODEL = "llama3.2:3b"


def main() -> int:
    executable = shutil.which("ollama")
    if executable is None:
        system = platform.system()
        print("Ollama is not installed or is not on PATH.")
        if system == "Windows":
            print("Install Ollama from https://ollama.com/download/windows")
            print("Alternatively, if available: winget install Ollama.Ollama")
        elif system == "Darwin":
            print("Install Ollama from https://ollama.com/download/mac")
            print("Alternatively, with Homebrew: brew install --cask ollama")
        elif system == "Linux":
            print("Follow https://ollama.com/download/linux")
            print("Official installation command (review before running):")
            print("  curl -fsSL https://ollama.com/install.sh | sh")
        else:
            print("See https://ollama.com/download")
        print("After installation, start Ollama and rerun this script.")
        return 1

    print(f"Downloading optional local model {MODEL} via Ollama...")
    print("Ollama must be running; start the Ollama app or run 'ollama serve'.")
    try:
        subprocess.run([executable, "pull", MODEL], check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Model download failed (exit code {exc.returncode}).", file=sys.stderr)
        print("Check that Ollama is running, then rerun this script.", file=sys.stderr)
        return exc.returncode or 1
    print("Optional Llama model installed.")
    print("For local_llama_chat(), set OLLAMA_BASE_URL=http://localhost:11434")
    print(f"and OLLAMA_MODEL={MODEL} in your local .env file.")
    print("Hosted Qwen remains the default; no automatic fallback is enabled.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

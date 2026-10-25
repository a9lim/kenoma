"""Command line, config, and startup."""

from __future__ import annotations

import argparse
import os
import sys
import tomllib
from pathlib import Path
from typing import Any

from kenoma import __version__

DEFAULT_MODEL = "Qwen/Qwen3.5-4B-Base"

# Precedence: CLI flags > KENOMA_<KEY> env vars > config file > DEFAULTS.
CONFIG_KEYS: dict[str, type] = {
    "model": str,
    "url": str,
    "api_key": str,
    "bits": int,
    "temperature": float,
    "top_p": float,
    "max_new_tokens": int,
    "context_tokens": int,
    "prompt": str,
    "history": int,
    "tmux_lines": int,
    "ghost": bool,
}

DEFAULTS: dict[str, Any] = {
    "model": None,  # DEFAULT_MODEL locally; the server's first model with --url
    "url": None,
    "api_key": None,
    "bits": 4,
    "temperature": 1.0,
    "top_p": 0.95,
    "max_new_tokens": 2048,
    "context_tokens": 16384,
    "prompt": None,
    "history": 20,
    "tmux_lines": 300,
    "ghost": True,
}

ENV_PREFIX = "KENOMA_"


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / "kenoma" / "config.toml"


def _coerce(key: str, value: Any) -> Any:
    typ = CONFIG_KEYS[key]
    if typ is bool and isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "y", "on")
    return typ(value)


def load_config_file() -> dict[str, Any]:
    p = config_path()
    if not p.is_file():
        return {}
    try:
        with open(p, "rb") as f:
            data = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError) as e:
        print(f"[kenoma: failed to read {p}: {e}]", file=sys.stderr)
        return {}
    out: dict[str, Any] = {}
    for k, v in data.items():
        if k not in CONFIG_KEYS:
            print(f"[kenoma: unknown config key {k!r} in {p}, ignored]", file=sys.stderr)
            continue
        try:
            out[k] = _coerce(k, v)
        except (ValueError, TypeError):
            print(f"[kenoma: invalid {k}={v!r} in {p}, ignored]", file=sys.stderr)
    return out


def load_env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in CONFIG_KEYS:
        name = ENV_PREFIX + k.upper()
        if name in os.environ:
            try:
                out[k] = _coerce(k, os.environ[name])
            except (ValueError, TypeError):
                print(f"[kenoma: ignoring invalid {name}={os.environ[name]!r}]", file=sys.stderr)
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    d = {**DEFAULTS, **load_config_file(), **load_env_overrides()}
    ap = argparse.ArgumentParser(prog="kenoma", description="A shell powered by raw LLM completion.")
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    ap.add_argument("model", nargs="?", default=d["model"],
                    help=f"base model: HF id or local path for the MLX engine (default {DEFAULT_MODEL}), "
                         "or the served model name with --url")
    ap.add_argument("--url", default=d["url"],
                    help="OpenAI-compatible server (vLLM, SGLang, llama-server, mlx_lm.server) instead "
                         "of the local MLX engine, e.g. http://localhost:8000")
    ap.add_argument("--api-key", default=d["api_key"], help="bearer token for --url")
    ap.add_argument("--bits", type=int, choices=[3, 4, 5, 6, 8, 16], default=d["bits"],
                    help="quantize the local model's weights at load (16 = keep them as they are)")
    ap.add_argument("--temperature", type=float, default=d["temperature"])
    ap.add_argument("--top-p", type=float, default=d["top_p"])
    ap.add_argument("--max-new-tokens", type=int, default=d["max_new_tokens"],
                    help="per-turn cap on generated tokens")
    ap.add_argument("--context-tokens", type=int, default=d["context_tokens"],
                    help="transcript budget; past it the oldest turns are dropped down to half")
    ap.add_argument("--prompt", default=d["prompt"], help="use this prompt instead of your rendered PS1")
    ap.add_argument("--history", type=int, default=d["history"],
                    help="without tmux, seed with your last N shell commands (shown as `history` output)")
    ap.add_argument("--tmux-lines", type=int, default=d["tmux_lines"],
                    help="inside tmux, seed with the last N lines of the pane (real commands and outputs)")
    ap.add_argument("--no-ghost", dest="ghost", action="store_false", default=d["ghost"],
                    help="don't show the model's guess at your next command")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    from kenoma import shell

    # Capture the pane before kenoma prints anything into it.
    tmux_seed = shell.capture_tmux_pane(args.tmux_lines)

    from kenoma.engine import make_engine
    from kenoma.session import Session
    from kenoma.template import Template
    from kenoma.ui import UI

    raw = args.prompt if args.prompt is not None else shell.capture_prompt()
    if raw is None:
        raw = shell.fallback_prompt()
    cwd = os.getcwd()
    tmpl = Template.from_render(raw, cwd, os.path.expanduser("~"), shell.git_branch(cwd))
    if not tmpl.anchored:
        print("[kenoma: your prompt has no fixed text to recognize; using a default one. "
              "Pass --prompt to choose]", file=sys.stderr)
        tmpl = Template.from_render(shell.fallback_prompt(), cwd, os.path.expanduser("~"))
    if not tmpl.has_cwd:
        print("[kenoma: couldn't find the cwd in your prompt, so it won't follow a fake `cd`; "
              "pass --prompt to set one that shows it]", file=sys.stderr)

    if args.url:
        print(f"[kenoma: using {args.url}; your seed transcript is sent there]", file=sys.stderr)
    else:
        print(f"[kenoma: loading {args.model or DEFAULT_MODEL}]", file=sys.stderr)
    engine = make_engine(args.model or ("" if args.url else DEFAULT_MODEL), args.url,
                         args.temperature, args.top_p, args.api_key, args.bits)

    history = shell.read_history()
    if tmux_seed:
        seed = tmux_seed.rstrip("\n") + "\n"
    else:
        seed = shell.history_seed(tmpl.text(tmpl.values), history, args.history, cwd)
    session = Session(engine, tmpl, max_new_tokens=args.max_new_tokens,
                      context_tokens=args.context_tokens, ghost=args.ghost)
    session.start(seed)
    UI(session, history).run()


if __name__ == "__main__":
    main()

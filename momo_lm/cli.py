from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from .backend import get_backend
from .bootstrap import initialize_weights
from .config import MomoConfig
from .runtime import MomoRuntime
from .server import MomoServer


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="momo", description="Momo-LM local AI workbench")
    parser.add_argument("--home", type=Path, help="Runtime directory (default: ~/.momo-lm)")
    parser.add_argument("--config", type=Path, help="Path to config.json")
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="Start the local chat and training workbench")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument("--token", help="Visible-ASCII API token (or MOMO_ACCESS_TOKEN)")
    serve.add_argument(
        "--allow-host", action="append", default=[], help="Additional exact Host value"
    )
    serve.add_argument("--no-browser", action="store_true")

    chat = sub.add_parser("chat", help="Start terminal chat")
    chat.add_argument("message", nargs="*")

    train = sub.add_parser("train", help="Train from a UTF-8 text file")
    train.add_argument("file", type=Path)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=None)

    ingest = sub.add_parser("ingest", help="Add a text file to local retrieval memory")
    ingest.add_argument("file", type=Path)
    ingest.add_argument("--train", action="store_true")

    crawl = sub.add_parser("crawl", help="Learn from allowed same-site web pages")
    crawl.add_argument("url")
    crawl.add_argument("--max-pages", type=int, default=8)
    crawl.add_argument("--train", action="store_true")

    image = sub.add_parser("image", help="Generate a local abstract image")
    image.add_argument("prompt")
    image.add_argument("--output", type=Path, default=Path("momo-image.png"))
    image.add_argument("--width", type=int, default=512)
    image.add_argument("--height", type=int, default=512)
    image.add_argument("--seed", type=int)
    image.add_argument(
        "--style",
        choices=("anime", "manga", "illustration", "realistic"),
        default="illustration",
    )
    image.add_argument("--negative-prompt", default="")
    image.add_argument("--quality", choices=("draft", "standard", "high"), default="standard")
    image.add_argument("--steps", type=int)
    image.add_argument("--tile-size", type=int, default=128)

    speech = sub.add_parser("tts", help="Synthesize speech using an offline system voice")
    speech.add_argument("text")
    speech.add_argument("--output", type=Path, default=Path("momo-speech.wav"))
    speech.add_argument("--rate", type=int, default=170)

    sub.add_parser("inspect", help="Print model weights and runtime statistics")
    sub.add_parser("backend", help="Show the active tensor and inference backend")
    benchmark = sub.add_parser("benchmark", help="Benchmark the active matrix kernel")
    benchmark.add_argument("--size", type=int, default=256)
    benchmark.add_argument("--rounds", type=int, default=5)
    init = sub.add_parser("init", help="Create config and copy bundled starter weights")
    init.add_argument("--force", action="store_true")

    agent = sub.add_parser("agent", help="Run and manage persistent capability-limited agents")
    agent_sub = agent.add_subparsers(dest="agent_command", required=True)
    agent_run = agent_sub.add_parser("run", help="Create and run a deterministic local agent")
    agent_run.add_argument("goal")
    agent_run.add_argument(
        "--profile", choices=("training", "coding", "workplace", "copilot"), default="copilot"
    )
    agent_run.add_argument("--capability", action="append", default=[])
    agent_run.add_argument("--max-steps", type=int)
    agent_run.add_argument("--max-tool-calls", type=int)
    agent_run.add_argument("--max-input-chars", type=int)
    agent_list = agent_sub.add_parser("list", help="List saved agents")
    agent_list.add_argument("--limit", type=int, default=100)
    agent_status = agent_sub.add_parser("status", help="Show one saved agent")
    agent_status.add_argument("agent_id")
    agent_events = agent_sub.add_parser("events", help="Show append-only agent events")
    agent_events.add_argument("agent_id")
    agent_events.add_argument("--after", type=int, default=0)
    agent_events.add_argument("--limit", type=int, default=100)
    agent_approve = agent_sub.add_parser("approve", help="Consume one exact pending approval")
    agent_approve.add_argument("agent_id")
    agent_approve.add_argument("approval_id")
    agent_cancel = agent_sub.add_parser("cancel", help="Cancel a pending or running agent")
    agent_cancel.add_argument("agent_id")
    return parser


def _config(arguments: argparse.Namespace) -> MomoConfig:
    return MomoConfig.load(arguments.config, arguments.home)


def _terminal_chat(runtime: MomoRuntime, initial: str = "") -> None:
    if initial:
        print(runtime.chat(initial)["response"])
        return
    print("Momo-LM terminal chat. Type /exit to stop.")
    while True:
        try:
            message = input("You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message in {"/exit", "/quit"}:
            break
        if message:
            print(f"Momo > {runtime.chat(message)['response']}")


def _benchmark(size: int, rounds: int) -> dict[str, object]:
    size = max(16, min(size, 2048))
    rounds = max(1, min(rounds, 100))
    backend = get_backend()
    rng = np.random.default_rng(20260803)
    left = rng.normal(size=(size, size)).astype(np.float32)
    right = rng.normal(size=(size, size)).astype(np.float32)
    backend.matmul(left, right)
    started = time.perf_counter()
    for _ in range(rounds):
        backend.matmul(left, right)
    elapsed = time.perf_counter() - started
    operations = 2.0 * size**3 * rounds
    return {
        "backend": backend.describe(),
        "matrix": [size, size],
        "rounds": rounds,
        "seconds": elapsed,
        "gflops": operations / max(elapsed, 1e-9) / 1_000_000_000,
    }


def _agent_command(runtime: MomoRuntime, arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.agent_command == "run":
        budgets = {
            key: value
            for key, value in {
                "max_steps": arguments.max_steps,
                "max_tool_calls": arguments.max_tool_calls,
                "max_input_chars": arguments.max_input_chars,
            }.items()
            if value is not None
        }
        return runtime.create_agent(
            arguments.goal,
            profile=arguments.profile,
            capabilities=arguments.capability or None,
            budgets=budgets or None,
        )
    if arguments.agent_command == "list":
        return {"agents": runtime.list_agents(limit=arguments.limit)}
    if arguments.agent_command == "status":
        return {"agent": runtime.get_agent(arguments.agent_id)}
    if arguments.agent_command == "events":
        return {
            "events": runtime.agent_events(
                arguments.agent_id, after=arguments.after, limit=arguments.limit
            )
        }
    if arguments.agent_command == "approve":
        return {
            "agent": runtime.approve_agent(arguments.agent_id, arguments.approval_id)
        }
    if arguments.agent_command == "cancel":
        return {"agent": runtime.cancel_agent(arguments.agent_id)}
    raise ValueError("Unknown agent command")


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "init":
        config = _config(arguments)
        print(json.dumps(initialize_weights(config, force=arguments.force), ensure_ascii=False, indent=2))
        return 0
    if arguments.command == "backend":
        print(json.dumps(get_backend().describe(), ensure_ascii=False, indent=2))
        return 0
    if arguments.command == "benchmark":
        print(json.dumps(_benchmark(arguments.size, arguments.rounds), ensure_ascii=False, indent=2))
        return 0
    config = _config(arguments)
    if arguments.command in {None, "serve"}:
        token = getattr(arguments, "token", None)
        if token is not None:
            config.access_token = token
        allowed_hosts = getattr(arguments, "allow_host", [])
        if allowed_hosts:
            config.allowed_hosts = [*config.allowed_hosts, *allowed_hosts]
        config.validate()
    runtime = MomoRuntime(config)
    try:
        if arguments.command in {None, "serve"}:
            MomoServer(runtime, getattr(arguments, "host", None), getattr(arguments, "port", None)).serve(
                open_browser=not getattr(arguments, "no_browser", False)
            )
        elif arguments.command == "chat":
            _terminal_chat(runtime, " ".join(arguments.message))
        elif arguments.command in {"train", "ingest"}:
            text = arguments.file.read_text(encoding="utf-8")
            if arguments.command == "train":
                result = runtime.train(text, epochs=arguments.epochs, learning_rate=arguments.learning_rate, source=str(arguments.file))
            else:
                result = runtime.ingest(text, source=str(arguments.file), train=arguments.train)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif arguments.command == "crawl":
            print(json.dumps(runtime.crawl(arguments.url, max_pages=arguments.max_pages, train=arguments.train), ensure_ascii=False, indent=2))
        elif arguments.command == "image":
            output = runtime.generate_image(
                arguments.prompt,
                arguments.output,
                width=arguments.width,
                height=arguments.height,
                seed=arguments.seed,
                style=arguments.style,
                negative_prompt=arguments.negative_prompt,
                quality=arguments.quality,
                steps=arguments.steps,
                tile_size=arguments.tile_size,
            )
            print(output.resolve())
        elif arguments.command == "tts":
            print(json.dumps(runtime.speech.synthesize(arguments.text, arguments.output, rate=arguments.rate), ensure_ascii=False, indent=2))
        elif arguments.command == "inspect":
            print(json.dumps(runtime.status(), ensure_ascii=False, indent=2))
        elif arguments.command == "agent":
            print(json.dumps(_agent_command(runtime, arguments), ensure_ascii=False, indent=2))
        return 0
    except (KeyError, OSError, PermissionError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

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
    serve.add_argument("--no-browser", action="store_true")

    chat = sub.add_parser("chat", help="Start terminal chat")
    chat.add_argument("message", nargs="*")

    train = sub.add_parser("train", help="Train from a UTF-8 text file")
    train.add_argument("file", type=Path)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--learning-rate", type=float, default=0.025)

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
            output = runtime.generate_image(arguments.prompt, arguments.output, width=arguments.width, height=arguments.height, seed=arguments.seed)
            print(output.resolve())
        elif arguments.command == "tts":
            print(json.dumps(runtime.speech.synthesize(arguments.text, arguments.output, rate=arguments.rate), ensure_ascii=False, indent=2))
        elif arguments.command == "inspect":
            print(json.dumps(runtime.status(), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())

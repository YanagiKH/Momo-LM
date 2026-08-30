from __future__ import annotations

import re
import threading
from pathlib import Path
from typing import Any

from .agent import AgentManager
from .agent_store import AgentStore
from .agent_tools import AgentToolbox
from .bootstrap import initialize_weights
from .config import MomoConfig, validate_training_rate
from .image_model import TinyCanvasModel
from .knowledge import KnowledgeStore
from .learner import Learner
from .model import NeuralTextModel
from .mods import ModManager
from .paths import package_root
from .speech import OfflineSpeech
from .version import __version__


class MomoRuntime:
    def __init__(self, config: MomoConfig | None = None) -> None:
        self.config = config or MomoConfig.load()
        self.config.validate()
        initialize_weights(self.config)
        self.model = NeuralTextModel.load(self.config.model_path)
        self.image_model = TinyCanvasModel.load(self.config.image_model_path)
        self._model_lock = threading.RLock()
        self.store = KnowledgeStore(self.config.database_path)
        self._seed_starter_knowledge()
        self.learner = Learner(self.model, self.store)
        self.mods = ModManager(self.config.mods_path)
        self.mods.load()
        self.speech = OfflineSpeech()
        self.agent_store = AgentStore(self.config.agent_database_path)
        self.agent_tools = AgentToolbox(self, self.config.agent_workspace_path)
        self.agents = AgentManager(
            self.agent_store,
            self.agent_tools,
            approval_ttl_seconds=self.config.agent_approval_ttl_seconds,
            recover=False,
        )
        self.agents.recover()

    def _seed_starter_knowledge(self) -> None:
        if self.store.stats()["documents"]:
            return
        corpus_path = package_root() / "assets" / "corpus" / "starter.txt"
        if not corpus_path.exists():
            return
        lines = [line.strip() for line in corpus_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for index in range(0, len(lines) - 1, 2):
            if lines[index].startswith("User:") and lines[index + 1].startswith("Momo:"):
                self.store.add_document("momo-starter-corpus", f"{lines[index]}\n{lines[index + 1]}")

    def chat(self, text: str, *, learn: bool | None = None) -> dict[str, Any]:
        original = text.strip()
        if not original:
            raise ValueError("Message is empty")
        if len(original) > 12_000:
            raise ValueError("Message exceeds 12,000 characters")
        transformed = self.mods.process_before(original)
        mod_response = self.mods.command(transformed)
        hits = self.store.search(transformed, limit=3)
        response = mod_response or self._compose_response(transformed, hits)
        response = self.mods.process_after(transformed, response)
        self.store.add_turn("user", original)
        self.store.add_turn("assistant", response)
        should_learn = self.config.self_learning if learn is None else learn
        if should_learn:
            with self._model_lock:
                self.model.train_text(
                    f"User: {original}\nMomo: {response}\n",
                    epochs=1,
                    learning_rate=self.config.learning_rate * 0.2,
                    batch_size=96,
                )
                self.model.save(self.config.model_path)
        return {
            "response": response,
            "learned": bool(should_learn),
            "sources": [hit.source for hit in hits],
        }

    def _compose_response(self, text: str, hits: list[Any]) -> str:
        lower = text.lower()
        if any(word in lower for word in ("hello", "hi", "你好", "哈囉", "こんにちは", "嗨")):
            return "你好，我是 Momo。這次想一起研究、建立或訓練什麼內容？"
        if self._needs_clarification(text):
            return "我可以處理這個方向。請補充目標輸出、可用資料或限制條件中的任一項，我會依它繼續推進。"
        if hits:
            best = hits[0].content
            if hits[0].source == "momo-starter-corpus" and "Momo:" in best:
                return best.split("Momo:", 1)[1].strip()
            evidence = " ".join(hit.content for hit in hits)[:900]
            return f"根據已學習的本機資料：{evidence}"
        prompt = f"User: {text}\nMomo:"
        with self._model_lock:
            generated = self.model.generate(
                prompt,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                top_k=self.config.top_k,
            )
        generated = self._clean_generation(generated)
        if len(generated) >= 8:
            return generated
        return "我目前沒有足夠的已學習資料能可靠回答。可以把相關文字貼到「學習資料」頁籤，或提供允許讀取的網址。"

    @staticmethod
    def _needs_clarification(text: str) -> bool:
        vague = {"這個", "那個", "幫我弄", "處理一下", "do it", "fix it", "これ", "それ"}
        return len(text) < 24 and any(item in text.lower() for item in vague)

    @staticmethod
    def _clean_generation(text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
        text = text.split("\nUser:", 1)[0].strip()
        printable = sum(character.isprintable() for character in text)
        if not text or printable / max(len(text), 1) < 0.92:
            return ""
        return text[:1200]

    def train(self, text: str, *, epochs: int = 3, learning_rate: float | None = None, source: str = "manual") -> dict[str, Any]:
        resolved_rate = validate_training_rate(
            self.config.learning_rate if learning_rate is None else learning_rate
        )
        with self._model_lock:
            result = self.learner.ingest_text(
                text,
                source,
                train=True,
                epochs=max(1, min(epochs, 100)),
                learning_rate=resolved_rate,
            )
            self.model.save(self.config.model_path)
        return result

    def ingest(self, text: str, *, source: str = "manual", train: bool = False) -> dict[str, Any]:
        with self._model_lock:
            result = self.learner.ingest_text(text, source, train=train, learning_rate=self.config.learning_rate)
            if train:
                self.model.save(self.config.model_path)
        return result

    def crawl(self, url: str, *, max_pages: int | None = None, train: bool = False) -> dict[str, Any]:
        with self._model_lock:
            result = self.learner.crawl(
                url,
                max_pages=max_pages or self.config.max_crawl_pages,
                timeout=self.config.request_timeout,
                train=train,
            )
            if train:
                self.model.save(self.config.model_path)
        return result

    def generate_image(
        self,
        prompt: str,
        output: Path,
        *,
        width: int = 512,
        height: int = 512,
        seed: int | None = None,
        style: str = "illustration",
        negative_prompt: str = "",
        quality: str = "standard",
        steps: int | None = None,
        tile_size: int = 128,
    ) -> Path:
        image = self.image_model.generate(
            prompt,
            width,
            height,
            seed,
            style=style,
            negative_prompt=negative_prompt,
            quality=quality,
            steps=steps,
            tile_size=tile_size,
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="PNG")
        return output

    def status(self) -> dict[str, Any]:
        agent_stats = self.agent_store.stats()
        return {
            "version": __version__,
            "home": str(self.config.home),
            "compute_backend": self.model.backend.describe(),
            "weights": self.model.inspect(),
            "image_weights": self.image_model.inspect(),
            "knowledge": self.store.stats(),
            "mods": self.mods.status(),
            "self_learning": self.config.self_learning,
            "agents": {
                **agent_stats,
                "profiles": self.agents.profiles(),
                "tools": self.agent_tools.describe(),
                "persistence": {"journal_mode": self.agent_store.journal_mode},
            },
        }

    def create_agent(
        self,
        goal: str,
        *,
        profile: str = "copilot",
        capabilities: list[str] | None = None,
        budgets: dict[str, Any] | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        return self.agents.create(
            goal,
            profile=profile,
            capabilities=capabilities,
            budgets=budgets,
            background=background,
        )

    def list_agents(self, *, limit: int = 100) -> list[dict[str, Any]]:
        return self.agent_store.list_agents(limit=limit)

    def get_agent(self, agent_id: str) -> dict[str, Any]:
        return self.agent_store.get_agent(agent_id)

    def cancel_agent(self, agent_id: str) -> dict[str, Any]:
        return self.agents.cancel(agent_id)

    def approve_agent(
        self, agent_id: str, approval_id: str, *, background: bool = False
    ) -> dict[str, Any]:
        return self.agents.approve(agent_id, approval_id, background=background)

    def agent_events(
        self, agent_id: str, *, after: int = 0, limit: int = 100
    ) -> list[dict[str, Any]]:
        return self.agent_store.events(agent_id, after=after, limit=limit)

    def reload_mods(self) -> dict[str, Any]:
        self.mods.load()
        return self.mods.status()

    def close(self) -> None:
        self.agents.close()
        self.agent_store.close()
        self.store.close()

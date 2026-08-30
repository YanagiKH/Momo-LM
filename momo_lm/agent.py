from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

from .agent_store import TERMINAL_STATUSES, AgentStore
from .agent_tools import TOOL_SPECS, AgentToolbox

DEFAULT_BUDGETS = {"max_steps": 8, "max_tool_calls": 8, "max_input_chars": 12_000}
MIN_BUDGETS = {"max_steps": 1, "max_tool_calls": 0, "max_input_chars": 256}
MAX_BUDGETS = {"max_steps": 128, "max_tool_calls": 128, "max_input_chars": 1_000_000}


@dataclass(frozen=True, slots=True)
class AgentProfile:
    name: str
    default_capabilities: frozenset[str]
    allowed_capabilities: frozenset[str]
    description: str


READ_CAPABILITIES = frozenset({"knowledge.read", "runtime.inspect", "workspace.read"})
PROFILE_DEFINITIONS: dict[str, AgentProfile] = {
    "training": AgentProfile(
        "training",
        frozenset({"knowledge.read", "runtime.inspect"}),
        frozenset({"knowledge.read", "runtime.inspect", "model.train"}),
        "Inspect local weights, prepare training, and optionally train with approval.",
    ),
    "coding": AgentProfile(
        "coding",
        frozenset({"knowledge.read", "runtime.inspect", "workspace.read"}),
        frozenset(
            {"knowledge.read", "runtime.inspect", "workspace.read", "workspace.write"}
        ),
        "Read an isolated workspace and optionally write one approved text file.",
    ),
    "workplace": AgentProfile(
        "workplace",
        frozenset({"knowledge.read"}),
        frozenset({"knowledge.read", "workspace.read", "workspace.write"}),
        "Draft local workplace material without sending or publishing it.",
    ),
    "copilot": AgentProfile(
        "copilot",
        READ_CAPABILITIES,
        frozenset(
            {
                "knowledge.read",
                "runtime.inspect",
                "workspace.read",
                "workspace.write",
                "model.train",
            }
        ),
        "Combine local inspection, knowledge, and an isolated workspace.",
    ),
}

TRAIN_PATTERN = re.compile(r"^train\s*:\s*(.+)$", re.IGNORECASE | re.DOTALL)
READ_PATTERN = re.compile(r"^read\s*:\s*([^\r\n]+)\s*$", re.IGNORECASE)
WRITE_PATTERN = re.compile(
    r"^write\s*:\s*([^\r\n]+)[\r\n]+([\s\S]+)$", re.IGNORECASE
)


class DeterministicPlanner:
    """Create a stable plan from explicit, deliberately small command forms."""

    @staticmethod
    def normalize_budgets(values: dict[str, Any] | None) -> dict[str, int]:
        budgets = dict(DEFAULT_BUDGETS)
        if values is None:
            return budgets
        if not isinstance(values, dict):
            raise ValueError("budgets must be an object")
        unknown = set(values) - set(DEFAULT_BUDGETS)
        if unknown:
            raise ValueError(f"Unknown budget: {sorted(unknown)[0]}")
        for key, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"Budget {key} must be an integer")
            if value < MIN_BUDGETS[key] or value > MAX_BUDGETS[key]:
                raise ValueError(
                    f"Budget {key} must be between {MIN_BUDGETS[key]} and {MAX_BUDGETS[key]}"
                )
            budgets[key] = value
        return budgets

    @staticmethod
    def normalize_capabilities(profile: str, requested: list[str] | None) -> list[str]:
        definition = PROFILE_DEFINITIONS.get(profile)
        if definition is None:
            raise ValueError(f"Unknown agent profile: {profile}")
        capabilities = set(definition.default_capabilities)
        if requested is not None:
            if not isinstance(requested, list) or not all(
                isinstance(value, str) for value in requested
            ):
                raise ValueError("capabilities must be a list of strings")
            capabilities.update(requested)
        denied = capabilities - set(definition.allowed_capabilities)
        if denied:
            raise ValueError(
                f"Capability {sorted(denied)[0]} is not allowed for profile {profile}"
            )
        return sorted(capabilities)

    def plan(
        self,
        goal: str,
        *,
        profile: str,
        capabilities: list[str] | None = None,
        budgets: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
        cleaned = goal.strip()
        if not cleaned:
            raise ValueError("Agent goal is empty")
        normalized_budgets = self.normalize_budgets(budgets)
        if len(cleaned) > normalized_budgets["max_input_chars"]:
            raise ValueError("Agent goal exceeds max_input_chars")
        normalized_capabilities = self.normalize_capabilities(profile, capabilities)

        if profile == "training":
            plan = self._training_plan(cleaned)
        elif profile == "coding":
            plan = self._coding_plan(cleaned)
        elif profile == "workplace":
            plan = self._workplace_plan(cleaned)
        else:
            plan = self._copilot_plan(cleaned)
        if len(plan) > normalized_budgets["max_steps"]:
            raise ValueError("Planned work exceeds max_steps")
        capability_set = set(normalized_capabilities)
        for step in plan:
            spec = TOOL_SPECS[step["tool"]]
            if spec.capability not in capability_set:
                raise ValueError(
                    f"Explicit capability {spec.capability} is required for this goal"
                )
        return plan, normalized_capabilities, normalized_budgets

    @staticmethod
    def _step(tool: str, arguments: dict[str, Any], purpose: str) -> dict[str, Any]:
        return {"tool": tool, "arguments": arguments, "purpose": purpose}

    def _training_plan(self, goal: str) -> list[dict[str, Any]]:
        plan = [self._step("inspect_runtime", {}, "Inspect the current local checkpoint")]
        match = TRAIN_PATTERN.fullmatch(goal)
        if match:
            plan.append(
                self._step(
                    "train_text",
                    {"text": match.group(1).strip(), "epochs": 1, "source": "agent-training"},
                    "Train only the exact supplied text after explicit approval",
                )
            )
        else:
            plan.append(
                self._step(
                    "training_guidance", {"goal": goal}, "Prepare a local training checklist"
                )
            )
        return plan

    def _coding_plan(self, goal: str) -> list[dict[str, Any]]:
        read_match = READ_PATTERN.fullmatch(goal)
        if read_match:
            return [
                self._step(
                    "read_text_file",
                    {"path": read_match.group(1).strip()},
                    "Read one explicitly named workspace file",
                )
            ]
        write_match = WRITE_PATTERN.fullmatch(goal)
        if write_match:
            return [
                self._step("list_files", {"path": ".", "limit": 100}, "Inspect workspace"),
                self._step(
                    "write_text_file",
                    {"path": write_match.group(1).strip(), "content": write_match.group(2)},
                    "Write exactly one file after explicit approval",
                ),
            ]
        return [
            self._step("list_files", {"path": ".", "limit": 100}, "Inspect workspace"),
            self._step(
                "draft_text",
                {"goal": goal, "profile": "coding"},
                "Produce a local implementation brief",
            ),
        ]

    def _workplace_plan(self, goal: str) -> list[dict[str, Any]]:
        return [
            self._step(
                "search_knowledge", {"query": goal, "limit": 4}, "Find local context"
            ),
            self._step(
                "draft_text",
                {"goal": goal, "profile": "workplace"},
                "Draft locally without sending or publishing",
            ),
        ]

    def _copilot_plan(self, goal: str) -> list[dict[str, Any]]:
        return [
            self._step("inspect_runtime", {}, "Inspect local runtime"),
            self._step(
                "search_knowledge", {"query": goal, "limit": 4}, "Find local context"
            ),
            self._step(
                "draft_text",
                {"goal": goal, "profile": "copilot"},
                "Produce a deterministic local brief",
            ),
        ]


class AgentManager:
    def __init__(
        self,
        store: AgentStore,
        toolbox: AgentToolbox,
        *,
        approval_ttl_seconds: int = 900,
        recover: bool = True,
    ) -> None:
        self.store = store
        self.toolbox = toolbox
        self.planner = DeterministicPlanner()
        self.approval_ttl_seconds = max(30, min(int(approval_ttl_seconds), 86_400))
        self._lock = threading.RLock()
        self._threads: dict[str, threading.Thread] = {}
        if recover:
            self.recover()

    def profiles(self) -> list[dict[str, Any]]:
        return [
            {
                "name": profile.name,
                "description": profile.description,
                "default_capabilities": sorted(profile.default_capabilities),
                "allowed_capabilities": sorted(profile.allowed_capabilities),
            }
            for profile in PROFILE_DEFINITIONS.values()
        ]

    def create(
        self,
        goal: str,
        *,
        profile: str = "copilot",
        capabilities: list[str] | None = None,
        budgets: dict[str, Any] | None = None,
        background: bool = False,
    ) -> dict[str, Any]:
        plan, normalized_capabilities, normalized_budgets = self.planner.plan(
            goal,
            profile=profile,
            capabilities=capabilities,
            budgets=budgets,
        )
        agent = self.store.create_agent(
            goal=goal.strip(),
            profile=profile,
            capabilities=normalized_capabilities,
            budgets=normalized_budgets,
            plan=plan,
        )
        if background:
            self.submit(agent["id"])
            return self.store.get_agent(agent["id"])
        return self.run(agent["id"])

    def run(self, agent_id: str) -> dict[str, Any]:
        with self._lock:
            current = self._threads.get(agent_id)
            if current is not None and current.is_alive() and current is not threading.current_thread():
                return self.store.get_agent(agent_id)
        while True:
            agent = self.store.internal_agent(agent_id)
            if agent["status"] in TERMINAL_STATUSES or agent["status"] == "waiting_approval":
                return self.store.get_agent(agent_id)
            if self.store.is_cancel_requested(agent_id):
                return self.store.request_cancel(agent_id)
            step_index = agent["next_step"]
            plan = agent["plan"]
            if step_index >= len(plan):
                return self.store.fail(agent_id, "Agent plan ended without a completion state")
            budgets = agent["budgets"]
            if agent["steps_used"] >= budgets["max_steps"]:
                return self.store.fail(agent_id, "Agent exceeded max_steps")
            if agent["tool_calls_used"] >= budgets["max_tool_calls"]:
                return self.store.fail(agent_id, "Agent exceeded max_tool_calls")
            step = plan[step_index]
            tool = str(step.get("tool", ""))
            arguments = step.get("arguments", {})
            if not isinstance(arguments, dict):
                return self.store.fail(agent_id, "Planner produced invalid tool arguments")
            spec = TOOL_SPECS.get(tool)
            if spec is None:
                return self.store.fail(agent_id, f"Planner selected unknown tool: {tool}")
            if spec.capability not in set(agent["capabilities"]):
                return self.store.fail(agent_id, f"Missing capability: {spec.capability}")
            if spec.mutating and not self.store.has_consumed_approval(
                agent_id, step_index=step_index, tool=tool, arguments=arguments
            ):
                return self.store.create_approval(
                    agent_id,
                    step_index=step_index,
                    tool=tool,
                    arguments=arguments,
                    reason=str(step.get("purpose", "Mutating local state")),
                    ttl_seconds=self.approval_ttl_seconds,
                )
            try:
                self.store.set_running(agent_id, step_index)
                output = self.toolbox.execute(
                    tool, arguments, capabilities=set(agent["capabilities"])
                )
                record = self.store.complete_step(
                    agent_id, {"step": step_index, "tool": tool, "result": output}
                )
            except Exception as exc:
                return self.store.fail(agent_id, f"{type(exc).__name__}: {exc}")
            if record["status"] in TERMINAL_STATUSES:
                return record

    def submit(self, agent_id: str) -> dict[str, Any]:
        with self._lock:
            existing = self._threads.get(agent_id)
            if existing is None or not existing.is_alive():
                thread = threading.Thread(
                    target=self._background_run,
                    args=(agent_id,),
                    daemon=True,
                    name=f"momo-agent-{agent_id[:8]}",
                )
                self._threads[agent_id] = thread
                thread.start()
        return self.store.get_agent(agent_id)

    def _background_run(self, agent_id: str) -> None:
        try:
            self.run(agent_id)
        finally:
            with self._lock:
                current = self._threads.get(agent_id)
                if current is threading.current_thread():
                    self._threads.pop(agent_id, None)

    def approve(
        self, agent_id: str, approval_id: str, *, background: bool = False
    ) -> dict[str, Any]:
        public = self.store.get_agent(agent_id)
        pending = public.get("pending_approval")
        if public["status"] != "waiting_approval" or not isinstance(pending, dict):
            raise ValueError("Agent is not waiting for approval")
        if pending["id"] != approval_id:
            raise ValueError("Approval does not match this agent's pending action")
        agent = self.store.internal_agent(agent_id)
        step_index = agent["next_step"]
        step = agent["plan"][step_index]
        self.store.consume_approval(
            agent_id,
            approval_id,
            tool=step["tool"],
            arguments=step["arguments"],
            step_index=step_index,
        )
        if background:
            return self.submit(agent_id)
        return self.run(agent_id)

    def cancel(self, agent_id: str) -> dict[str, Any]:
        return self.store.request_cancel(agent_id)

    def recover(self) -> None:
        for agent in self.store.recoverable_agents():
            if agent["status"] == "running" and agent["active_step"] is not None:
                step = agent["plan"][agent["active_step"]]
                spec = TOOL_SPECS.get(str(step.get("tool", "")))
                if spec is None or spec.mutating:
                    self.store.fail(
                        agent["id"],
                        "Interrupted during a mutating or unknown tool; action was not replayed",
                        event_type="recovery_failed",
                    )
                    continue
                self.store.reset_pending(
                    agent["id"], message="Interrupted read-only step scheduled for safe replay"
                )
            self.submit(agent["id"])

    def close(self) -> None:
        with self._lock:
            threads = list(self._threads.values())
        for thread in threads:
            thread.join()

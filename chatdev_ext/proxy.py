from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# A minimal, in-process "network" proxy for agent messages.
# It supports: drop, delay, duplicate, reorder, corrupt
# Policy can be static (env/JSON) or programmatic (callable).

@dataclass
class ProxyConfig:
    # probabilities in [0,1]
    p_drop: float = 0.0
    p_delay: float = 0.0
    p_dup: float = 0.0
    p_reorder: float = 0.0
    p_corrupt: float = 0.0
    # delay bounds (seconds)
    delay_min_s: float = 0.1
    delay_max_s: float = 0.5
    # corruption: simple char-level perturbation
    corrupt_max_chars: int = 12
    # deterministic runs
    seed: int | None = None

    @staticmethod
    def from_env(prefix: str = "CHATDEV_PROXY_") -> "ProxyConfig":
        def g(name, default):
            v = os.environ.get(prefix + name, None)
            if v is None:
                return default
            try:
                return type(default)(v)
            except Exception:
                return default

        cfg_json = os.environ.get(prefix + "JSON", None)
        if cfg_json:
            try:
                d = json.loads(cfg_json)
                return ProxyConfig(**d)
            except Exception:
                pass

        return ProxyConfig(
            p_drop=float(os.environ.get(prefix + "P_DROP", 0.0) or 0.0),
            p_delay=float(os.environ.get(prefix + "P_DELAY", 0.0) or 0.0),
            p_dup=float(os.environ.get(prefix + "P_DUP", 0.0) or 0.0),
            p_reorder=float(os.environ.get(prefix + "P_REORDER", 0.0) or 0.0),
            p_corrupt=float(os.environ.get(prefix + "P_CORRUPT", 0.0) or 0.0),
            delay_min_s=float(os.environ.get(prefix + "DELAY_MIN_S", 0.1) or 0.1),
            delay_max_s=float(os.environ.get(prefix + "DELAY_MAX_S", 0.5) or 0.5),
            corrupt_max_chars=int(os.environ.get(prefix + "CORRUPT_MAX_CHARS", 12) or 12),
            seed=int(os.environ.get(prefix + "SEED", "0")) if os.environ.get(prefix + "SEED") else None,
        )


def _force_text(msg: Any) -> str:
    if msg is None:
        return ""
    try:
        c = getattr(msg, "content", None)
        if isinstance(c, str):
            return c
        if isinstance(msg, str):
            return msg
        return str(msg)
    except Exception:
        return ""


def _apply_corruption(text: str, k: int) -> str:
    if not text or k <= 0:
        return text
    text = list(text)
    n = min(k, len(text))
    for _ in range(n):
        i = random.randrange(len(text))
        # simple bitflip-ish: random printable char
        text[i] = chr(random.randrange(32, 127))
    return "".join(text)


class MessageProxy:
    """
    Lightweight proxy queue per (phase, turn, sender->receiver) channel.
    For reorder: we buffer and occasionally release out of order.
    """
    def __init__(self, config: ProxyConfig | None = None):
        self.config = config or ProxyConfig.from_env()
        if self.config.seed is not None:
            random.seed(self.config.seed)
        # a tiny buffer to enable reordering
        self._buffer: List[Any] = []

    def _maybe_delay(self):
        if random.random() < self.config.p_delay:
            d = random.uniform(self.config.delay_min_s, self.config.delay_max_s)
            time.sleep(d)
            return True, d
        return False, 0.0

    def _maybe_dup(self) -> bool:
        return random.random() < self.config.p_dup

    def _maybe_drop(self) -> bool:
        return random.random() < self.config.p_drop

    def _maybe_reorder(self) -> bool:
        return random.random() < self.config.p_reorder

    def _maybe_corrupt(self, text: str) -> Tuple[bool, str]:
        if random.random() < self.config.p_corrupt:
            k = random.randint(1, self.config.corrupt_max_chars)
            return True, _apply_corruption(text, k)
        return False, text

    def transmit(self, msg_obj: Any, ctx: Dict[str, Any]) -> Tuple[List[Any], Dict[str, Any]]:
        """
        Takes a single outgoing message, returns a list of one or more messages
        to deliver (after applying faults), plus metadata.
        """
        meta = {
            "fault_injected": False,
            "faults": [],           # list of strings
            "delay_seconds": 0.0,
            "duplicated": False,
            "dropped": False,
            "reordered": False,
            "corrupted": False,
            "fault_type": "none",   # summary
        }

        # DROP
        if self._maybe_drop():
            meta.update({"fault_injected": True, "dropped": True, "faults": ["drop"], "fault_type": "drop"})
            return [], meta  # nothing to deliver

        # DELAY
        delayed, dsec = self._maybe_delay()
        if delayed:
            meta["fault_injected"] = True
            meta["delay_seconds"] = dsec
            meta["faults"].append("delay")

        # DUPLICATE
        deliver_count = 2 if self._maybe_dup() else 1
        if deliver_count == 2:
            meta["fault_injected"] = True
            meta["duplicated"] = True
            meta["faults"].append("duplicate")

        # CORRUPT (apply to each copy independently)
        text = _force_text(msg_obj)
        deliver_list: List[Any] = []
        for _ in range(deliver_count):
            corrupted, new_text = self._maybe_corrupt(text)
            if corrupted:
                meta["fault_injected"] = True
                meta["corrupted"] = True
                if "corrupt" not in meta["faults"]:
                    meta["faults"].append("corrupt")
            # Try to preserve the original object type if it has 'content'
            cloned = msg_obj
            try:
                if hasattr(cloned, "content"):
                    cloned = type(cloned)(**{**getattr(cloned, "__dict__", {}), "content": new_text})
                else:
                    cloned = new_text
            except Exception:
                cloned = new_text
            deliver_list.append(cloned)

        # REORDER (buffer + release out-of-order occasionally)
        if self._maybe_reorder():
            self._buffer.extend(deliver_list)
            meta["fault_injected"] = True
            meta["reordered"] = True
            meta["faults"].append("reorder")
            # release buffer in reverse order for a crude "reordering" effect
            deliver_list = list(reversed(self._buffer))
            self._buffer.clear()

        # summary fault_type
        if meta["faults"]:
            meta["fault_type"] = "+".join(meta["faults"])

        return deliver_list, meta

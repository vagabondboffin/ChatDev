# chatdev/instrumentation.py
"""
ALFIT OpenTelemetry instrumentation for ChatDev.

This module installs hierarchical spans:

- BatchRun             around run_batch_tasks.main / run_ollama_simple.main
- Task/*               around chatdev.chat_chain.ChatChain.execute_chain(...)
- Phase/*              around chatdev.phase.Phase.run/chatting/execute
- LLMCall/*            around ChatAgent.step(...)
- Msg/<role>           around message input/output per LLM call

It is activated purely by importing `chatdev.instrumentation`.
"""

from __future__ import annotations

import functools
import importlib
import logging
import os
import sys
from typing import Any, Callable, Tuple, Dict

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# Single logger for everything here
_logger = logging.getLogger("chatdev.instrumentation")

# Global tracer handle
_tracer = None  # type: ignore[assignment]
_configured = False


# ---------------------------------------------------------------------------
# Tracer / OTEL setup
# ---------------------------------------------------------------------------

def _ensure_tracer():
    """Configure the global tracer provider once and return a tracer."""
    global _tracer, _configured

    if _tracer is not None:
        return _tracer

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

    resource = Resource.create(
        {
            "service.name": "chatdev",
            "service.namespace": "alfit",
        }
    )
    provider = TracerProvider(resource=resource)

    # OTLP → Tempo / collector
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    _logger.info("[ALFIT] OTLPSpanExporter enabled: %s", endpoint)

    # Optional console exporter for debugging
    if os.getenv("CHATDEV_OTEL_CONSOLE", "0") == "1":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        _logger.info("[ALFIT] ConsoleSpanExporter enabled")

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer("chatdev")
    _configured = True

    _logger.info("Initialized OpenTelemetry tracer provider for service 'chatdev'")
    return _tracer


def get_tracer():
    """Public helper in case other modules want the tracer."""
    return _ensure_tracer()


# ---------------------------------------------------------------------------
# Generic helpers for wrapping functions & methods
# ---------------------------------------------------------------------------

SpanNameFn = Callable[[Tuple[Any, ...], Dict[str, Any]], str]


def _wrap_function(module_name: str, func_name: str, span_name_fn: SpanNameFn) -> None:
    """Wrap a top-level function with a span."""
    try:
        module = importlib.import_module(module_name)
        original = getattr(module, func_name)
    except Exception:
        _logger.debug("Instrumentation: %s.%s not found", module_name, func_name, exc_info=True)
        return

    if getattr(original, "_alfit_wrapped", False):
        return

    tracer = _ensure_tracer()

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        span_name = span_name_fn(args, kwargs)
        with tracer.start_as_current_span(span_name) as span:
            # Extra attributes for batch entrypoint
            if module_name == "run_batch_tasks" and func_name == "main":
                span.set_attribute("batch.entrypoint", "run_batch_tasks.main")
                span.set_attribute("batch.argv", " ".join(sys.argv[1:]))
                span.set_attribute("batch.cwd", os.getcwd())
            return original(*args, **kwargs)

    wrapper._alfit_wrapped = True  # type: ignore[attr-defined]
    setattr(module, func_name, wrapper)
    _logger.info("[ALFIT] Wrapped %s.%s → %s", module_name, func_name, span_name_fn.__name__)


def _wrap_method(
    module_name: str,
    class_name: str,
    method_name: str,
    span_name_fn: SpanNameFn,
) -> None:
    """Wrap an instance method with a span."""
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        original = getattr(cls, method_name)
    except Exception:
        _logger.debug(
            "Instrumentation: %s.%s.%s not found",
            module_name,
            class_name,
            method_name,
            exc_info=True,
        )
        return

    if getattr(original, "_alfit_wrapped", False):
        return

    tracer = _ensure_tracer()

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        span_name = span_name_fn(args, kwargs)
        with tracer.start_as_current_span(span_name) as span:
            self_obj = args[0] if args else None

            # Enrich Task spans with task name if we can
            if module_name == "chatdev.chat_chain" and class_name == "ChatChain":
                task_name = getattr(self_obj, "project_name", None) or getattr(self_obj, "task_name", None)
                if task_name:
                    span.set_attribute("task.name", str(task_name))

            # Enrich Phase spans with phase name
            if module_name == "chatdev.phase" and class_name == "Phase":
                phase_name = getattr(self_obj, "name", None) or getattr(self_obj, "phase_name", None)
                if phase_name:
                    span.set_attribute("phase.name", str(phase_name))

            return original(*args, **kwargs)

    wrapper._alfit_wrapped = True  # type: ignore[attr-defined]
    setattr(cls, method_name, wrapper)
    _logger.info("[ALFIT] Wrapped %s.%s.%s", module_name, class_name, method_name)


# ---------------------------------------------------------------------------
# Special wrapper for ChatAgent.step → LLMCall + Msg/user + Msg/assistant
# ---------------------------------------------------------------------------

def _wrap_chatagent_step(module_name: str) -> None:
    """Wrap ChatAgent.step in the given module with LLMCall + Msg spans."""
    try:
        module = importlib.import_module(module_name)
        ChatAgent = getattr(module, "ChatAgent")
        original = getattr(ChatAgent, "step")
    except Exception:
        _logger.debug("Instrumentation: %s.ChatAgent.step not found", module_name, exc_info=True)
        return

    if getattr(original, "_alfit_wrapped", False):
        return

    tracer = _ensure_tracer()

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        role = getattr(self, "role_name", None) or getattr(self, "name", None) or self.__class__.__name__
        span_name = f"LLMCall/{role}"

        with tracer.start_as_current_span(span_name) as llm_span:
            # Try to grab the incoming message (best-effort)
            msg_in = args[0] if args else None
            if msg_in is not None:
                content_in = getattr(msg_in, "content", None)
                role_in = getattr(msg_in, "role_name", None) or getattr(msg_in, "role", None) or "user"

                if content_in:
                    # Attribute on the LLM span
                    llm_span.set_attribute("llm.input", str(content_in)[:1024])

                    # Nested Msg/<role_in> span
                    with tracer.start_as_current_span(f"Msg/{role_in}") as msg_span:
                        msg_span.set_attribute("chat.role", str(role_in))
                        msg_span.set_attribute("chat.content", str(content_in)[:1024])

            # Call the real LLM step
            result = original(self, *args, **kwargs)

            # Try to capture the output message
            if result is not None:
                content_out = getattr(result, "content", None)
                role_out = getattr(result, "role_name", None) or getattr(result, "role", None) or "assistant"

                if content_out:
                    llm_span.set_attribute("llm.output", str(content_out)[:2048])

                    with tracer.start_as_current_span(f"Msg/{role_out}") as msg_span:
                        msg_span.set_attribute("chat.role", str(role_out))
                        msg_span.set_attribute("chat.content", str(content_out)[:2048])

            # Some extra metadata on the LLM span
            llm_span.set_attribute("llm.role", str(role))
            model_attr = getattr(self, "model_type", None)
            if model_attr:
                llm_span.set_attribute("llm.model", str(model_attr))

            backend = getattr(self, "model_backend", None)
            if backend is not None and hasattr(backend, "model_type"):
                try:
                    llm_span.set_attribute("llm.backend_model", str(getattr(backend, "model_type", "")))
                except Exception:
                    pass

            return result

    wrapper._alfit_wrapped = True  # type: ignore[attr-defined]
    setattr(ChatAgent, "step", wrapper)
    _logger.info("[ALFIT] Wrapped %s.ChatAgent.step → LLMCall/* + Msg/*", module_name)


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

def _install_wrappers() -> None:
    """Install all instrumentation hooks. Safe to call multiple times."""
    _ensure_tracer()

    # 1) Batch-level (run_batch_tasks / run_ollama_simple)
    _wrap_function("run_batch_tasks", "main", lambda *_: "BatchRun")
    _wrap_function("run_ollama_simple", "main", lambda *_: "BatchRun")

    # 2) Task-level: ChatChain.execute_chain (one per ChatDev project)
    def _task_span_name(args, kwargs):
        self_obj = args[0] if args else None
        task_name = None
        if self_obj is not None:
            task_name = getattr(self_obj, "project_name", None) or getattr(self_obj, "task_name", None)
            # Fallback if there's a 'task' dict
            if not task_name and hasattr(self_obj, "task"):
                task = getattr(self_obj, "task", None)
                if isinstance(task, dict):
                    task_name = task.get("name") or task.get("id") or task.get("task_id")
        return f"Task/{task_name}" if task_name else "Task"

    _wrap_method("chatdev.chat_chain", "ChatChain", "execute_chain", _task_span_name)

    # 3) Phase-level: various Phase methods
    def _phase_span_name(args, kwargs):
        self_obj = args[0] if args else None
        phase_name = None
        if self_obj is not None:
            phase_name = getattr(self_obj, "name", None) or getattr(self_obj, "phase_name", None)
        return f"Phase/{phase_name}" if phase_name else "Phase"

    for method in ("run", "execute", "chatting"):
        _wrap_method("chatdev.phase", "Phase", method, _phase_span_name)

    # 4) LLM calls: ChatAgent.step in both camel and chatdev
    for mod_name in ("camel.agents.chat_agent", "chatdev.chat_agent"):
        _wrap_chatagent_step(mod_name)

    _logger.info("[ALFIT] Wrapper installation complete.")


# Run once on import
_install_wrappers()

"""
OpenTelemetry bootstrap + ChatDev instrumentation for ALFIT.

Usage (recommended):
    PYTHONPATH=. \
    CHATDEV_OTEL_CONSOLE=1 \
    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318/v1/traces \
    python -c "import chatdev_ext.instrumentation_bootstrap; import run_batch_tasks as m; m.main()" \
      --json-file programdev_dataset.json

Environment variables:
    OTEL_EXPORTER_OTLP_ENDPOINT   OTLP endpoint (e.g. http://localhost:4318/v1/traces)
    CHATDEV_OTEL_CONSOLE          "1"/"true" to also log spans to console
    CHATDEV_OTEL_SERVICE_NAME     Optional override for service.name (default: "chatdev")
    CHATDEV_BATCH_ID              Optional batch id tag for BatchRun spans
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Any, Iterable

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger("chatdev.instrumentation")

if not logger.handlers:
    _handler = logging.StreamHandler()
    _formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _handler.setFormatter(_formatter)
    logger.addHandler(_handler)

# Default to INFO so you see wrapper-installation logs.
if logger.level == logging.NOTSET:
    logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

_TRACING_INSTALLED = False
_ENTRYPOINTS_PATCHED = False
_WRAPPERS_INSTALLED = False


# ---------------------------------------------------------------------------
# Helper: safe attribute setting (no NoneType warnings)
# ---------------------------------------------------------------------------

def _set_attr(span: trace.Span, key: str, value: Any) -> None:
    """
    Safely set OTEL span attribute, skipping None and coercing unsupported types.
    Prevents opentelemetry 'Invalid type NoneType' / serialization warnings.
    """
    if span is None:
        return
    if value is None:
        return

    # Primitive types are accepted directly.
    if isinstance(value, (bool, str, bytes, int, float)):
        span.set_attribute(key, value)
        return

    # Sequences: filter / coerce each element.
    if isinstance(value, (list, tuple)):
        cleaned: list[Any] = []
        for v in value:
            if v is None:
                continue
            if isinstance(v, (bool, str, bytes, int, float)):
                cleaned.append(v)
            else:
                cleaned.append(str(v))
        if cleaned:
            span.set_attribute(key, cleaned)
        return

    # Fallback: string representation.
    span.set_attribute(key, str(value))


# ---------------------------------------------------------------------------
# Tracer provider setup
# ---------------------------------------------------------------------------

def _setup_tracing() -> TracerProvider:
    global _TRACING_INSTALLED
    if _TRACING_INSTALLED:
        return trace.get_tracer_provider()  # type: ignore[return-value]

    service_name = os.getenv("CHATDEV_OTEL_SERVICE_NAME", "chatdev")

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": "alfit",
            # mildly helpful for multi-machine setups
            "service.instance.id": os.getenv("HOSTNAME")
            or os.getenv("COMPUTERNAME")
            or "",
        }
    )

    provider = TracerProvider(resource=resource)

    # Exporters
    exporters: list[Any] = []

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if endpoint:
        try:
            otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
            exporters.append(BatchSpanProcessor(otlp_exporter))
            logger.info("[ALFIT] OTLPSpanExporter enabled: %s", endpoint)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("[ALFIT] Failed to init OTLP exporter (%s)", exc)

    console_flag = os.getenv("CHATDEV_OTEL_CONSOLE", "").lower()
    if console_flag in ("1", "true", "yes"):
        exporters.append(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info("[ALFIT] ConsoleSpanExporter enabled")

    # If no exporter configured at all, default to console so something happens.
    if not exporters:
        exporters.append(SimpleSpanProcessor(ConsoleSpanExporter()))
        logger.info(
            "[ALFIT] No OTEL exporter configured; defaulting to ConsoleSpanExporter"
        )

    for p in exporters:
        provider.add_span_processor(p)

    trace.set_tracer_provider(provider)
    _TRACING_INSTALLED = True

    logger.info(
        "Initialized OpenTelemetry tracer provider for service '%s'", service_name
    )
    return provider


# ---------------------------------------------------------------------------
# Entry point patching: BatchRun
# ---------------------------------------------------------------------------

def _patch_entrypoints() -> None:
    """
    Wrap known ChatDev entrypoints (run_batch_tasks, run_ollama_simple) in a BatchRun span.
    """
    global _ENTRYPOINTS_PATCHED
    if _ENTRYPOINTS_PATCHED:
        return

    tracer = trace.get_tracer("chatdev.instrumentation")

    def _wrap_module_main(module_name: str) -> None:
        try:
            mod = __import__(module_name)
        except ImportError:
            # Not all repos have all entrypoints; that's fine.
            return

        main = getattr(mod, "main", None)
        if not callable(main):
            return

        def wrapped_main(*args: Any, **kwargs: Any) -> Any:
            # Ensure lower-level wrappers are installed before main executes.
            _install_chatdev_wrappers()
            argv = " ".join(sys.argv[1:])
            batch_id = os.getenv("CHATDEV_BATCH_ID") or None

            with tracer.start_as_current_span("BatchRun") as span:
                _set_attr(span, "batch.entrypoint", f"{module_name}.main")
                _set_attr(span, "batch.argv", argv)
                _set_attr(span, "batch.cwd", os.getcwd())
                _set_attr(span, "batch.id", batch_id)
                return main(*args, **kwargs)

        setattr(mod, "main", wrapped_main)
        logger.info("[ALFIT] Wrapped %s.main → BatchRun", module_name)

    # Known ChatDev entrypoints
    for mod_name in ("run_batch_tasks", "run_ollama_simple"):
        _wrap_module_main(mod_name)

    _ENTRYPOINTS_PATCHED = True


# ---------------------------------------------------------------------------
# ChatDev & Camel wrappers: Task / Phase / LLMCall
# ---------------------------------------------------------------------------

def _install_chatdev_wrappers() -> None:
    """
    Install wrappers around ChatDev & Camel primitives:

      - ChatChain.execute_chain → Task/*
      - Art.execute, CodeCompleteAll.execute, CodeReview.execute, ComposedPhase.execute → Phase/*
      - camel.agents.chat_agent.ChatAgent.step → LLMCall + Msg/*
    """
    global _WRAPPERS_INSTALLED
    if _WRAPPERS_INSTALLED:
        return
    _WRAPPERS_INSTALLED = True

    tracer = trace.get_tracer("chatdev.instrumentation")
    logger.info("[ALFIT] Applying ChatDev wrappers for observability hierarchy…")

    # ------------------------------
    # Task-level: ChatChain.execute_chain
    # ------------------------------
    try:
        import chatdev.chat_chain as chat_chain_mod  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(
            "[ALFIT] Could not import chatdev.chat_chain; skipping Task/* wrapper (%s)",
            exc,
        )
    else:
        ChatChain = getattr(chat_chain_mod, "ChatChain", None)
        if ChatChain is not None and hasattr(ChatChain, "execute_chain"):
            original_execute_chain = ChatChain.execute_chain

            def execute_chain_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                # Try to collect some metadata from the ChatChain instance.
                project_name = getattr(self, "project_name", None)
                task_prompt = getattr(self, "task_prompt", None)

                with tracer.start_as_current_span("Task/Execute") as span:
                    _set_attr(span, "task.name", project_name)
                    _set_attr(span, "task.prompt", task_prompt)
                    _set_attr(span, "task.class", type(self).__name__)

                    result = original_execute_chain(self, *args, **kwargs)

                    # If chat_env is available later, we can attach status-ish info.
                    chat_env = getattr(self, "chat_env", None)
                    status = getattr(chat_env, "status", None) if chat_env else None
                    _set_attr(span, "task.status", status)

                    return result

            ChatChain.execute_chain = execute_chain_wrapper  # type: ignore[assignment]
            logger.info("[ALFIT] Wrapped ChatChain.execute_chain → Task/*")

    # ------------------------------
    # Phase-level: Art, CodeCompleteAll, CodeReview, ComposedPhase
    # ------------------------------
    try:
        import chatdev.phase as phase_mod  # type: ignore[import]
    except ImportError:
        logger.info("[ALFIT] chatdev.phase not found; skipping Phase/* wrappers")
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("[ALFIT] Error importing chatdev.phase (%s)", exc)
    else:
        def _wrap_phase_class(cls_name: str) -> None:
            cls = getattr(phase_mod, cls_name, None)
            if cls is None:
                return
            if not hasattr(cls, "execute"):
                return

            original_execute = cls.execute

            def phase_execute(self, *args: Any, **kwargs: Any) -> Any:
                # ChatDev usually puts ChatEnv as first arg; be defensive.
                chat_env = args[0] if args else None

                phase_name = getattr(self, "phase_name", None) or cls.__name__
                project_name = (
                    getattr(chat_env, "project_name", None) if chat_env else None
                )
                task_prompt = (
                    getattr(chat_env, "task_prompt", None) if chat_env else None
                )

                span_name = f"Phase/{phase_name}"

                with tracer.start_as_current_span(span_name) as span:
                    _set_attr(span, "phase.name", phase_name)
                    _set_attr(span, "phase.class", cls.__name__)
                    _set_attr(span, "task.name", project_name)
                    _set_attr(span, "task.prompt", task_prompt)

                    return original_execute(self, *args, **kwargs)

            cls.execute = phase_execute  # type: ignore[assignment]
            logger.info("[ALFIT] Wrapped %s.execute → Phase/*", cls_name)

        for _cls_name in ("Art", "CodeCompleteAll", "CodeReview", "ComposedPhase"):
            _wrap_phase_class(_cls_name)

    # ------------------------------
    # LLMCall-level: Camel ChatAgent.step
    # ------------------------------
    try:
        from camel.agents.chat_agent import ChatAgent  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - defensive
        logger.info(
            "[ALFIT] Could not import camel.agents.chat_agent.ChatAgent; "
            "skipping LLMCall/* wrapper (%s)",
            exc,
        )
    else:
        if hasattr(ChatAgent, "step"):
            original_step = ChatAgent.step

            def llm_step_wrapper(self, *args: Any, **kwargs: Any) -> Any:
                # Try to recover model name from backend or the agent itself.
                model_name = None
                backend = getattr(self, "model_backend", None)
                if backend is not None:
                    model_name = getattr(backend, "model_name", None) or getattr(
                        backend, "model", None
                    )
                if model_name is None:
                    model_name = getattr(self, "model_name", None)

                role_name = getattr(self, "role_name", None) or getattr(
                    self, "assistant_role_name", None
                )

                with tracer.start_as_current_span("LLMCall") as span:
                    _set_attr(span, "llm.model", model_name)
                    _set_attr(span, "agent.role", role_name)
                    _set_attr(span, "agent.class", type(self).__name__)

                    # --------------------
                    # Input message (first positional arg, typically a Message)
                    # --------------------
                    if args:
                        msg = args[0]
                        in_role = None
                        in_content = None

                        try:
                            in_role = getattr(msg, "role", None)
                        except Exception:
                            pass
                        if in_role is None:
                            try:
                                in_role = msg.get("role")  # type: ignore[call-arg]
                            except Exception:
                                pass

                        try:
                            in_content = getattr(msg, "content", None)
                        except Exception:
                            pass
                        if in_content is None:
                            try:
                                in_content = msg.get("content")  # type: ignore[call-arg]
                            except Exception:
                                pass

                        _set_attr(span, "msg.input.role", in_role)
                        if isinstance(in_content, str):
                            _set_attr(span, "msg.input.length", len(in_content))

                    # --------------------
                    # Call original step
                    # --------------------
                    result = original_step(self, *args, **kwargs)

                    # --------------------
                    # Output message (best-effort)
                    # --------------------
                    try:
                        out_msg = None

                        if hasattr(result, "msgs"):
                            msgs = result.msgs  # type: ignore[attr-defined]
                            if isinstance(msgs, (list, tuple)) and msgs:
                                out_msg = msgs[0]
                        elif isinstance(result, (list, tuple)) and result:
                            out_msg = result[0]
                        else:
                            out_msg = result

                        out_role = None
                        out_content = None

                        if out_msg is not None:
                            try:
                                out_role = getattr(out_msg, "role", None)
                            except Exception:
                                pass
                            if out_role is None:
                                try:
                                    out_role = getattr(out_msg, "sender_role", None)
                                except Exception:
                                    pass

                            try:
                                out_content = getattr(out_msg, "content", None)
                            except Exception:
                                pass

                        _set_attr(span, "msg.output.role", out_role)
                        if isinstance(out_content, str):
                            _set_attr(span, "msg.output.length", len(out_content))
                    except Exception:
                        # Never break ChatDev because of tracing.
                        pass

                    return result

            ChatAgent.step = llm_step_wrapper  # type: ignore[assignment]
            logger.info("[ALFIT] Wrapped ChatAgent.step → LLMCall/* + Msg/*")

    logger.info("[ALFIT] Wrapper installation complete.")


# ---------------------------------------------------------------------------
# Public bootstrap API
# ---------------------------------------------------------------------------

def bootstrap() -> None:
    """
    Install tracing + wrappers.

    Safe to call multiple times. Typically you don't call this manually;
    importing this module from your entrypoint is enough.
    """
    _setup_tracing()
    _patch_entrypoints()


# Run bootstrap on import so your one-liner
#   python -c "import chatdev_ext.instrumentation_bootstrap; import run_batch_tasks as m; m.main()"
# just works.
bootstrap()


import os
import logging
from contextvars import ContextVar
from itertools import count

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace.export import ConsoleSpanExporter

logger = logging.getLogger("chatdev.telemetry")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)


# ---- Context for hierarchy -------------------------------------------------

# Current logical task (e.g., "TicTacToe")
CURRENT_TASK: ContextVar[str | None] = ContextVar("chatdev.current_task", default=None)

# Current phase name (e.g., "Coding")
CURRENT_PHASE: ContextVar[str | None] = ContextVar("chatdev.current_phase", default=None)

# Current conversation turn ID (per batch run, monotonically increasing)
CURRENT_CONV_ID: ContextVar[int | None] = ContextVar("chatdev.current_conversation_id", default=None)

# Global counter for conversation turns
_CONV_COUNTER = count(1)


def next_conversation_id() -> int:
    """Return a new global conversation ID."""
    return next(_CONV_COUNTER)


# ---- OpenTelemetry bootstrap -----------------------------------------------

_OTEL_INITIALIZED = False


def init_otel(service_name: str = "chatdev") -> None:
    """
    Initialize OpenTelemetry tracer provider if not yet initialized.

    Uses OTLP HTTP exporter by default. You can configure:

    - OTEL_EXPORTER_OTLP_ENDPOINT (e.g. http://localhost:4318/v1/traces)
    - CHATDEV_OTEL_CONSOLE=1 to also mirror spans to console.
    """
    global _OTEL_INITIALIZED

    if _OTEL_INITIALIZED:
        return

    # If someone already set a non-default provider, don't override it.
    if isinstance(trace.get_tracer_provider(), TracerProvider):
        _OTEL_INITIALIZED = True
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    if os.getenv("CHATDEV_OTEL_CONSOLE") == "1":
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _OTEL_INITIALIZED = True
    logger.info("Initialized OpenTelemetry tracer provider for service '%s'", service_name)


def tracer():
    """Convenience accessor for the tracer used across ChatDev."""
    return trace.get_tracer("chatdev.tracer")

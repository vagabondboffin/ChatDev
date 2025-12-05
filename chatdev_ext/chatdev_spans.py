"""
Extra OpenTelemetry spans for ChatDev using Ollama.

This file assumes that `chatdev_ext.instrumentation_bootstrap` has already:
- set up the OpenTelemetry tracer provider
- configured OTLP + console exporters

Here we ONLY:
- wrap `ollama.chat` to emit LLMCall + Msg spans
"""

import json
import logging
from collections.abc import Iterator

from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = logging.getLogger("chatdev.instrumentation")

# Use the existing tracer provider that instrumentation_bootstrap already configured
tracer = trace.get_tracer("chatdev")


def _serialize_messages(messages):
    """Best-effort serialization of Ollama messages to a string."""
    if messages is None:
        return ""
    try:
        return json.dumps(messages, ensure_ascii=False)
    except Exception:
        return str(messages)


def _install_ollama_chat_wrapper():
    try:
        import ollama  # type: ignore
    except ImportError:
        logger.warning("[ALFIT] ollama package not available; skipping LLM instrumentation")
        return

    original_chat = getattr(ollama, "chat", None)
    if original_chat is None:
        logger.warning("[ALFIT] ollama.chat missing; skipping LLM instrumentation")
        return

    # Avoid double-wrapping
    if getattr(original_chat, "_alfit_wrapped", False):
        logger.info("[ALFIT] ollama.chat already wrapped; skipping")
        return

    def wrapped_chat(*args, **kwargs):
        # Support both positional and keyword usage:
        #   ollama.chat(model, messages, ...)
        #   ollama.chat(model="...", messages=[...], ...)
        model = kwargs.get("model")
        messages = kwargs.get("messages")

        if model is None and args:
            model = args[0]
        if messages is None and len(args) > 1:
            messages = args[1]

        stream = bool(kwargs.get("stream", False))

        messages_repr = _serialize_messages(messages)

        with tracer.start_as_current_span(
            "LLMCall/Ollama",
            kind=SpanKind.CLIENT,
        ) as span:
            # High-level LLM call attributes
            span.set_attribute("llm.provider", "ollama")
            if model:
                span.set_attribute("llm.model", model)
            span.set_attribute("llm.stream", stream)
            if messages_repr:
                span.set_attribute("llm.input_messages", messages_repr)

            # Child span for the prompt
            if messages_repr:
                with tracer.start_as_current_span("Msg/prompt") as msg_span:
                    msg_span.set_attribute("msg.role", "user")
                    msg_span.set_attribute("msg.content", messages_repr)

            # Call the real Ollama client
            try:
                response = original_chat(*args, **kwargs)
            except Exception as exc:
                # Record the exception on the LLMCall span but don't swallow it
                span.record_exception(exc)
                raise

            # ---- Non-streaming path -------------------------------------------------
            if not isinstance(response, Iterator):
                completion_text = None
                try:
                    # Standard ollama.chat response: {"message": {"role": "...", "content": "..."}}
                    message = response.get("message") or {}
                    completion_text = message.get("content")
                except Exception:
                    completion_text = None

                if completion_text:
                    span.set_attribute("llm.output_text", completion_text)
                    with tracer.start_as_current_span("Msg/completion") as msg_span:
                        msg_span.set_attribute("msg.role", "assistant")
                        msg_span.set_attribute("msg.content", completion_text)

                return response

            # ---- Streaming path -----------------------------------------------------
            # If ChatDev ever uses `stream=True`, we wrap the iterator
            def _streaming_wrapper(iterator):
                chunks = []

                try:
                    for chunk in iterator:
                        try:
                            msg = chunk.get("message") or {}
                            text_piece = msg.get("content") or ""
                        except Exception:
                            text_piece = ""
                        if text_piece:
                            chunks.append(text_piece)
                        # Yield the original chunk unchanged
                        yield chunk
                finally:
                    # Once the stream ends, record the full completion
                    completion_text = "".join(chunks)
                    if completion_text:
                        span.set_attribute("llm.output_text", completion_text)
                        with tracer.start_as_current_span("Msg/completion") as msg_span:
                            msg_span.set_attribute("msg.role", "assistant")
                            msg_span.set_attribute("msg.content", completion_text)

            return _streaming_wrapper(response)

    # Mark wrapper and install
    wrapped_chat._alfit_wrapped = True  # type: ignore[attr-defined]
    import ollama  # re-import to be sure we patch the same module object
    ollama.chat = wrapped_chat  # type: ignore[assignment]

    logger.info("[ALFIT] Wrapped ollama.chat → LLMCall/Ollama + Msg/*")


# Install immediately when this module is imported
_install_ollama_chat_wrapper()
logger.info("[ALFIT] chatdev_spans: ollama-based LLM tracing initialized")

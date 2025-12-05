"""
Additional LLM-level OpenTelemetry instrumentation for ChatDev.

This module wraps the *effective* chat completion API that ChatDev uses.
Given your logs, ChatDev is going through an OpenAI-compatible interface
that has already been patched to talk to Ollama.

We stay agnostic about *how* that patch works and simply wrap
`openai.ChatCompletion.create`, which is the likely entrypoint.
"""

import json
import logging
from collections.abc import Iterator

from opentelemetry import trace
from opentelemetry.trace import SpanKind

logger = logging.getLogger("chatdev.instrumentation")
tracer = trace.get_tracer("chatdev")


def _serialize_messages(messages) -> str:
    if messages is None:
        return ""
    try:
        return json.dumps(messages, ensure_ascii=False)
    except Exception:
        return str(messages)


def _wrap_openai_chatcompletion_create():
    try:
        import openai  # type: ignore
    except ImportError:
        logger.warning("[ALFIT] openai not available; skipping LLM ChatCompletion instrumentation")
        return

    ChatCompletion = getattr(openai, "ChatCompletion", None)
    if ChatCompletion is None:
        logger.warning("[ALFIT] openai.ChatCompletion missing; skipping LLM ChatCompletion instrumentation")
        return

    original_create = getattr(ChatCompletion, "create", None)
    if original_create is None:
        logger.warning("[ALFIT] openai.ChatCompletion.create missing; skipping")
        return

    if getattr(original_create, "_alfit_wrapped", False):
        logger.info("[ALFIT] openai.ChatCompletion.create already wrapped; skipping")
        return

    def wrapped_create(*args, **kwargs):
        """
        Wrap ChatCompletion.create with an OTEL span.

        We don't assume exact signature; we just best-effort extract:
        - model
        - messages
        - stream flag
        """
        model = kwargs.get("model")
        messages = kwargs.get("messages")
        stream = bool(kwargs.get("stream", False))

        # If model/messages came in positionally, try to guess:
        #   ChatCompletion.create(model, messages, ...)
        if model is None and args:
            model = args[0]
        if messages is None and len(args) > 1:
            messages = args[1]

        messages_repr = _serialize_messages(messages)

        with tracer.start_as_current_span(
            "LLMCall/OpenAICompat",
            kind=SpanKind.CLIENT,
        ) as span:
            span.set_attribute("llm.provider", "openai-compat")
            if model:
                span.set_attribute("llm.model", str(model))
            span.set_attribute("llm.stream", stream)
            if messages_repr:
                span.set_attribute("llm.input_messages", messages_repr)

            # Child span for prompt
            if messages_repr:
                with tracer.start_as_current_span("Msg/prompt") as msg_span:
                    msg_span.set_attribute("msg.role", "user")
                    msg_span.set_attribute("msg.content", messages_repr)

            try:
                response = original_create(*args, **kwargs)
            except Exception as exc:
                span.record_exception(exc)
                raise

            # Non-streaming responses: dict-ish or OpenAI objects
            if not stream and not isinstance(response, Iterator):
                completion_text = None
                try:
                    # Typical dict form: {"choices": [{"message": {"role": "...", "content": "..."}}]}
                    if isinstance(response, dict):
                        choices = response.get("choices") or []
                        if choices:
                            msg = choices[0].get("message") or {}
                            completion_text = msg.get("content")
                    else:
                        # OpenAI object-ish form
                        choices = getattr(response, "choices", None)
                        if choices:
                            first = choices[0]
                            message = getattr(first, "message", None)
                            if message is not None:
                                completion_text = getattr(message, "content", None)
                except Exception:
                    completion_text = None

                if completion_text:
                    span.set_attribute("llm.output_text", completion_text)
                    with tracer.start_as_current_span("Msg/completion") as msg_span:
                        msg_span.set_attribute("msg.role", "assistant")
                        msg_span.set_attribute("msg.content", completion_text)

                return response

            # Streaming responses (iterator of chunks)
            def _streaming_wrapper(iterator):
                chunks = []

                try:
                    for chunk in iterator:
                        text_piece = ""
                        try:
                            if isinstance(chunk, dict):
                                choices = chunk.get("choices") or []
                                if choices:
                                    delta = choices[0].get("delta") or {}
                                    text_piece = delta.get("content") or ""
                            else:
                                # Object-like streaming chunk
                                choices = getattr(chunk, "choices", None) or []
                                if choices:
                                    delta = getattr(choices[0], "delta", None)
                                    if delta is not None:
                                        text_piece = getattr(delta, "content", "") or ""
                        except Exception:
                            text_piece = ""

                        if text_piece:
                            chunks.append(text_piece)

                        # Yield original chunk unchanged
                        yield chunk
                finally:
                    completion_text = "".join(chunks)
                    if completion_text:
                        span.set_attribute("llm.output_text", completion_text)
                        with tracer.start_as_current_span("Msg/completion") as msg_span:
                            msg_span.set_attribute("msg.role", "assistant")
                            msg_span.set_attribute("msg.content", completion_text)

            # If the patched backend returns an iterator, wrap it
            if isinstance(response, Iterator):
                return _streaming_wrapper(response)

            # Fallback: just return whatever it is
            return response

    wrapped_create._alfit_wrapped = True  # type: ignore[attr-defined]
    ChatCompletion.create = wrapped_create
    logger.info("[ALFIT] Wrapped openai.ChatCompletion.create → LLMCall/OpenAICompat + Msg/*")


# Install immediately when this module is imported
_wrap_openai_chatcompletion_create()

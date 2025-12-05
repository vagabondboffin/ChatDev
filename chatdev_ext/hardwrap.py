# chatdev_ext/hardwrap.py
import os
import types
import time
import importlib
import inspect
from contextlib import contextmanager

# ---------- OpenTelemetry setup ----------
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter as OTLPSpanExporterGRPC
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter as OTLPSpanExporterHTTP

DEBUG = os.getenv("CHATDEV_OTEL_DEBUG", "0") == "1"

def _log(msg):
    if DEBUG:
        print(f"[ALFIT-HARDWRAP] {msg}", flush=True)

def init_otel(service_name: str = "chatdev"):
    # choose protocol automatically if not set
    protocol = os.getenv("OTEL_EXPORTER_OTLP_PROTOCOL", "grpc").lower()
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")
    resource = Resource.create({"service.name": service_name})
    provider = TracerProvider(resource=resource)
    if protocol == "http" or endpoint.endswith("/v1/traces"):
        exporter = OTLPSpanExporterHTTP(endpoint=endpoint)
    else:
        exporter = OTLPSpanExporterGRPC(endpoint=endpoint, insecure=True)
    provider.add_span_processor(BatchSpanProcessor(exporter))

    if os.getenv("CHATDEV_OTEL_CONSOLE", "0") == "1":
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _log(f"OTel initialized (service={service_name}, protocol={protocol}, endpoint={endpoint})")

def tracer():
    return trace.get_tracer("chatdev.hardwrap")

@contextmanager
def span(name: str, attrs: dict | None = None):
    with tracer().start_as_current_span(name) as sp:
        if attrs:
            for k, v in attrs.items():
                sp.set_attribute(k, v)
        t0 = time.time()
        try:
            yield sp
            sp.set_status(Status(StatusCode.UNSET))
        except Exception as e:
            sp.record_exception(e)
            sp.set_status(Status(StatusCode.ERROR, str(e)))
            raise
        finally:
            sp.set_attribute("duration_ms", round((time.time() - t0)*1000, 2))

# ---------- generic method wrapper ----------
def wrap_method(obj, method_name, span_name_fn):
    if not hasattr(obj, method_name):
        _log(f"Missing {obj}.{method_name} — skip")
        return False

    orig = getattr(obj, method_name)
    if not callable(orig):
        _log(f"{obj}.{method_name} not callable — skip")
        return False

    if getattr(orig, "__alfit_wrapped__", False):
        _log(f"{obj}.{method_name} already wrapped")
        return True

    def wrapped(*args, **kwargs):
        try:
            n = span_name_fn(args, kwargs)
        except Exception:
            n = f"{obj.__name__}.{method_name}"
        with span(n):
            _log(f"ENTER {n}")
            return orig(*args, **kwargs)
    wrapped.__alfit_wrapped__ = True
    setattr(obj, method_name, wrapped)
    _log(f"Wrapped {obj.__name__}.{method_name}")
    return True

# ---------- attempt to import modules and wrap likely callpoints ----------
def apply_all_wrappers():
    # 0) root "BatchRun" span around the batch (optional; your run_* already prints)
    #    we keep bootstrap spans minimal.

    # 1) ChatDev phases
    for mod_name, cls_name in [
        ("chatdev.phase", "Phase"),
        ("chatdev.composed_phase", "ComposedPhase"),
    ]:
        try:
            m = importlib.import_module(mod_name)
            C = getattr(m, cls_name, None)
            if C is None:
                _log(f"{mod_name}.{cls_name} not found")
                continue

            # Phase.execute(self)
            def _phase_span_name(args, _):
                self = args[0]
                # many forks store .name or .phase_name
                pname = getattr(self, "name", None) or getattr(self, "phase_name", None) or cls_name
                return f"Phase/{pname}"
            wrap_method(C, "execute", _phase_span_name)

            # Phase.run(self) — some forks use run instead of execute
            wrap_method(C, "run", _phase_span_name)

        except Exception as e:
            _log(f"wrap {mod_name}.{cls_name} failed: {e}")

    # 2) Chat chain (root orchestration)
    try:
        m = importlib.import_module("chatdev.chat_chain")
        Ch = getattr(m, "ChatChain", None)
        if Ch:
            wrap_method(Ch, "run", lambda a, k: "Run/ChatChain")
    except Exception as e:
        _log(f"wrap chatdev.chat_chain.ChatChain.run failed: {e}")

    # 3) CAMEL agents – ChatAgent.step / TaskAgent.step / CriticAgent.step
    for mod_name, classes in [
        ("camel.agents.chat_agent", ["ChatAgent"]),
        ("camel.agents.task_agent", ["TaskAgent"]),
        ("camel.agents.critic_agent", ["CriticAgent"]),
        ("camel.agents.role_playing", ["RolePlaying"]),
    ]:
        try:
            m = importlib.import_module(mod_name)
            for cname in classes:
                C = getattr(m, cname, None)
                if C is None:
                    continue

                # step(self, msg) or step(self, *args)
                def _llm_span_name(args, _):
                    self = args[0]
                    role = getattr(self, "role_name", None) or getattr(self, "name", None) or cname
                    return f"LLMCall/{role}"
                wrap_method(C, "step", _llm_span_name)

                # some forks have chat loop method names
                wrap_method(C, "chat", lambda a, k: f"LLMChat/{cname}")
                wrap_method(C, "init_chat", lambda a, k: f"LLMInit/{cname}")

        except Exception as e:
            _log(f"wrap {mod_name} failed: {e}")

    # 4) Message send/receive path (camel.messages.chat_messages)
    try:
        m = importlib.import_module("camel.messages.chat_messages")
        # common helpers: to_openai_message / from_openai_message / add_message / get_recent_messages
        for fname in ["add_message", "to_openai_message", "from_openai_message", "get_recent_messages"]:
            if hasattr(m, fname):
                fn = getattr(m, fname)
                if not getattr(fn, "__alfit_wrapped__", False) and callable(fn):
                    def make_wrap(name, fn_ref):
                        def inner(*args, **kwargs):
                            with span(f"Msg/{name}"):
                                _log(f"ENTER Msg/{name}")
                                return fn_ref(*args, **kwargs)
                        inner.__alfit_wrapped__ = True
                        return inner
                    setattr(m, fname, make_wrap(fname, fn))
                    _log(f"Wrapped camel.messages.chat_messages.{fname}")
    except Exception as e:
        _log(f"wrap camel.messages.chat_messages failed: {e}")

def bootstrap():
    if os.getenv("CHATDEV_OTEL_AUTOSTART", "0") != "1":
        return
    init_otel(os.getenv("CHATDEV_OTEL_SERVICE", "chatdev"))
    apply_all_wrappers()
    _log("Bootstrap complete.")

# Executed on import
bootstrap()

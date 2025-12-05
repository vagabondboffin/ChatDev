"""
Auto-load ChatDev instrumentation in every Python process.

Python automatically imports `site`, which then tries to import
`sitecustomize` if it exists on sys.path.

Because your project root is on PYTHONPATH (PYTHONPATH=.),
this file will be imported in:

- the batch driver (run_batch_tasks.py)
- every worker process (e.g., run_ollama_simple.py)
"""

import logging

logger = logging.getLogger("chatdev.instrumentation")

try:
    # This is your existing setup: exporters, ChatChain/ChatAgent wrappers, Ollama patch, etc.
    import chatdev_ext.instrumentation_bootstrap  # noqa: F401
    logger.info("[ALFIT] sitecustomize: instrumentation_bootstrap loaded in this process")
except Exception as exc:
    logger.exception("[ALFIT] sitecustomize: failed to load instrumentation_bootstrap: %s", exc)

try:
    # Extra layer: wrap the actual LLM boundary (openai.ChatCompletion.create)
    import chatdev_ext.llm_instrumentation  # noqa: F401
    logger.info("[ALFIT] sitecustomize: llm_instrumentation loaded in this process")
except Exception as exc:
    logger.exception("[ALFIT] sitecustomize: failed to load llm_instrumentation: %s", exc)

import os
os.environ.setdefault("CHATDEV_OTEL_AUTOSTART", "1")
os.environ.setdefault("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")

import chatdev_ext.instrumentation_bootstrap

import run_batch_tasks
if hasattr(run_batch_tasks, "main"):
    run_batch_tasks.main()

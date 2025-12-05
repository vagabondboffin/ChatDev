from __future__ import annotations
from typing import Any, Dict, Tuple

def identity_policy(message_obj: Any, ctx: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
    """
    Baseline policy: do nothing; return the message unchanged with metadata.
    Returning shape:
        (possibly_modified_message, {"fault_injected": bool, "fault_type": str})
    """
    return message_obj, {"fault_injected": False, "fault_type": "none"}

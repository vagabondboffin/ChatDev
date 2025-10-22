import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List
import os


class EnhancedLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Create timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"chatdev_trace_{self.timestamp}.jsonl")

        # Setup structured logging
        self.setup_logging()

    def setup_logging(self):
        """Configure logging format and handlers"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/execution_{self.timestamp}.log"),
                logging.StreamHandler()
            ]
        )

    def log_agent_interaction(self,
                              phase: str,
                              from_agent: str,
                              to_agent: str,
                              message: str,
                              message_type: str = "text",
                              tool_calls: List[Dict] = None,
                              timestamp: float = None):
        """Log detailed agent interaction"""

        if timestamp is None:
            timestamp = time.time()

        log_entry = {
            "timestamp": timestamp,
            "iso_timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "phase": phase,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message_type": message_type,
            "message_content": message,
            "tool_calls": tool_calls or [],
            "metadata": {
                "log_type": "agent_interaction"
            }
        }

        # Write to JSONL file
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Also log to console
        logging.info(f"AGENT_TRACE: {from_agent} -> {to_agent} | Phase: {phase} | Content: {message[:100]}...")

    def log_llm_call(self,
                     agent: str,
                     messages: List[Dict],
                     response: str,
                     model: str,
                     timestamp: float = None):
        """Log LLM API calls"""

        if timestamp is None:
            timestamp = time.time()

        log_entry = {
            "timestamp": timestamp,
            "iso_timestamp": datetime.fromtimestamp(timestamp).isoformat(),
            "agent": agent,
            "model": model,
            "request_messages": messages,
            "response": response,
            "metadata": {
                "log_type": "llm_call"
            }
        }

        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")


# Global logger instance
enhanced_logger = EnhancedLogger()
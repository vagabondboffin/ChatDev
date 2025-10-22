import requests
import json
import logging
from typing import Dict, Any, List


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2"):
        self.base_url = base_url
        self.model = model
        self.logger = logging.getLogger("OllamaClient")

    def format_messages_for_ollama(self, messages: List[Dict]) -> str:
        """Convert ChatML format to Ollama prompt format"""
        formatted_text = ""
        for message in messages:
            role = message["role"]
            content = message["content"]
            formatted_text += f"{role}: {content}\n"
        return formatted_text

    def chat_completion(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Create chat completion using Ollama API"""

        # Enhanced logging
        self.logger.info(f"Ollama Request - Model: {self.model}, Messages: {len(messages)}")
        for i, msg in enumerate(messages):
            self.logger.info(f"Message {i}: role={msg['role']}, content={msg.get('content', '')[:200]}...")

        # Prepare the request
        prompt = self.format_messages_for_ollama(messages)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9),
            }
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=120
            )
            response.raise_for_status()

            result = response.json()

            # Log the response
            self.logger.info(f"Ollama Response - Status: {response.status_code}")
            self.logger.info(f"Ollama Response - Content: {result.get('response', '')[:200]}...")

            # Format response to match OpenAI format
            return {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": result.get("response", ""),
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,  # Ollama doesn't provide token counts
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            raise


# Global instance
ollama_client = OllamaClient()
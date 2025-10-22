import os
import requests
import json
import logging
from typing import List, Dict, Any
import time
from datetime import datetime

from camel.typing import ModelType

# Import our enhanced logger
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from enhanced_logger import enhanced_logger


class OllamaModelBackend:
    def __init__(self, model_type):
        self.model_type = model_type
        self.model_name = os.environ.get("OLLAMA_MODEL", "llama2")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1").replace('/v1', '')
        self.logger = logging.getLogger("OllamaModelBackend")

    def run(self, messages: List[Dict], **kwargs) -> Dict[str, Any]:
        """Main method to get responses from Ollama using OpenAI-compatible API"""

        start_time = time.time()

        # Log the request
        enhanced_logger.log_llm_call(
            agent="System",
            messages=messages,
            response="",  # Will be filled after we get response
            model=self.model_name,
            timestamp=start_time
        )

        # Prepare the request for Ollama's OpenAI-compatible endpoint
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 4096),
            "stream": False
        }

        try:
            self.logger.info(f"Sending request to Ollama: model={self.model_name}, messages={len(messages)}")

            # Use the OpenAI-compatible endpoint
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()

            result = response.json()

            # Log the raw response for debugging
            self.logger.info(f"Raw Ollama response: {json.dumps(result)[:500]}...")

            # Extract the response text - handle different possible response structures
            response_text = ""
            if 'choices' in result and len(result['choices']) > 0:
                choice = result['choices'][0]
                if 'message' in choice and 'content' in choice['message']:
                    response_text = choice['message']['content']
                elif 'text' in choice:  # Some models might return 'text' instead
                    response_text = choice['text']
                else:
                    response_text = str(choice)  # Fallback
            elif 'response' in result:  # Direct Ollama API format
                response_text = result['response']
            else:
                response_text = "No response content found"

            # Log the response
            enhanced_logger.log_llm_call(
                agent="System",
                messages=messages,
                response=response_text,
                model=self.model_name,
                timestamp=start_time
            )

            # Return in the exact format that ChatDev expects
            # This matches the old OpenAI API response format
            return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_text,
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

        except Exception as e:
            self.logger.error(f"Ollama API error: {str(e)}")
            # Log the error
            enhanced_logger.log_llm_call(
                agent="System",
                messages=messages,
                response=f"ERROR: {str(e)}",
                model=self.model_name,
                timestamp=start_time
            )

            # Return a properly formatted fallback response
            return {
                "id": f"chatcmpl-error-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Error: {str(e)} - Please try again",
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }


class ModelBackend:
    def __new__(cls, model_type):
        # Always use Ollama backend regardless of model_type
        return OllamaModelBackend(model_type)


class ModelFactory:
    @staticmethod
    def create(model_type, model_config_dict=None):
        return ModelBackend(model_type)
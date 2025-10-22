#!/usr/bin/env python3
import requests
import json
import os


def test_ollama_connection():
    """Test if Ollama is responding to OpenAI-compatible API calls"""

    # Set up the same environment as our runner
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OLLAMA_MODEL"] = "llama2"

    model_name = os.environ.get("OLLAMA_MODEL", "llama2")
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1").replace('/v1', '')

    print(f"Testing Ollama connection...")
    print(f"Model: {model_name}")
    print(f"Base URL: {base_url}")

    # Test the OpenAI-compatible endpoint
    test_payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "Hello! Please respond with just 'OK' to confirm you're working."
            }
        ],
        "max_tokens": 10,
        "stream": False
    }

    try:
        response = requests.post(
            f"{base_url}/v1/chat/completions",
            json=test_payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                content = result['choices'][0]['message']['content']
                print(f"✅ Ollama is working! Response: {content}")
                return True
            else:
                print(f"❌ Unexpected response format: {result}")
                return False
        else:
            print(f"❌ Ollama returned status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except Exception as e:
        print(f"❌ Error connecting to Ollama: {e}")
        return False


if __name__ == "__main__":
    test_ollama_connection()
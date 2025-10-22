"""
Patched version of web_spider that works with Ollama and disables web functionality
"""
import os
import requests


def modal_trans(task_dsp):
    """
    Simplified version that uses Ollama instead of web spidering
    For research purposes, we can skip the Wikipedia/baike lookup
    """
    print("Web spider functionality simplified for Ollama integration")

    # If you want to use Ollama for some basic processing:
    try:
        ollama_model = os.environ.get("OLLAMA_MODEL", "llama2")

        # Simple prompt to extract keyword (mimicking original behavior but without web calls)
        task_in = f"Extract the most important single keyword from this software task: '{task_dsp}'. Return only one keyword without explanation."

        response = requests.post(
            "http://localhost:11434/v1/chat/completions",
            json={
                "model": ollama_model,
                "messages": [{"role": "user", "content": task_in}],
                "temperature": 0.2,
                "max_tokens": 50
            }
        )

        if response.status_code == 200:
            keyword = response.json()['choices'][0]['message']['content'].strip()
            print(f"Extracted keyword: {keyword}")

            # Return a simple summary based on the keyword (without web lookup)
            summary_prompt = f"Based on the keyword '{keyword}', provide a brief 1-2 sentence context for software development."
            summary_response = requests.post(
                "http://localhost:11434/v1/chat/completions",
                json={
                    "model": ollama_model,
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "temperature": 0.2,
                    "max_tokens": 100
                }
            )

            if summary_response.status_code == 200:
                result = summary_response.json()['choices'][0]['message']['content'].strip()
                print(f"Web spider content (via Ollama): {result}")
                return result
    except Exception as e:
        print(f"Web spider simplified processing failed: {e}")

    # Fallback: return empty string like original function
    return ''
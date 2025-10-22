#!/usr/bin/env python3
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import our patched backend to override the original
try:
    from camel.model_backend import OllamaModelBackend, ModelBackend, ModelFactory
    print("✅ Using patched Ollama model backend")
except ImportError as e:
    print(f"Warning: Could not import patched backend: {e}")

def setup_ollama_environment(model: str = "llama2"):
    """
    Set up environment variables to make ChatDev use Ollama instead of OpenAI
    This is the key insight - we're using Ollama's OpenAI compatibility
    """
    # These environment variables will be used by the OpenAI client in ChatDev
    os.environ["OPENAI_API_KEY"] = "ollama"  # Required but unused by Ollama
    os.environ["OPENAI_BASE_URL"] = "http://localhost:11434/v1"
    os.environ["OLLAMA_MODEL"] = model

    if 'BASE_URL' in os.environ:
        del os.environ['BASE_URL']

    print(f"✅ Environment configured for Ollama with model: {model}")
    print(f"   OPENAI_BASE_URL: {os.environ['OPENAI_BASE_URL']}")
    print(f"   OLLAMA_MODEL: {os.environ['OLLAMA_MODEL']}")


def main():
    parser = argparse.ArgumentParser(description="Run ChatDev with Ollama (OpenAI Compatibility)")
    parser.add_argument("--task", type=str, required=True, help="Task description")
    parser.add_argument("--name", type=str, required=True, help="Project name")
    parser.add_argument("--model", type=str, default="llama2", help="Ollama model to use")
    parser.add_argument("--config-path", type=str, default="CompanyConfig/Default/ChatChainConfig.json",
                        help="Path to ChatChainConfig.json")
    parser.add_argument("--config-phase-path", type=str, default="CompanyConfig/Default/PhaseConfig.json",
                        help="Path to PhaseConfig.json")
    parser.add_argument("--config-role-path", type=str, default="CompanyConfig/Default/RoleConfig.json",
                        help="Path to RoleConfig.json")

    args = parser.parse_args()

    # Setup Ollama environment
    setup_ollama_environment(args.model)

    # Verify config files exist
    config_files = {
        "ChatChainConfig": args.config_path,
        "PhaseConfig": args.config_phase_path,
        "RoleConfig": args.config_role_path
    }

    for config_name, config_path in config_files.items():
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            print(f"   Please make sure {config_name} exists at the specified path")
            return
        else:
            print(f"✅ Found config: {config_path}")

    # Import after setting environment variables
    from chatdev.chat_chain import ChatChain
    from camel.typing import ModelType

    # Initialize ChatChain with all required configs
    chat_chain = ChatChain(
        config_path=args.config_path,
        config_phase_path=args.config_phase_path,
        config_role_path=args.config_role_path,
        task_prompt=args.task,
        project_name=args.name,
        org_name="DefaultOrganization",
        model_type=ModelType.GPT_3_5_TURBO
    )

    # Execute the chat chain
    print(f"🚀 Starting ChatDev with Ollama (OpenAI Compatibility Mode)")
    print(f"   Task: {args.task}")
    print(f"   Project: {args.name}")
    print(f"   Model: {args.model}")

    try:
        chat_chain.pre_processing()
        chat_chain.make_recruitment()
        chat_chain.execute_chain()
        chat_chain.post_processing()
        print("✅ ChatDev completed successfully!")
    except Exception as e:
        print(f"❌ ChatDev failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
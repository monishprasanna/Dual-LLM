import time
import argparse
import sys
import requests
from openai import OpenAI

def get_available_models(base_url: str):
    """Fetches currently loaded or available models from LM Studio /v1/models."""
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/models", timeout=3)
        if resp.status_code == 200:
            data = resp.json()
            models = [m['id'] for m in data.get('data', [])]
            return models
    except Exception:
        pass
    return []

def run_auto_conversation(
    base_url: str = "http://localhost:1234/v1",
    model_a: str = "llama-3-8b-instruct",
    model_b: str = "mistral-7b-instruct",
    initial_prompt: str = "which came first chicken or egg?",
    max_turns: int = 5,
    system_a: str = "You are Model A. Respond directly to Model B. Keep answers clear and concise (1-3 sentences).",
    system_b: str = "You are Model B. Respond directly to Model A. Keep answers clear and concise (1-3 sentences).",
    max_history_turns: int = 10,
    temperature: float = 0.7
):
    """
    Orchestrates an automated back-and-forth chat between two models in LM Studio.
    Includes sliding-window context management to avoid context window overflow.
    """
    client = OpenAI(
        base_url=base_url,
        api_key="lm-studio"
    )

    # Check available models
    available_models = get_available_models(base_url)
    if available_models:
        print(f"📡 Found loaded models in LM Studio: {available_models}")
        # If user didn't specify or models aren't in available, pick top models if available
        if model_a not in available_models and len(available_models) > 0:
            print(f"⚠️ Model A '{model_a}' not explicitly listed, available: {available_models}. Using '{available_models[0]}'")
            model_a = available_models[0]
        if model_b not in available_models and len(available_models) > 0:
            alt_b = available_models[1] if len(available_models) > 1 else available_models[0]
            print(f"⚠️ Model B '{model_b}' not explicitly listed, available: {available_models}. Using '{alt_b}'")
            model_b = alt_b
    else:
        print(f"⚠️ Could not automatically verify models from {base_url}/models (LM Studio may still be starting up). Using defaults.")

    print(f"\n" + "=" * 60)
    print(f"🤖 Model A: {model_a}")
    print(f"🧠 Model B: {model_b}")
    print(f"🚀 INITIAL PROMPT: {initial_prompt}")
    print("=" * 60 + "\n")

    # Set up message histories
    history_a = [
        {"role": "system", "content": system_a},
        {"role": "user", "content": initial_prompt}
    ]
    history_b = [
        {"role": "system", "content": system_b}
    ]

    for turn in range(1, max_turns + 1):
        print(f"------------------- Turn {turn} / {max_turns} -------------------")

        # --- MODEL A GENERATES RESPONSE ---
        # Context sliding window for Model A (keep system prompt + last N messages)
        trimmed_history_a = [history_a[0]] + history_a[-max_history_turns:] if len(history_a) > max_history_turns + 1 else history_a

        try:
            res_a = client.chat.completions.create(
                model=model_a,
                messages=trimmed_history_a,
                temperature=temperature
            )
            msg_a = res_a.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error getting response from Model A ({model_a}): {e}")
            break

        print(f"\n🤖 [Model A - {model_a}]:\n{msg_a}\n")

        # Update history for Model A and feed output to Model B
        history_a.append({"role": "assistant", "content": msg_a})
        history_b.append({"role": "user", "content": msg_a})

        # --- MODEL B GENERATES RESPONSE ---
        # Context sliding window for Model B
        trimmed_history_b = [history_b[0]] + history_b[-max_history_turns:] if len(history_b) > max_history_turns + 1 else history_b

        try:
            res_b = client.chat.completions.create(
                model=model_b,
                messages=trimmed_history_b,
                temperature=temperature
            )
            msg_b = res_b.choices[0].message.content.strip()
        except Exception as e:
            print(f"❌ Error getting response from Model B ({model_b}): {e}")
            break

        print(f"🧠 [Model B - {model_b}]:\n{msg_b}\n")

        # Update history for Model B and feed output to Model A
        history_b.append({"role": "assistant", "content": msg_b})
        history_a.append({"role": "user", "content": msg_b})

        time.sleep(1) # Small pause between turns

    print("\n✅ Auto-conversation finished!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated dual-model dialogue using LM Studio local API.")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LM Studio server URL")
    parser.add_argument("--model-a", default="llama-3-8b-instruct", help="Model A ID")
    parser.add_argument("--model-b", default="mistral-7b-instruct", help="Model B ID")
    parser.add_argument("--turns", type=int, default=5, help="Total back-and-forth turns")
    parser.add_argument("--prompt", default="which came first chicken or egg?", help="Initial starting prompt")

    args = parser.parse_args()

    run_auto_conversation(
        base_url=args.base_url,
        model_a=args.model_a,
        model_b=args.model_b,
        initial_prompt=args.prompt,
        max_turns=args.turns
    )

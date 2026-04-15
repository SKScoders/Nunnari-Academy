

import time
import ollama

DEFAULT_ROLES: dict[str, str] = {
    "Python Tutor": (
        "You are an expert Python tutor. Explain concepts clearly with "
        "short code examples. Encourage the learner and correct mistakes gently."
    ),
    "Fitness Coach": (
        "You are an enthusiastic personal fitness coach. Give safe, practical "
        "workout and nutrition advice. Always remind users to consult a doctor "
        "before starting a new programme."
    ),
    "Travel Guide": (
        "You are a knowledgeable travel guide. Suggest destinations, share "
        "cultural tips, visa info, packing advice, and local food recommendations."
    ),
    "Interview Coach": (
        "You are a senior software-engineering interview coach. Help the user "
        "practise coding problems, system design, and behavioural questions. "
        "Give detailed feedback on every answer."
    ),
    "Storyteller": (
        "You are a creative storyteller. Craft imaginative, engaging stories "
        "based on the user's prompts. Use vivid descriptions and interesting "
        "characters. Ask follow-up questions to shape the story together."
    ),
}

MODEL = "llama3.2"   


def clear_line() -> None:
    print()


def print_banner() -> None:
    print("\n" + "═" * 54)
    print("     Role-Based Chat with Ollama")
    print("   Nunnari Academy — LLM Capstone Task")
    print("═" * 54)


def count_tokens(text: str) -> int:
    """Rough token estimate: ~4 characters per token (good enough for display)."""
    return max(1, len(text) // 4)



def show_role_menu(roles: dict[str, str]) -> None:
    print("\n─────────────────────────────────────────")
    print("           Choose Your Role                ")
    print("─────────────────────────────────────────")
    for idx, name in enumerate(roles, start=1):
        print(f"│  {idx}. {name:<38}│")
    print("─────────────────────────────────────────")


def pick_role(roles: dict[str, str]) -> tuple[str, str]:
    """Return (role_name, system_prompt) chosen by the user."""
    role_names = list(roles.keys())
    show_role_menu(roles)

    while True:
        choice = input("\nEnter role number: ").strip()
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(role_names):
                name = role_names[idx]
                print(f"\n  Role selected: {name}")
                return name, roles[name]
        print("  Invalid choice. Please try again.")



def add_custom_role(roles: dict[str, str]) -> None:
    print("\n─── Add a Custom Role ───")
    name = input("Role name   : ").strip()
    if not name:
        print("  Role name cannot be empty.")
        return
    prompt = input("System prompt: ").strip()
    if not prompt:
        print("  System prompt cannot be empty.")
        return
    roles[name] = prompt
    print(f"  Custom role '{name}' added successfully!")



def chat_loop(roles: dict[str, str]) -> None:
    """Main conversation loop."""

    role_name, system_prompt = pick_role(roles)

    # Chat history — always starts fresh when a role is (re)selected
    history: list[dict] = []

    print(f"\n  Chatting as '{role_name}'")
    print("    Commands: 'switch' | 'roles' | 'quit'\n")
    print("─" * 54)

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() == "quit":
            print("\n  Thanks for chatting! Goodbye.")
            break

        if user_input.lower() == "switch":
            print("\n  Switching role…")
            role_name, system_prompt = pick_role(roles)
            history = []          # reset history for the new role
            print(f"\n  Now chatting as '{role_name}'\n" + "─" * 54)
            continue

        if user_input.lower() == "roles":
            add_custom_role(roles)
            continue

        history.append({"role": "user", "content": user_input})

        messages = [{"role": "system", "content": system_prompt}] + history

        print(f"\n{role_name}: ", end="", flush=True)
        start_time = time.time()

        try:
            response = ollama.chat(model=MODEL, messages=messages)
        except Exception as e:
            print(f"\n  Error talking to Ollama: {e}")
            print(f"    Make sure Ollama is running and model '{MODEL}' is pulled.")
            print(f"    Run:  ollama pull {MODEL}\n")
            history.pop()          # remove the message we couldn't send
            continue

        elapsed = time.time() - start_time
        assistant_text: str = response["message"]["content"]

        print(assistant_text)

        in_tokens  = sum(count_tokens(m["content"]) for m in history)
        out_tokens = count_tokens(assistant_text)
        print(f"\n    {elapsed:.2f}s  |  "
              f"~{in_tokens} in-tokens  |  ~{out_tokens} out-tokens")
        print("─" * 54)

        history.append({"role": "assistant", "content": assistant_text})



def main() -> None:
    print_banner()

    roles = dict(DEFAULT_ROLES)

    chat_loop(roles)


if __name__ == "__main__":
    main()

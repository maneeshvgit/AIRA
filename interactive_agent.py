# interactive_agent.py

from agent import ask_agent  # Import the helper function from your agent.py

def main():
    print("AI Teacher Agent Interactive Terminal")
    print("Type 'exit' or 'quit' to stop.")
    print("Start by asking the AI teacher to explain a topic or ask a question.\n")

    while True:
        user_input = input("You (student): ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        # Send user input to AI teacher agent and get response
        response = ask_agent(user_input)

        print("\nAI Teacher:")
        print(response)
        print("\n---")

if __name__ == "__main__":
    main()

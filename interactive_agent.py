from agent import ask_agent
import uuid

def main():
    print("AI Teacher Agent Interactive Terminal")
    print("Type 'exit' or 'quit' to stop.")
    print("Press Enter to let the teacher continue, or type a question to interrupt.\n")

    thread_id = f"classroom_session_{uuid.uuid4().hex[:8]}"  # Persistent session ID
    lesson_buffer = []               # Current lesson lines
    current_line_idx = 0

    while True:
        # If no ongoing buffered lesson, get fresh query from student
        if not lesson_buffer or current_line_idx >= len(lesson_buffer):
            user_input = input("You (student): ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break

            # Get AI's response and split into sentences
            ai_response = ask_agent(user_input, thread_id=thread_id)
            lesson_buffer = [line.strip() for line in ai_response.split(". ") if line.strip()]
            current_line_idx = 0
            print("\nAI Teacher:\n")

        # Display one chunk/sentence at a time
        while current_line_idx < len(lesson_buffer):
            print(lesson_buffer[current_line_idx])
            current_line_idx += 1

            # Ask student if they want to continue or interrupt
            student_action = input("(Enter = continue, or type question): ").strip()

            if student_action.lower() in ("exit", "quit"):
                print("Goodbye!")
                return

            if student_action:
                # Interruption detected → Save recent context
                start_idx = max(0, current_line_idx - 3)
                recent_lines = lesson_buffer[start_idx:current_line_idx]
                recap_context = " ".join(recent_lines)

                # Ask the teacher to resume naturally after answering
                followup_prompt = (
                    f"Before we paused, we were discussing: '{recap_context}'. "
                    f"A student asked: '{student_action}'. "
                    f"Please answer the question clearly and then smoothly continue the lesson from there."
                )

                ai_followup = ask_agent(followup_prompt, thread_id=thread_id)
                # Replace buffer with new continuation
                lesson_buffer = [line.strip() for line in ai_followup.split(". ") if line.strip()]
                current_line_idx = 0
                print("\nAI Teacher:\n")
                break  # break to outer while loop to print the new flow


if __name__ == "__main__":
    main()

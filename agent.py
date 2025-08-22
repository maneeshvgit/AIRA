from langchain_openai import ChatOpenAI   # OpenAI-compatible wrapper for Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from agent_tools import knowledgebase_tool, image_tool, video_tool,lesson_builder
import traceback

# =======================
# Your Groq API Key
# =======================
GROQ_API_KEY = "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"

# =======================
# Initialize Groq LLM
# =======================
llm = ChatOpenAI(
    model="llama3-70b-8192",   # choices: "llama3-70b-8192", "mixtral-8x7b-32768", "llama3-8b-8192"
    temperature=0.7,
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)
print(f"[DEBUG] Initialized Groq LLM: model={llm.model_name}, temperature={llm.temperature}")

# =======================
# System prompt
# =======================
agent_system_prompt = """
You are an engaging, empathetic, and knowledgeable AI science teacher for middle-school students.

INSTRUCTIONS:

* Always teach in student-friendly, enthusiastic language, using analogies and stories.
* Clearly distinguish between three types of student input:
    * New topic → Call LessonBuilder once. LessonBuilder will automatically gather the textbook explanation, an image, and a short video (if available). If any resource is missing, ignore it silently. Introduce the results naturally in your teaching (e.g., “Here’s a diagram to help you picture this” or “Let’s watch a short video to see this in action”).
    * Doubts or follow-up questions → Call KnowledgeBaseSearch only once. If no result is found, say: “That’s not exactly in your syllabus, but I can still explain it in a simple way,” and then give a simplified answer.
    * Casual chit-chat → Do not call any tools. Just reply warmly, like a friendly teacher.
* After any tool returns, continue your explanation smoothly, weaving the result into your teaching.
* When a student interrupts with a question, pause, answer it, and then resume the lesson from where you left off.
* If you don’t know something, admit it warmly and encourage curiosity.
* Never repeat the same content unnecessarily — adapt to the student’s feedback and questions.
* Maintain the flow of a real classroom: be warm, engaging, and adaptive.

TOOLS AVAILABLE:

- LessonBuilder (for structured lessons with explanation, image, and video)
- KnowledgeBaseSearch (for textbook-style explanations when answering doubts)
- Behave like a passionate human teacher, not a robot.
"""

# =======================
# Conversation memory
# =======================
memory_saver = MemorySaver()

# =======================
# Create ReAct agent
# =======================
agent = create_react_agent(
    llm,
    tools=[lesson_builder, knowledgebase_tool],
    prompt=agent_system_prompt,
    checkpointer=memory_saver
)
# =======================
# Query function
# =======================
def ask_agent(question: str, thread_id="main") -> str:
    """
    Send a query to the dynamic AI teacher agent and get a response.
    Maintains memory context across the session using thread_id.
    Includes debug statements for all steps.
    """
    print(f"[DEBUG] Calling agent.invoke with question: {question} | thread_id: {thread_id}")
    try:
        response = agent.invoke(
            {"messages": [("human", question)]},
            config={"configurable": {"thread_id": thread_id}}
        )
        if response and response.get("messages"):
            output = response["messages"][-1].content
            # print(f"[DEBUG] AI output (final message): {output}")
            return output
        else:
            print("[DEBUG] agent.invoke: No messages in response, returning empty string.")
            return ""
    except Exception as e:
        print(f"[ERROR] agent.invoke raised an exception: {e}")
        traceback.print_exc()
        return f"Sorry, an error occurred while processing your request: {e}"

if __name__ == "__main__":
    print("Testing dynamic AI Teacher Agent with LangGraph (Groq LLaMA)...")
    test_query = "Explain photosynthesis."
    print(f"Question: {test_query}")
    print("Answer:")
    print(ask_agent(test_query))


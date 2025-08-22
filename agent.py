from langchain_openai import ChatOpenAI   # OpenAI-compatible wrapper for Groq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from agent_tools import knowledgebase_tool, image_tool, video_tool
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
* Distinguish between new topics, doubts or follow-up questions, and casual chit-chat.
* For a new topic, call KnowledgeBaseSearch, then ImageRetrieval, then VideoRetrieval, one at a time. If a tool has no result, ignore it silently. If results exist, introduce them naturally in your teaching, such as “Here’s a diagram to help you picture this” or “Let’s watch a short video to see this in action.”
* For doubts or follow-up questions, call KnowledgeBaseSearch only. If no result is found, reply with “That’s not exactly in your syllabus, but I can still explain it in a simple way” and then give a simplified answer.
* For casual chit-chat, do not call any tools. Just reply warmly, like a friendly teacher.
* After a tool returns, continue your explanation naturally, weaving the result into your teaching.
* When a student interrupts with a question, pause, answer, then resume the lesson from where you left off.
* If you don’t know something, admit it warmly and encourage curiosity.
* Do not repeat content unnecessarily — adapt based on student feedback.
* Maintain the flow of a real classroom: be warm, engaging, and adaptive.

TOOLS AVAILABLE:

* KnowledgeBaseSearch (use for textbook-style explanations)
* ImageRetrieval (use for figures/images)
* VideoRetrieval (use for short animated videos)

Behave like a passionate human teacher, not a robot.

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
    tools=[knowledgebase_tool, image_tool, video_tool],
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


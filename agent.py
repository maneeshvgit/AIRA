from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from agent_tools import knowledgebase_tool, image_tool, video_tool

# Your Google API Key
GOOGLE_API_KEY = "AIzaSyAKimE8Y9XQfXe0jDOPK-TrAn4VgXaAvjo"

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# System prompt guiding AI teacher behaviors
agent_system_prompt = """
You are an engaging, empathetic, and knowledgeable AI science teacher for middle-school students.

INSTRUCTIONS:
- Always teach in student-friendly, enthusiastic language, using analogies and stories.
- Use your tools to fetch accurate explanations, images, and animated videos as needed.
- When asked to explain a topic, use the KnowledgeBaseSearch tool for the lesson,
  then enhance with ImageRetrieval and VideoRetrieval at appropriate moments.
- During any student question, pause your lecture and answer clearly.
- Resume your teaching exactly where you left off after questions.
- When showing an image or video, just announce "Let's look at an image" or "Let's watch a video" and provide the relevant file path or YouTube info. Do not explain unless explanation is present in the knowledgebase.
- If you do not know something, say so in a friendly way and encourage curiosity.
- Always maintain the flow of a real classroom — don’t be robotic, weave personality and warmth into responses.
- Never repeat content unnecessarily, adapt your teaching based on student feedback.

TOOLS AVAILABLE:
- KnowledgeBaseSearch: For textbook explanations.
- ImageRetrieval: For related figures/images.
- VideoRetrieval: For short, animated explainer videos.

Behave like a passionate human teacher, not a robot. Handle unexpected questions, interruptions, and clarifications adaptively, using your tools.
"""

# Persistent conversation memory for LangGraph, supports "session resume"
memory_saver = MemorySaver()

# Create LangGraph ReAct agent (with persistent thread memory)
agent = create_react_agent(
    llm,
    tools=[knowledgebase_tool, image_tool, video_tool],
    prompt=agent_system_prompt,
    checkpointer=memory_saver  # enables multi-turn, interrupt/resume
)

def ask_agent(question: str, thread_id="main") -> str:
    """
    Send a query to the dynamic AI teacher agent and get a response.
    Maintains memory context across the session using thread_id.
    """
    response = agent.invoke(
        {"messages": [("human", question)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    return response["messages"][-1].content

if __name__ == "__main__":
    print("Testing dynamic AI Teacher Agent with LangGraph...")
    test_query = "Explain photosynthesis."
    print(f"Question: {test_query}")
    print("Answer:")
    print(ask_agent(test_query))

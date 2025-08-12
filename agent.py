from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langgraph.prebuilt import create_react_agent
from agent_tools import knowledgebase_tool, image_tool, video_tool

GOOGLE_API_KEY = "AIzaSyAKimE8Y9XQfXe0jDOPK-TrAn4VgXaAvjo"  # Use your key

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

agent_system_prompt = """
You are an engaging, empathetic, and knowledgeable AI science teacher for middle-school students.

INSTRUCTIONS:
- Always teach in student-friendly, enthusiastic language, using analogies and stories.
- Use your tools to fetch accurate explanations, images, and animated videos as needed.
- When asked to explain a topic, use the KnowledgeBaseSearch tool for the lesson,
  then enhance with ImageRetrieval and VideoRetrieval at appropriate moments.
- During any student question, pause your lecture and answer clearly.
- Resume your teaching exactly where you left off after questions.
- When showing an image, describe it vividly; when it’s time for a video, announce the break, pause narration,
  play video, then continue teaching after.
- If you do not know something, say so in a friendly way and encourage curiosity.
- Always maintain the flow of a real classroom — don’t be robotic, weave personality and warmth into responses.
- Never repeat content unnecessarily, adapt your teaching based on student feedback.

TOOLS AVAILABLE:
- KnowledgeBaseSearch: For textbook explanations.
- ImageRetrieval: For related figures/images.
- VideoRetrieval: For short, animated explainer videos.

Behave like a passionate human teacher, not a robot. Handle unexpected questions, interruptions, and clarifications adaptively, using your tools.
"""

memory = ConversationBufferMemory(memory_key="chat_history")

agent = create_react_agent(
    llm,
    tools=[knowledgebase_tool, image_tool, video_tool],
    prompt=agent_system_prompt
)


def ask_agent(question: str) -> str:
    response = agent.invoke({"messages": [("human", question)]})
    return response["messages"][-1].content

if __name__ == "__main__":
    print("Testing dynamic AI Teacher Agent with LangGraph...")
    test_query = "Explain photosynthesis."
    print(f"Question: {test_query}")
    print("Answer:", ask_agent(test_query))

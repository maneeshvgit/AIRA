# agent.py

from langchain_community.chat_models import ChatPerplexity
from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from agent_tools import knowledgebase_tool, image_tool, video_tool

# --------------------------------------------
# 1. Set your Perplexity API key here (replace with your actual key)
PERPLEXITY_API_KEY = "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"

# --------------------------------------------
# 2. Initialize the Perplexity LLM
llm = ChatPerplexity(
    model="llama-3.1-sonar-small-128k-online",  # Example model name, update if needed
    temperature=0.7,
    pplx_api_key=PERPLEXITY_API_KEY
)

# --------------------------------------------
# 3. Full detailed system prompt guiding AI teacher behavior
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

# --------------------------------------------
# 4. Initialize conversation memory to keep context during interaction
memory = ConversationBufferMemory(memory_key="chat_history")

# --------------------------------------------
# 5. Initialize the LangChain conversational agent with tools, prompt, memory, and LLM
agent = initialize_agent(
    tools=[knowledgebase_tool, image_tool, video_tool],
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    system_prompt=agent_system_prompt
)

# --------------------------------------------
# 6. Helper function for easier agent interaction
def ask_agent(question: str) -> str:
    """
    Send a query to the AI teacher agent and get a response.
    Maintains conversation state with memory.
    """
    response = agent.run(question)
    return response


# Optional: For quick test when running this file directly
if __name__ == "__main__":
    print("Testing AI Teacher Agent...")
    test_query = "Explain photosynthesis."
    print(f"Question: {test_query}")
    print("Answer:", ask_agent(test_query))

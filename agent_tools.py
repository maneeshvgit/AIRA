# agent_tools.py

from langchain.tools import Tool
from utils import search, fetch_figures_only, fetch_animated_videos

# Tool 1: Knowledge Base (Textbook) Search Tool
def knowledgebase_tool_func(query: str) -> str:
    results = search(query, mode="hybrid", top_k=1)
    if results:
        return results[0]['content']
    else:
        return "Sorry, I couldn't find information for that topic."

knowledgebase_tool = Tool(
    name="KnowledgeBaseSearch",
    func=knowledgebase_tool_func,
    description="Retrieves explanations from the science textbook knowledge base."
)

# Tool 2: Image/Figure Retrieval Tool
def image_tool_func(topic: str) -> str:
    results = fetch_figures_only(topic)
    if isinstance(results, str):  # error or no figures found
        return results
    elif results:
        # Format image info textually; actual display handled later by front end
        imgs = [f"{img['name']}: {img['desc']} (path: {img['path']})" for img in results]
        return "\n".join(imgs)
    else:
        return "No relevant images found."

image_tool = Tool(
    name="ImageRetrieval",
    func=image_tool_func,
    description="Fetches relevant figures and descriptive details for a science topic."
)

# Tool 3: Animated Video Retrieval Tool
def video_tool_func(topic: str) -> str:
    result = fetch_animated_videos(topic)
    if result:
        # Provide video title and Youtube ID so front end can embed if needed
        return f"Video: {result['title']} (YouTube ID: {result['id']})"
    else:
        return "No animation video found for this topic."

video_tool = Tool(
    name="VideoRetrieval",
    func=video_tool_func,
    description="Finds short animated explainer videos for science concepts."
)

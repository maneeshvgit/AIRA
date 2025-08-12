# agent_tools.py

from langchain.tools import Tool
from utils import search, fetch_figures_only, fetch_animated_videos

# Tool 1: Knowledge Base (Textbook) Search Tool
def knowledgebase_tool_func(query: str) -> str:
    print(f"[DEBUG] knowledgebase_tool_func called with query: {query}")
    results = search(query, mode="hybrid", top_k=1)
    if results:
        output = results[0]['content']
    else:
        output = "Sorry, I couldn't find information for that topic."
    print(f"[DEBUG] knowledgebase_tool_func output:\n{output}\n")
    return output

knowledgebase_tool = Tool(
    name="KnowledgeBaseSearch",
    func=knowledgebase_tool_func,
    description="Retrieves explanations from the science textbook knowledge base."
)

# Tool 2: Image/Figure Retrieval Tool
def image_tool_func(topic: str) -> str:
    print(f"[DEBUG] image_tool_func called with topic: {topic}")
    results = fetch_figures_only(topic)
    if isinstance(results, str):  # error or no figures found
        output = results
    elif results:
        imgs = [f"{img['name']}: {img['desc']} (path: {img['path']})" for img in results]
        output = "\n".join(imgs)
    else:
        output = "No relevant images found."
    print(f"[DEBUG] image_tool_func output:\n{output}\n")
    return output

image_tool = Tool(
    name="ImageRetrieval",
    func=image_tool_func,
    description="Fetches relevant figures and descriptive details for a science topic."
)

# Tool 3: Animated Video Retrieval Tool
def video_tool_func(topic: str) -> str:
    print(f"[DEBUG] video_tool_func called with topic: {topic}")
    # Avoid calling video search with full video titles or if topic contains YouTube ID
    if "YouTube ID:" in topic or "youtube.com" in topic.lower() or "http" in topic.lower():
        output = "Skipping video search on likely video title or URL."
        print(f"[DEBUG] video_tool_func output:\n{output}\n")
        return output
    result = fetch_animated_videos(topic)
    if result:
        output = f"Video: {result['title']} (YouTube ID: {result['id']})"
    else:
        output = "No animation video found for this topic."
    print(f"[DEBUG] video_tool_func output:\n{output}\n")
    return output


video_tool = Tool(
    name="VideoRetrieval",
    func=video_tool_func,
    description="Finds short animated explainer videos for science concepts."
)

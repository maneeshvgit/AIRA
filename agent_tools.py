from langchain.tools import Tool
from utils import search, fetch_animated_videos
import json
import os
import faiss
import torch
from sentence_transformers import SentenceTransformer

# === New Image Retrieval Logic ===
FIGURE_JSON = r"/home/ailab/Documents/working/AIRA/output.json"
IMAGE_DIR = r"/home/ailab/Documents/working/AIRA/images"
FAISS_INDEX_FILE = r"/home/ailab/Documents/working/AIRA/subchapter_faiss.index"
METADATA_FILE = r"/home/ailab/Documents/working/AIRA/subchapter_metadata.json"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").to(device)

with open(FIGURE_JSON, "r", encoding="utf-8") as f:
    figures_data = json.load(f)

with open(METADATA_FILE, "r", encoding="utf-8") as f:
    metadata_figures = json.load(f)

index_figures = faiss.read_index(FAISS_INDEX_FILE)

def get_image_path(figure_ref, image_dir=IMAGE_DIR):
    base_name = figure_ref.replace(" ", "_")
    attempts = [
        f"{base_name}.png",
        f"{base_name}.jpg",
        f"figure_{base_name}.png"
    ]
    for attempt in attempts:
        test_path = os.path.join(image_dir, attempt)
        if os.path.exists(test_path):
            return test_path
    return None

def fetch_figures_only(subchapter_name):
    figures = [fig for fig in figures_data if fig["subchapter"] == subchapter_name]
    figure_blocks = []
    for fig in figures:
        fig_path = get_image_path(fig['figure'])
        if fig_path:
            figure_blocks.append({
                "name": fig['figure'],
                "path": fig_path,
                "desc": fig['description']
            })
    return figure_blocks

def search_subchapter_by_query(query, top_k=1):
    query_embedding = image_model.encode([query], convert_to_numpy=True).astype('float32')
    _, indices = index_figures.search(query_embedding.reshape(1, -1), top_k)
    best_match_index = str(indices[0][0])
    return metadata_figures.get(best_match_index, None)

def fetch_images_for_topic(query):
    subchapter = search_subchapter_by_query(query)
    if not subchapter:
        return []
    return fetch_figures_only(subchapter)

# === Tools ===

# Tool 1: Knowledge Base Search Tool
def knowledgebase_tool_func(query: str) -> str:
    print(f"[DEBUG] knowledgebase_tool_func called with query: {query}")
    results = search(query, mode="hybrid", top_k=1)
    if results:
        output = results[0]['content']
    else:
        output = "Sorry, I couldn't find information for that topic."
    # print(f"[DEBUG] knowledgebase_tool_func output:\n{output}\n")
    return output

knowledgebase_tool = Tool(
    name="KnowledgeBaseSearch",
    func=knowledgebase_tool_func,
    description="Retrieves explanations from the science textbook knowledge base."
)

# Tool 2: Image/Figure Retrieval Tool (Updated)
def image_tool_func(topic: str) -> str:
    print(f"[DEBUG] image_tool_func called with topic: {topic}")
    results = fetch_images_for_topic(topic)
    if isinstance(results, str):  # error or no figures found
        output = results
    elif results:
        imgs = [f"Let's look at an image: {img['name']} (Description: {img['desc']}) - Path: {img['path']}" for img in results]
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
    if "YouTube ID:" in topic or "youtube.com" in topic.lower() or "http" in topic.lower():
        output = "Skipping video search on likely video title or URL."
        print(f"[DEBUG] video_tool_func output:\n{output}\n")
        return output
    result = fetch_animated_videos(topic)
    if result:
        output = f"Let's watch a video: {result['title']} (YouTube ID: {result['id']})"
    else:
        output = "No animation video found for this topic."
    print(f"[DEBUG] video_tool_func output:\n{output}\n")
    return output

video_tool = Tool(
    name="VideoRetrieval",
    func=video_tool_func,
    description="Finds short animated explainer videos for science concepts."
)

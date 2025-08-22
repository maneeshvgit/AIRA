from langchain.tools import tool
from utils import search
import json
import os
from textwrap import dedent
import yt_dlp
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

@tool
def knowledgebase_tool(query: str) -> str:
    """Retrieves explanations from the science textbook knowledge base."""
    print(f"[DEBUG] knowledgebase_tool called with query: {query}")
    results = search(query, mode="hybrid", top_k=1)
    if results:
        output = results[0]['content']
    else:
        output = "Sorry, I couldn't find information for that topic."
    return output


@tool
def image_tool(topic: str) -> str:
    """Fetches relevant figures and descriptive details for a science topic."""
    print(f"[DEBUG] image_tool called with topic: {topic}")
    results = fetch_images_for_topic(topic)
    if isinstance(results, str):
        output = results
    elif results:
        imgs = [f"{img['name']} — {img['desc']} (see: {img['path']})" for img in results]
        output = "\n".join(imgs)
    else:
        output = "No relevant images found."
    print(f"[DEBUG] image_tool output:\n{output}\n")
    return output


def fetch_animated_videos(topic, num_videos=1):
    search_query = f"ytsearch{num_videos}:{topic} animation explained in english"
    ydl_opts = {
        "quiet": True,
        "extract_flat": True,
        "force_generic_extractor": True
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(search_query, download=False)
        if "entries" in info and len(info["entries"]) > 0:
            video = info["entries"][0]
            if video.get("duration", 301) <= 300:  # only pick videos ≤ 5 min
                url = f"https://www.youtube.com/watch?v={video['id']}"
                return {
                    "title": video["title"],
                    "url": url,
                    "id": video["id"]
                }
    return None


@tool
def video_tool(topic: str) -> str:
    """Finds short animated explainer videos for science concepts."""
    print(f"[DEBUG] video_tool called with topic: {topic}")
    
    if "youtube.com" in topic.lower() or "http" in topic.lower():
        output = "Skipping video search on likely video URL."
        print(f"[DEBUG] video_tool output:\n{output}\n")
        return output
    
    result = fetch_animated_videos(topic)
    if result:
        output = f"{result['title']} (YouTube: {result['url']})"
    else:
        output = "No animation video found for this topic."
    
    print(f"[DEBUG] video_tool output:\n{output}\n")
    return output


@tool
def lesson_builder(topic: str) -> str:
    """Builds a structured lesson by combining text, images, and video (returns formatted content)."""
    print(f"[DEBUG] lesson_builder called with topic: {topic}")

    kb_text = knowledgebase_tool.invoke(topic)
    image_info = image_tool.invoke(topic)
    video_info = video_tool.invoke(topic)

    image_ref = image_info if image_info and "see:" in image_info else None
    video_ref = video_info if video_info and "http" in video_info else None

    # Structured lesson content (no LLM here!)
    lesson = dedent(f"""
    Topic: {topic}

    Main Explanation:
    {kb_text if kb_text else "No detailed explanation available."}

    {"Here’s a diagram to help you picture this: " + image_ref if image_ref else ""}

    {"Let’s watch a short video to see this in action: " + video_ref if video_ref else ""}
    """).strip()

    print(f"[DEBUG] lesson_builder output:\n{lesson}\n")
    return lesson

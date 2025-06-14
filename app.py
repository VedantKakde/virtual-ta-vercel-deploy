import os
import json
import sqlite3
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import aiohttp
import asyncio
import logging
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "knowledge_base.db"
SIMILARITY_THRESHOLD = 0.68
MAX_RESULTS = 10
MAX_CONTEXT_CHUNKS = 4

load_dotenv()
API_KEY = os.getenv("API_KEY")

# --- Models ---
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None  # Base64 encoded image

class LinkInfo(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[LinkInfo]

# --- FastAPI App Initialization ---
app = FastAPI(title="RAG Query API", description="API for querying the RAG knowledge base")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Database Utilities ---
def get_db_connection():
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        logger.error(f"Database connection error: {e}")
        raise HTTPException(status_code=500, detail="Database connection error")

def initialize_database():
    if not os.path.exists(DB_PATH):
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS discourse_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER, topic_id INTEGER, topic_title TEXT, post_number INTEGER,
                author TEXT, created_at TEXT, likes INTEGER, chunk_index INTEGER,
                content TEXT, url TEXT, embedding BLOB
            )
        ''')
        c.execute('''
            CREATE TABLE IF NOT EXISTS markdown_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_title TEXT, original_url TEXT, downloaded_at TEXT, chunk_index INTEGER,
                content TEXT, embedding BLOB
            )
        ''')
        conn.commit()
        conn.close()
initialize_database()

# --- Embedding and Similarity ---
def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

async def get_embedding(text, max_retries=3):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["data"][0]["embedding"]
                    elif response.status == 429:
                        await asyncio.sleep(5 * (attempt + 1))
                    else:
                        raise HTTPException(status_code=response.status, detail=await response.text())
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(3 * (attempt + 1))

# --- Content Retrieval ---
async def find_similar_content(query_embedding, conn):
    cursor = conn.cursor()
    results = []
    # Discourse
    cursor.execute("SELECT * FROM discourse_chunks WHERE embedding IS NOT NULL")
    for row in cursor.fetchall():
        emb = json.loads(row["embedding"])
        sim = cosine_similarity(query_embedding, emb)
        if sim >= SIMILARITY_THRESHOLD:
            url = row["url"]
            if not url.startswith("http"):
                url = f"https://discourse.onlinedegree.iitm.ac.in/t/{url}"
            results.append({
                "source": "discourse", "id": row["id"], "post_id": row["post_id"],
                "title": row["topic_title"], "url": url, "content": row["content"],
                "author": row["author"], "created_at": row["created_at"],
                "chunk_index": row["chunk_index"], "similarity": sim
            })
    # Markdown
    cursor.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
    for row in cursor.fetchall():
        emb = json.loads(row["embedding"])
        sim = cosine_similarity(query_embedding, emb)
        if sim >= SIMILARITY_THRESHOLD:
            url = row["original_url"] or f"https://docs.onlinedegree.iitm.ac.in/{row['doc_title']}"
            results.append({
                "source": "markdown", "id": row["id"], "title": row["doc_title"],
                "url": url, "content": row["content"], "chunk_index": row["chunk_index"],
                "similarity": sim
            })
    # Group and select top results
    results.sort(key=lambda x: x["similarity"], reverse=True)
    grouped = {}
    for r in results:
        key = f"{r['source']}_{r.get('post_id', r['title'])}"
        grouped.setdefault(key, []).append(r)
    final_results = []
    for chunks in grouped.values():
        final_results.extend(sorted(chunks, key=lambda x: x["similarity"], reverse=True)[:MAX_CONTEXT_CHUNKS])
    return sorted(final_results, key=lambda x: x["similarity"], reverse=True)[:MAX_RESULTS]

async def enrich_with_adjacent_chunks(conn, results):
    cursor = conn.cursor()
    enriched = []
    for r in results:
        content = r["content"]
        if r["source"] == "discourse":
            post_id, idx = r["post_id"], r["chunk_index"]
            for adj in [idx - 1, idx + 1]:
                if adj >= 0:
                    cursor.execute(
                        "SELECT content FROM discourse_chunks WHERE post_id=? AND chunk_index=?",
                        (post_id, adj)
                    )
                    row = cursor.fetchone()
                    if row:
                        content += " " + row["content"]
        else:
            title, idx = r["title"], r["chunk_index"]
            for adj in [idx - 1, idx + 1]:
                if adj >= 0:
                    cursor.execute(
                        "SELECT content FROM markdown_chunks WHERE doc_title=? AND chunk_index=?",
                        (title, adj)
                    )
                    row = cursor.fetchone()
                    if row:
                        content += " " + row["content"]
        enriched.append({**r, "content": content})
    return enriched

# --- LLM Answer Generation ---
async def generate_answer(question, relevant_results, max_retries=2):
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API_KEY not set")
    context = ""
    for r in relevant_results:
        source_type = "Discourse post" if r["source"] == "discourse" else "Documentation"
        context += f"\n\n{source_type} (URL: {r['url']}):\n{r['content'][:1500]}"
    prompt = (
        "Answer the following question based ONLY on the provided context. "
        "If you cannot answer, say \"I don't have enough information to answer this question.\"\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        "Return your response in this exact format:\n"
        "1. A comprehensive yet concise answer\n"
        "2. A \"Sources:\" section that lists the URLs and relevant text snippets you used to answer\n\n"
        "Sources:\n1. URL: [exact_url_1], Text: [brief quote or description]\n"
        "2. URL: [exact_url_2], Text: [brief quote or description]\n"
    )
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that provides accurate answers based only on the provided context. Always include sources in your response with exact URLs."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3
    }
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data["choices"][0]["message"]["content"]
                    elif response.status == 429:
                        await asyncio.sleep(3 * (attempt + 1))
                    else:
                        raise HTTPException(status_code=response.status, detail=await response.text())
        except Exception as e:
            if attempt == max_retries - 1:
                raise HTTPException(status_code=500, detail=str(e))
            await asyncio.sleep(2)

# --- Multimodal Query Embedding ---
async def process_multimodal_query(question, image_base64):
    if not image_base64:
        return await get_embedding(question)
    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    image_content = f"data:image/jpeg;base64,{image_base64}"
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Look at this image and tell me what you see related to this question: {question}"},
                    {"type": "image_url", "image_url": {"url": image_content}}
                ]
            }
        ]
    }
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                if response.status == 200:
                    data = await response.json()
                    image_desc = data["choices"][0]["message"]["content"]
                    return await get_embedding(f"{question}\nImage context: {image_desc}")
    except Exception:
        pass
    return await get_embedding(question)

# --- LLM Response Parsing ---
def parse_llm_response(response):
    parts = response.split("Sources:", 1)
    if len(parts) == 1:
        for heading in ["Source:", "References:", "Reference:"]:
            if heading in response:
                parts = response.split(heading, 1)
                break
    answer = parts[0].strip()
    links = []
    if len(parts) > 1:
        for line in parts[1].strip().split("\n"):
            line = re.sub(r'^\d+\.\s*', '', line).strip()
            url_match = re.search(r'URL:\s*\[(.*?)\]|url:\s*\[(.*?)\]|\[(http[^\]]+)\]|URL:\s*(http\S+)|url:\s*(http\S+)|(http\S+)', line, re.IGNORECASE)
            text_match = re.search(r'Text:\s*\[(.*?)\]|text:\s*\[(.*?)\]|[""](.*?)[""]|Text:\s*"(.*?)"|text:\s*"(.*?)"', line, re.IGNORECASE)
            url = next((g for g in url_match.groups() if g), "") if url_match else ""
            text = next((g for g in text_match.groups() if g), "Source reference") if text_match else "Source reference"
            if url and url.startswith("http"):
                links.append({"url": url, "text": text})
    return {"answer": answer, "links": links}

# --- API Endpoints ---
@app.post("/query")
async def query_knowledge_base(request: QueryRequest):
    if not API_KEY:
        return JSONResponse(status_code=500, content={"error": "API_KEY not set"})
    conn = get_db_connection()
    try:
        query_embedding = await process_multimodal_query(request.question, request.image)
        relevant_results = await find_similar_content(query_embedding, conn)
        if not relevant_results:
            return {"answer": "I couldn't find any relevant information in my knowledge base.", "links": []}
        enriched_results = await enrich_with_adjacent_chunks(conn, relevant_results)
        llm_response = await generate_answer(request.question, enriched_results)
        result = parse_llm_response(llm_response)
        if not result["links"]:
            unique_urls = set()
            links = []
            for res in relevant_results[:5]:
                url = res["url"]
                if url not in unique_urls:
                    unique_urls.add(url)
                    snippet = res["content"][:100] + "..." if len(res["content"]) > 100 else res["content"]
                    links.append({"url": url, "text": snippet})
            result["links"] = links
        return result
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        conn.close()

@app.get("/health")
async def health_check():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks")
        discourse_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks")
        markdown_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM discourse_chunks WHERE embedding IS NOT NULL")
        discourse_embeddings = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM markdown_chunks WHERE embedding IS NOT NULL")
        markdown_embeddings = cursor.fetchone()[0]
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "api_key_set": bool(API_KEY),
            "discourse_chunks": discourse_count,
            "markdown_chunks": markdown_count,
            "discourse_embeddings": discourse_embeddings,
            "markdown_embeddings": markdown_embeddings
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(status_code=500, content={"status": "unhealthy", "error": str(e), "api_key_set": bool(API_KEY)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

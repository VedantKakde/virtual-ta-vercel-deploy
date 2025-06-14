import os
import json
import sqlite3
import numpy as np
import aiohttp
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional

# Load API key
load_dotenv()
API_KEY = os.getenv("API_KEY")

# DB path
DB_PATH = "knowledge_base.db"

# Similarity threshold
SIMILARITY_THRESHOLD = 0.68

# FastAPI init
app = FastAPI(title="TDS Virtual TA")

# Input/output schemas
class QueryRequest(BaseModel):
    question: str
    image: Optional[str] = None

class Link(BaseModel):
    url: str
    text: str

class QueryResponse(BaseModel):
    answer: str
    links: List[Link]

# DB connection
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Cosine similarity
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Embedding function (using AIPipe)
async def get_embedding(text):
    url = "https://aipipe.org/openai/v1/embeddings"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {"model": "text-embedding-3-small", "input": text}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            res = await resp.json()
            return res["data"][0]["embedding"]

# Search relevant chunks
async def search_db(embedding):
    conn = get_connection()
    cur = conn.cursor()
    results = []

    # Search discourse_chunks
    cur.execute("SELECT * FROM discourse_chunks WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    for row in rows:
        emb = json.loads(row["embedding"])
        sim = cosine_similarity(embedding, emb)
        if sim > SIMILARITY_THRESHOLD:
            results.append({
                "url": row["url"],
                "content": row["content"],
                "similarity": sim
            })

    # Search markdown_chunks
    cur.execute("SELECT * FROM markdown_chunks WHERE embedding IS NOT NULL")
    rows = cur.fetchall()
    for row in rows:
        emb = json.loads(row["embedding"])
        sim = cosine_similarity(embedding, emb)
        if sim > SIMILARITY_THRESHOLD:
            results.append({
                "url": row["original_url"],
                "content": row["content"],
                "similarity": sim
            })

    conn.close()
    return sorted(results, key=lambda x: x["similarity"], reverse=True)[:5]

# Generate final answer using AIPipe Chat API
async def generate_answer(question, context_chunks):
    context = "\n\n".join(chunk["content"] for chunk in context_chunks)

    prompt = (
        "Answer based only on the context provided.\n"
        "If not enough information, say: 'I don't have enough information.'\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )

    url = "https://aipipe.org/openai/v1/chat/completions"
    headers = {"Authorization": API_KEY, "Content-Type": "application/json"}
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that strictly answers based only on the given context."},
            {"role": "user", "content": prompt}
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            res = await resp.json()
            return res["choices"][0]["message"]["content"]

# Main query endpoint
@app.post("/query", response_model=QueryResponse)
async def query_api(req: QueryRequest):
    query_embedding = await get_embedding(req.question)
    relevant_chunks = await search_db(query_embedding)

    if not relevant_chunks:
        return QueryResponse(answer="I don't have enough information to answer.", links=[])

    answer = await generate_answer(req.question, relevant_chunks)

    links = [
        Link(url=chunk["url"], text=chunk["content"][:100] + "...")
        for chunk in relevant_chunks
    ]

    return QueryResponse(answer=answer, links=links)

# Health check (optional)
@app.get("/health")
async def health():
    return {"status": "ok"}

import json
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional
import uvicorn
import httpx
import os
from dotenv import load_dotenv
from contextlib import asynccontextmanager

from logic import get_structured_intent, get_decomposed_queries, strategic_reranker, balanced_select

class RecommendRequest(BaseModel):
    query: str

class AssessmentResponse(BaseModel):
    url: str
    name: str
    adaptive_support: str
    description: str
    duration: Optional[int] = None 
    remote_support: str
    test_type: List[str]

class RecommendResponse(BaseModel):
    recommended_assessments: List[AssessmentResponse]

model_data = {}

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-preview-09-2025:generateContent?key=" + str(api_key)
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Server starting up...")
    model_data['http_client'] = httpx.AsyncClient()
    
    print("Loading SentenceTransformer model...")
    model_data['model'] = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    
    print("Loading vector database...")
    DB_FILE = 'data/assessments_with_embedding.json' # Adjust path
    try:
        with open(DB_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            model_data['assessments'] = data 
            model_data['assessment_embeddings'] = np.array([item['embedding'] for item in data])
        print(f"Successfully loaded {len(model_data['assessments'])} assessments.")
    except Exception as e:
        print(f"CRITICAL ERROR loading database: {e}")

    yield

    print("Server shutting down...")
    await model_data['http_client'].aclose()

# --- FastAPI App ---
app = FastAPI(
    title="SHL Assessment Recommendation System",
    description="An AI-powered RAG API to recommend SHL assessments.",
    version="3.0.0",
    lifespan=lifespan
)
origins = [
    "http://localhost", 
    "http://localhost:3000",  
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For simplicity, allow all. Or use the 'origins' list above for security.
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],
)

# --- API Endpoints ---
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/recommend", response_model=RecommendResponse)
async def recommend_assessments_endpoint(request: RecommendRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is not configured on the server.")
    
    try:
        query_text = request.query
        model = model_data['model']
        assessments = model_data['assessments']
        emb_matrix = model_data['assessment_embeddings']
        client = model_data['http_client']

        # --- RAG PIPELINE ---
        intent = await get_structured_intent(query_text, client, GEMINI_API_URL)
        sub_queries = await get_decomposed_queries(query_text, client, GEMINI_API_URL)
        expert_queries = intent.get('primary_technical_skills', [])
        combined_queries = list(set(sub_queries + expert_queries))
        if not combined_queries: combined_queries = [query_text]

        all_candidates = {}
        for sub_query in combined_queries:
            query_emb = model.encode(sub_query).reshape(1, -1)
            sims = cosine_similarity(query_emb, emb_matrix)[0]
            top_indices = np.argsort(sims)[-40:][::-1]
            for i in top_indices:
                url = assessments[i]['original_data']['url']
                if url not in all_candidates or sims[i] > all_candidates[url]['score']:
                    all_candidates[url] = {'assessment': assessments[i], 'score': sims[i]}
        
        candidates = list(all_candidates.values())
        ranked = strategic_reranker(candidates, intent)
        top_results = balanced_select(ranked, 10)

        recommendations = []
        for item in top_results:
            assessment_data = item['assessment']['original_data']
            recommendations.append(AssessmentResponse(**assessment_data))

        return RecommendResponse(recommended_assessments=recommendations)

    except Exception as e:
        print(f"Error during recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred: {str(e)}")

# To run the server: uvicorn app.main:app --reload
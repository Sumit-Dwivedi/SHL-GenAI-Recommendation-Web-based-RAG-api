import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os     
import pandas as pd
import httpx  
import sys
import time
from dotenv import load_dotenv

# Add the app directory to the path to import logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app.logic import get_structured_intent, get_decomposed_queries, strategic_reranker, balanced_select

# --- SETUP ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY") 
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-preview-09-2025:generateContent?key=" + str(api_key)
)


# ==============================================================================
#  1. QUERY UNDERSTANDING LAYER (Based on Your Analysis)
# ==============================================================================

# --- PROMPT A: For Decomposing the Query into Multiple Search Intents ---
GEMINI_DECOMPOSITION_PROMPT = """
You are a search query decomposition expert. Your task is to break down a user's complex hiring request into a list of simple, self-contained search queries. Each query should focus on a single, distinct aspect (e.g., one for a technical skill, one for a behavioral skill).
Return a JSON object with a single key "decomposed_queries" which is an array of strings.

Example Input: "I am hiring for Java developers who can also collaborate effectively with my business teams."
Example Output:
{
  "decomposed_queries": [
    "Java programming skills assessment for developers",
    "Behavioral assessment for collaboration and teamwork skills"
  ]
}

Example Input: "Senior Data Analyst with 5 years of experience and expertise in SQL, Excel and Python"
Example Output:
{
  "decomposed_queries": [
    "Practical SQL test for senior data analyst",
    "Advanced Excel simulation for data analysis",
    "Python knowledge test for data science"
  ]
}
"""

GEMINI_DECOMPOSITION_SCHEMA = {
    "type": "OBJECT",
    "properties": {"decomposed_queries": {"type": "ARRAY", "items": {"type": "STRING"}}}
}

# --- PROMPT B: For Extracting Structured Filters and Reranking Rules ---
GEMINI_INTENT_PROMPT = """
You are an expert HR Talent Acquisition analyst with deep domain knowledge. Your function is to decompose a hiring query into a structured JSON object.

### Domain-Specific Rules & Heuristics:
1.  **Marketing vs. Sales Distinction:** If the query is for a "Marketing" role (focused on brand, content, strategy), you MUST add "sales" to `negative_keywords` to avoid suggesting sales-quota-based roles.
2.  **Administrative & Entry-Level Roles:** If the query is for an "administrative", "assistant", or general "entry-level" role, the `primary_intent` is "general_role_fit". These roles implicitly require foundational skills. Your search should reflect this by including "Basic Computer Literacy", "Verify - Numerical Ability", and "Verify - Verbal Ability" in the skill extraction.
3.  **I/O Psychology Domain:** If the query mentions "I/O Psychology", "Industrial Psychology", or "psychometrics", the primary tools are SHL's own psychometric assessments. You MUST identify "Occupational Personality Questionnaire OPQ32r" and core cognitive tests like "Verify Numerical" and "Verify Verbal" as the `primary_technical_skills`.
4.  **Disambiguation:** For a "Java developer" query, "javascript" is a `negative_keyword`. For a general sales query, "Salesforce" is a `negative_keyword` because it's a technical tool, not a sales skill.

### Your Task:
Analyze the user's query and use the rules above to populate the following JSON schema. Adhere strictly to the schema.

- primary_technical_skills: List critical technical skills, tools, or specific assessment names derived from the rules.
- primary_soft_skills: List critical behavioral traits.
- job_level: Infer the seniority ("entry-level", "mid-professional", "manager", "senior", "executive").
- job_domain: Infer the business function ("engineering", "sales", "marketing", "hr", "finance", "administrative").
- negative_keywords: List terms to DOWN-rank results, based on the rules.
- primary_intent: Determine the main hiring goal ("technical_screening", "behavioral_fit", "cognitive_ability", "general_role_fit").
"""

GEMINI_INTENT_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "primary_technical_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "primary_soft_skills": {"type": "ARRAY", "items": {"type": "STRING"}},
        "job_level": {"type": "STRING", "nullable": True},
        "job_domain": {"type": "STRING", "nullable": True},
        "negative_keywords": {"type": "ARRAY", "items": {"type": "STRING"}},
        "primary_intent": {"type": "STRING", "nullable": True}
    }
}

def load_data_and_models():
    print("Loading models and data...")
    model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')
    with open('data/assessments_with_embedding.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    assessments = data
    emb_matrix = np.array([item['embedding'] for item in data])
    return model, assessments, emb_matrix

def load_test_queries():
    print("Loading unlabeled test set queries...")
    df = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Test-Set", engine="openpyxl")
    return df['Query'].dropna().tolist()

def generate_predictions():
    if not api_key:
        print("CRITICAL ERROR: GEMINI_API_KEY is missing.")
        return

    model, assessments, emb_matrix = load_data_and_models()
    test_queries = load_test_queries()
    
    results_list = []

    with httpx.Client() as client: # Use a synchronous client for a simple script
        for query in test_queries:
            print(f"Processing query: '{query[:50]}...'")
            
            # --- RAG PIPELINE (Synchronous version) ---
            intent = json.loads(get_structured_intent(query, client, GEMINI_API_URL)) # Adapt to sync
            time.sleep(10) 
            sub_queries = json.loads(get_decomposed_queries(query, client, GEMINI_API_URL)).get('decomposed_queries', [query])
            
            expert_queries = intent.get('primary_technical_skills', [])
            combined_queries = list(set(sub_queries + expert_queries))
            if not combined_queries: combined_queries = [query]

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

            # Add to results list in the required format
            for item in top_results:
                results_list.append({
                    'Query': query,
                    'Assessment_url': item['assessment']['original_data']['url']
                })
            print(f"Query {i+1} processed. Waiting for 5 seconds to respect API rate limits...")
            time.sleep(60) 
    # Create DataFrame and save to CSV
    submission_df = pd.DataFrame(results_list)
    output_filename = 'predictions.csv'
    submission_df.to_csv(output_filename, index=False)
    print(f"\nSUCCESS: Submission file '{output_filename}' created.")
    print("Please check the file to ensure it matches the format in Appendix 3.")

if __name__ == "__main__":
    # A synchronous version of the Gemini calls for the script
    def get_structured_intent(query_text: str, client: httpx.Client, api_url: str) -> str:
        payload = {"contents": [{"parts": [{"text": query_text}]}], "systemInstruction": {"parts": [{"text": GEMINI_INTENT_PROMPT}]}, "generationConfig": {"responseMimeType": "application/json", "responseSchema": GEMINI_INTENT_SCHEMA, "temperature": 0.0}}
        try:
            r = client.post(api_url, json=payload, timeout=30.0); r.raise_for_status()
            return r.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","{}")
        except Exception as e: print(f"Sync Gemini intent failed: {e}"); return "{}"

    def get_decomposed_queries(query_text: str, client: httpx.Client, api_url: str) -> str:
        payload = {"contents": [{"parts": [{"text": query_text}]}], "systemInstruction": {"parts": [{"text": GEMINI_DECOMPOSITION_PROMPT}]}, "generationConfig": {"responseMimeType": "application/json", "responseSchema": GEMINI_DECOMPOSITION_SCHEMA, "temperature": 0.0}}
        try:
            r = client.post(api_url, json=payload, timeout=45.0); r.raise_for_status()
            return r.json().get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","{}")
        except Exception as e: print(f"Sync Gemini decomp failed: {e}"); return '{"decomposed_queries": []}'
        
    generate_predictions()
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import os     
import pandas as pd
import httpx  
import time   
from dotenv import load_dotenv

# --- SCRIPT SETUP ---
load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY") 
print(f"API Key loaded: {'Yes' if api_key else 'No'}")

# Use a stable Gemini model endpoint
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

# --- HELPER FUNCTIONS FOR GEMINI ---

def get_decomposed_queries(query_text: str, client: httpx.Client) -> list:
    payload = {
        "contents": [{"parts": [{"text": query_text}]}],
        "systemInstruction": {"parts": [{"text": GEMINI_DECOMPOSITION_PROMPT}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_DECOMPOSITION_SCHEMA,
            "temperature": 0.0 # Crucial for consistent results
        }
    }
    try:
        r = client.post(GEMINI_API_URL, json=payload, timeout=45.0)
        r.raise_for_status()
        resp = r.json()
        txt = resp.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","{}")
        return json.loads(txt).get("decomposed_queries", [query_text])
    except Exception as e:
        print(f"Gemini decomposition failed: {e}. Falling back to original query.")
        return [query_text]

def get_structured_intent(query_text: str, client: httpx.Client) -> dict:
    payload = {
        "contents": [{"parts": [{"text": query_text}]}],
        "systemInstruction": {"parts": [{"text": GEMINI_INTENT_PROMPT}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_INTENT_SCHEMA,
            "temperature": 0.0 # Crucial for consistent results
        }
    }
    for attempt in range(3):
        try:
            r = client.post(GEMINI_API_URL, json=payload, timeout=30.0)
            r.raise_for_status()
            resp = r.json()
            txt = resp.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","{}")
            return json.loads(txt)
        except Exception as e:
            print(f"Gemini intent extraction failed: {e}")
            time.sleep(120)
    return {}


# ==============================================================================
#  2. RETRIEVAL & RE-RANKING LAYER
# ==============================================================================

def strategic_reranker(candidates: list, intent: dict) -> list:
    """
    Implements the "Improve Retrieval Strategy" recommendations from your analysis.
    """
    final_results = []
    
    tech_skills = intent.get('primary_technical_skills', [])
    soft_skills = intent.get('primary_soft_skills', [])
    job_level = intent.get('job_level')
    primary_intent = intent.get('primary_intent')
    negative_kws = intent.get('negative_keywords', [])

    for cand in candidates:
        assessment = cand['assessment']
        score = cand['score'] # Initial semantic score from retrieval
        text = assessment['search_text'].lower()
        name = assessment['original_data']['name'].lower()

        # Rule 1: Negative Keyword Penalty
        if any(kw.lower() in text for kw in negative_kws):
            score *= 0.5

        # Rule 2: Prioritize Practical Simulations for technical roles
        if primary_intent == 'technical_screening' and 'practical_simulation' in text:
            score *= 1.25

        # Rule 3: Seniority Handling
        if job_level in ["senior", "manager", "executive"] and "appropriate_for_senior" in text:
            score *= 1.15
        if job_level == "entry-level" and "entry_level" not in text:
            score *= 0.9

        # Rule 4: Intent Matching
        if primary_intent == 'behavioral_fit' and 'behavioral' in text:
            score *= 1.2
        
        # Rule 5: Boost direct keyword matches in assessment name
        all_kws = tech_skills + soft_skills
        if any(kw.lower() in name for kw in all_kws):
            score *= 1.1

        if primary_intent == 'general_role_fit' and any(kw in text for kw in ["foundational_numerical", "foundational_verbal", "foundational_computer"]):
            score *= 1.2 # Strong boost for core skills when a general fit is needed

        final_results.append({'assessment': assessment, 'score': score})

    return sorted(final_results, key=lambda x: x['score'], reverse=True)

def get_recommendations(query_text, model, assessments, emb_matrix, gem_client, k=10):
    # Step 1: Extract the structured intent for re-ranking later
    intent = get_structured_intent(query_text, gem_client)
    print(f"\n[Structured Intent]: {intent}")

    # Step 2: Decompose the query into multiple search vectors
    sub_queries = get_decomposed_queries(query_text, gem_client)
    print(f"[Decomposed Queries]: {sub_queries}")

    # Step 3: Retrieve candidates for EACH sub-query and combine them
    all_candidates = {}
    for sub_query in sub_queries:
        query_emb = model.encode(sub_query).reshape(1, -1)
        sims = cosine_similarity(query_emb, emb_matrix)[0]
        
        # Retrieve a smaller pool for each sub-query to get diverse results
        top_indices = np.argsort(sims)[-40:][::-1]
        
        for i in top_indices:
            url = assessments[i]['original_data']['url']
            # Add to a dictionary to auto-deduplicate, keeping the highest score for each unique URL
            if url not in all_candidates or sims[i] > all_candidates[url]['score']:
                all_candidates[url] = {'assessment': assessments[i], 'score': sims[i]}

    candidates = list(all_candidates.values())

    # Step 4: Apply the strategic re-ranker to the combined, richer pool of candidates
    ranked = strategic_reranker(candidates, intent)
    
    # Step 5: Apply balanced selection for final result diversity
    top10 = balanced_select(ranked, k)
    return [x['assessment']['original_data']['url'] for x in top10]

def balanced_select(results, k=10):
    tech = [r for r in results if "technical" in r['assessment']['search_text'].lower()]
    behavior = [r for r in results if "behavioral" in r['assessment']['search_text'].lower()]
    final, added_urls = [], set()
    
    def add_to_final(item):
        url = item['assessment']['original_data']['url']
        if url not in added_urls:
            final.append(item)
            added_urls.add(url)
    
    # Add top 5 tech and top 5 behavioral, ensuring no duplicates
    list(map(add_to_final, tech[:5]))
    list(map(add_to_final, behavior[:5]))
    
    # Fill any remaining spots with the highest-scored overall results
    if len(final) < k:
        list(map(add_to_final, results))
    
    return final[:k]


# ==============================================================================
#  3. UTILITY & EVALUATION FUNCTIONS
# ==============================================================================

def normalize_url(url):
    if not isinstance(url, str): return ""
    normalized = url.lower().strip().replace("https://www.","https://").strip('/')
    # This line is critical to handle the data inconsistency you found
    normalized = normalized.replace("shl.com/solutions/products", "shl.com/products")
    return normalized

def load_ground_truth(_):
    df = pd.read_excel("data/Gen_AI Dataset.xlsx", sheet_name="Train-Set", engine="openpyxl")
    truth = defaultdict(set)
    for _, row in df.iterrows():
        if pd.notna(row.get("Query")) and pd.notna(row.get("Assessment_url")):
            truth[row["Query"]].add(row["Assessment_url"])
    return truth

def load_model_and_data():
    print("Loading embedding model (multi-qa-mpnet-base-dot-v1)...")
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    print("Loading vector database...")
    with open("data/assessments_with_embedding.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    emb = np.array([d['embedding'] for d in data])
    return model, data, emb

def calculate_recall_at_k(predicted_urls, relevant_urls, k=10):
    predicted_set = {normalize_url(url) for url in predicted_urls[:k]}
    relevant_set = {normalize_url(url) for url in relevant_urls}
    if not relevant_set: return 0, set(), set(), predicted_set
    found_hits = predicted_set.intersection(relevant_set)
    missed_hits = relevant_set - predicted_set
    incorrect_predictions = predicted_set - relevant_set
    recall = len(found_hits) / len(relevant_set) if relevant_set else 0
    return recall, found_hits, missed_hits, incorrect_predictions


# ==============================================================================
#  4. MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    if not api_key:
        print("CRITICAL ERROR: GEMINI_API_KEY environment variable is missing.")
        return
    
    print("--- Loading Ground Truth and Models ---")
    truth = load_ground_truth(None)
    model, assessments, emb = load_model_and_data()
    print("--- Setup Complete. Starting Evaluation ---")
    
    recalls = []

    with httpx.Client() as gem_client:
        for i, (q, rel) in enumerate(truth.items()):
            preds = get_recommendations(q, model, assessments, emb, gem_client, k=10)
            r, found, missed, incorrect = calculate_recall_at_k(preds, rel)
            recalls.append(r)

            print(f"\n--- Analysis for Query {i+1}/{len(truth)} ---")
            print(f"QUERY: \"{q[:150]}...\"") # Print truncated query
            print(f"Recall@10 = {r:.2f} (Found {len(found)}/{len(rel)})")

    if recalls:
        print("\n\n--- Evaluation Complete ---")
        print(f"Final Mean Recall@10 (RAG): {sum(recalls)/len(recalls):.4f}")

if __name__ == "__main__":
    main()
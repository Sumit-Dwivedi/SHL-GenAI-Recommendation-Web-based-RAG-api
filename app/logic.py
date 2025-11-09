import httpx
import json
import time

# ==============================================================================
#  1. QUERY UNDERSTANDING LAYER (Prompts & Schemas)
# ==============================================================================

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


# ==============================================================================
#  2. HELPER FUNCTIONS FOR GEMINI API CALLS
# ==============================================================================

async def get_decomposed_queries(query_text: str, client: httpx.AsyncClient, api_url: str) -> list:
    payload = {
        "contents": [{"parts": [{"text": query_text}]}],
        "systemInstruction": {"parts": [{"text": GEMINI_DECOMPOSITION_PROMPT}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_DECOMPOSITION_SCHEMA,
            "temperature": 0.0
        }
    }
    try:
        response = await client.post(api_url, json=payload, timeout=45.0)
        response.raise_for_status()
        resp_json = response.json()
        txt = resp_json.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","{}")
        return json.loads(txt).get("decomposed_queries", [query_text])
    except Exception as e:
        print(f"Gemini decomposition failed: {e}. Falling back to original query.")
        return [query_text]

async def get_structured_intent(query_text: str, client: httpx.AsyncClient, api_url: str) -> dict:
    payload = {
        "contents": [{"parts": [{"text": query_text}]}],
        "systemInstruction": {"parts": [{"text": GEMINI_INTENT_PROMPT}]},
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_INTENT_SCHEMA,
            "temperature": 0.0
        }
    }
    try:
        response = await client.post(api_url, json=payload, timeout=30.0)
        response.raise_for_status()
        resp_json = response.json()
        txt = resp_json.get("candidates",[{}])[0].get("content",{}).get("parts",[{}])[0].get("text","{}")
        return json.loads(txt)
    except Exception as e:
        print(f"Gemini intent extraction failed: {e}")
        return {}

# ==============================================================================
#  3. RE-RANKING & SELECTION LOGIC
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
        score = cand['score']
        text = assessment['search_text'].lower()
        name = assessment['original_data']['name'].lower()

        if any(kw.lower() in text for kw in negative_kws): score *= 0.5
        if primary_intent == 'technical_screening' and 'practical_simulation' in text: score *= 1.25
        if job_level in ["senior", "manager", "executive"] and "appropriate_for_senior" in text: score *= 1.15
        if job_level == "entry-level" and "entry_level" not in text: score *= 0.9
        if primary_intent == 'behavioral_fit' and 'behavioral' in text: score *= 1.2
        if any(kw.lower() in name for kw in (tech_skills + soft_skills)): score *= 1.1
        if primary_intent == 'general_role_fit' and any(kw in text for kw in ["foundational_numerical", "foundational_verbal", "foundational_computer"]): score *= 1.2

        final_results.append({'assessment': assessment, 'score': score})

    return sorted(final_results, key=lambda x: x['score'], reverse=True)

def balanced_select(results: list, k: int = 10) -> list:
    tech = [r for r in results if "technical" in r['assessment']['search_text'].lower()]
    behavior = [r for r in results if "behavioral" in r['assessment']['search_text'].lower()]
    final, added_urls = [], set()
    
    def add_to_final(item):
        url = item['assessment']['original_data']['url']
        if url not in added_urls:
            final.append(item)
            added_urls.add(url)
    
    list(map(add_to_final, tech[:5]))
    list(map(add_to_final, behavior[:5]))
    
    if len(final) < k:
        list(map(add_to_final, results))
    
    return final[:k]





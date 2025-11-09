import json
import pandas as pd
from collections import defaultdict

GROUND_TRUTH_FILE = 'data/Gen_AI Dataset.xlsx' 
TRAIN_SHEET_NAME = 'Train-Set'
ASSESSMENTS_DATA_FILE = 'data/shl_test_solutions_detailed.json'

def normalize_url(url):
    """
    Normalizes a URL for robust comparison. This function MUST be identical
    to the one used in your evaluation.py script for consistent results.
    - Removes protocol and www.
    - Strips trailing slashes.
    - Critically, removes the '/solutions' path segment.
    """
    if not isinstance(url, str):
        return ""
    
    normalized = url.lower().strip()
    
    if normalized.startswith("https://www."):
        normalized = normalized[12:]
    elif normalized.startswith("http://www."):
        normalized = normalized[11:]
    elif normalized.startswith("https://"):
        normalized = normalized[8:]
    elif normalized.startswith("http://"):
        normalized = normalized[7:]
        
    normalized = normalized.replace("shl.com/solutions/products", "shl.com/products")
    
    return normalized.strip('/')

def run_contextual_analysis():
    """
    Analyzes the train-set by correctly matching URLs and displaying the
    assessment content, allowing for a deep dive into the context and patterns.
    """
    
    try:
        df = pd.read_excel(GROUND_TRUTH_FILE, sheet_name=TRAIN_SHEET_NAME, engine="openpyxl")
    except Exception as e:
        print(f"ERROR: Could not read the Excel file. Details: {e}")
        return

    ground_truth = defaultdict(set)
    for _, row in df.iterrows():
        if pd.notna(row.get("Query")) and pd.notna(row.get("Assessment_url")):
            ground_truth[row["Query"]].add(row["Assessment_url"])

    try:
        with open(ASSESSMENTS_DATA_FILE, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
    except Exception as e:
        print(f"ERROR: Could not load assessment data file. Details: {e}")
        return
        
    assessments_by_normalized_url = {
        normalize_url(item['url']): item for item in assessments
    }

    print("--- Starting Contextual Analysis of the Training Set ---\n")
        
    for i, (query, relevant_urls) in enumerate(ground_truth.items()):
        print("="*80)
        print(f"ANALYZING QUERY #{i+1}:")
        print(f"  '{query}'")
        print("="*80)
        print("Human-labeled relevant assessments for this query are:\n")

        for url in relevant_urls:
            normalized_gt_url = normalize_url(url)
            assessment = assessments_by_normalized_url.get(normalized_gt_url)
            
            if assessment:
                print(f"  - Name: {assessment.get('name')}")
                print(f"    Original URL (from JSON): {assessment.get('url')}")
                print(f"    Test Types: {assessment.get('test_type')}")
                print(f"    Description: {assessment.get('description')}")
                print("-" * 40)
            else:
                print(f"  - ❌ WARNING: Could not find assessment for URL: {url}")
                print(f"    (Looked for normalized key: '{normalized_gt_url}')")
                print("-" * 40)
        
        print("\n\n")

if __name__ == "__main__":
    run_contextual_analysis()
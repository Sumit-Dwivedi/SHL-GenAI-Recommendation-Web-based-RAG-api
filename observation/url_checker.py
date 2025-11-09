import json
import pandas as pd
from collections import defaultdict

GROUND_TRUTH_FILE = 'data/Gen_AI Dataset.xlsx'
TRAIN_SHEET_NAME = 'Train-Set'
ASSESSMENTS_DATA_FILE = 'data/shl_test_solutions_detailed.json'

def normalize_url_for_check(url):
    if not isinstance(url, str):
        return ""
    return url.lower().strip().strip('/')

def check_urls():
    print("--- Starting URL Sanity Check ---")

    try:
        with open(ASSESSMENTS_DATA_FILE, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
        scraped_urls = {normalize_url_for_check(item['url']) for item in assessments}
        print(f"Loaded {len(scraped_urls)} unique, normalized URLs from '{ASSESSMENTS_DATA_FILE}'.")
    except Exception as e:
        print(f"ERROR: Could not load assessment data file. {e}")
        return

    try:
        train_df = pd.read_excel(GROUND_TRUTH_FILE, sheet_name=TRAIN_SHEET_NAME, engine="openpyxl")
        ground_truth_urls = train_df['Assessment_url'].dropna().unique()
        print(f"Loaded {len(ground_truth_urls)} unique URLs from the '{TRAIN_SHEET_NAME}' sheet.")
    except Exception as e:
        print(f"ERROR: Could not read the Excel file. Check file name and sheet name. {e}")
        return
        
    missing_count = 0
    for url in ground_truth_urls:
        normalized_gt_url = normalize_url_for_check(url)
        if normalized_gt_url not in scraped_urls:
            if missing_count == 0:
                print("\n--- Found Mismatched URLs ---")
                print("The following URLs from your TRAIN-SET do not exist in your scraped JSON data:")
            print(f"  - MISSING: {url}")
            missing_count += 1
        
    if missing_count == 0:
        print("\n--- URL Check Complete ---")
        print("✅ SUCCESS: All URLs from your training set were found in the scraped data.")
        print("This confirms that URL mismatch is NOT the primary cause of your low score.")
    else:
        print("\n--- URL Check Complete ---")
        print(f"❌ FAILED: Found {missing_count} URLs in your training set that are not in your JSON data.")
        print("This is contributing to your low score. You must fix your scraper or manually correct the data to improve results.")

if __name__ == "__main__":
    check_urls()
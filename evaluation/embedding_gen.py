import json
from sentence_transformers import SentenceTransformer
import time

def create_searchable_text(a):
    """
    Implements the "Enhance Knowledge Base Structure" strategy from the analysis.
    It infers and injects rich metadata into the text for each assessment
    before embedding, creating a much more powerful vector representation.
    """
    name = a.get('name', '')
    desc = a.get('description', '')
    test_types = a.get('test_type', [])
    job_levels = [level.lower() for level in a.get('job_levels', [])]

    # --- 1. Infer "Assessment Style" (Recommendation from Query 10 Analysis) ---
    assessment_style = "knowledge_test"
    if "simulation" in name.lower() or any("simulation" in t.lower() for t in test_types):
        assessment_style = "practical_simulation"
    elif any(t in ["Ability & Aptitude", "Biodata & Situational Judgement"] for t in test_types):
        assessment_style = "aptitude_or_judgment_test"

    # --- 2. Infer "Primary Use" & "Skill Type" (Recommendation from Query 5) ---
    primary_use_keywords = []
    if any(skill in name.lower() for skill in ["java", "python", "sql", "c#", ".net", "javascript", "selenium", "excel"]):
        primary_use_keywords.append("technical_skill")
    if any(trait in name.lower() for trait in ["personality", "leadership", "behavior", "communication", "opq"]):
        primary_use_keywords.append("soft_skill")
        primary_use_keywords.append("behavioral_fit")
    if "english" in name.lower():
        primary_use_keywords.append("core_language_skill")
        
    # --- 3. Infer "Seniority Appropriateness" (Recommendation from Query 10) ---
    seniority_tags = []
    if any(level in ["manager", "director", "executive"] for level in job_levels):
        seniority_tags.append("appropriate_for_senior")
    if "entry-level" in job_levels:
        seniority_tags.append("entry_level")

    # --- 4. Create the final, metadata-rich text block ---
    # This combines all inferred tags with the core content. It's dense and highly searchable.
    
    # --- ADD THIS NEW SECTION at the end of the inference part ---
    # --- 5. Add Core Competency Keywords (for Admin/Entry roles) ---
    core_competencies = []
    if "numerical ability" in name.lower(): core_competencies.append("foundational_numerical_skill")
    if "verbal ability" in name.lower(): core_competencies.append("foundational_verbal_skill")
    if "computer literacy" in name.lower(): core_competencies.append("foundational_computer_skill")

    # --- Update the final text block to include these new tags ---
    metadata_tags = [assessment_style] + primary_use_keywords + seniority_tags + core_competencies
    text_block = (
        f"Name: {name}. "
        f"Metadata: {' '.join(metadata_tags)}. "
        f"Primary assessment types: {', '.join(test_types)}. "
        f"Target job levels: {', '.join(job_levels)}. "
        f"Description: {desc}"
    )

    return " ".join(text_block.split())


def main():
    INPUT_FILE = 'data/shl_test_solutions_detailed.json'
    OUTPUT_FILE = 'data/assessments_with_embedding.json'
    
    print("Loading embedding model (multi-qa-mpnet-base-dot-v1)...")
    model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
    print("Model loaded.")

    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            assessments = json.load(f)
    except Exception as e:
        print(f"Failed to load {INPUT_FILE}:", e)
        return

    print(f"Loaded {len(assessments)} assessments. Generating new embeddings...")
    start = time.time()

    output_data = []
    for i, assessment in enumerate(assessments):
        search_text = create_searchable_text(assessment)
        embedding = model.encode(search_text).tolist()

        output_data.append({
            "original_data": assessment,
            "search_text": search_text,
            "embedding": embedding
        })
        if (i + 1) % 50 == 0:
            print(f"  Processed {i+1}/{len(assessments)}...")

    end = time.time()
    print(f"Done. Took {end - start:.2f} seconds.")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"Saved new vector database to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
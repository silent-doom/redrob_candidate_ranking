import json
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm

def build_offline_artifacts():
    print("Loading Sentence Transformer model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    candidates_data = []
    embeddings_text = []
    
    consulting_firms = {'tcs', 'infosys', 'wipro', 'accenture', 'cognizant', 'capgemini'}
    print("Parsing candidates.jsonl...")
    
    with open('candidates.jsonl', 'rt', encoding='utf-8') as f:
        for line in tqdm(f, total=100000, desc="Extracting Features"):
            if not line.strip(): continue
            cand = json.loads(line)
            
            profile = cand.get('profile', {})
            signals = cand.get('redrob_signals', {})
            history = cand.get('career_history', [])
            skills = cand.get('skills', [])
            
            latest_role_desc = history[0].get('description', '') if history else ''
            semantic_string = f"{profile.get('headline', '')}. {profile.get('summary', '')}. {latest_role_desc}"
            embeddings_text.append(semantic_string)
            
            is_honeypot = 0
            years_exp = profile.get('years_of_experience', 0)
            
            if years_exp > 45:
                is_honeypot = 1
                
            for skill in skills:
                if skill.get('proficiency') in ['advanced', 'expert'] and skill.get('duration_months', 0) == 0:
                    is_honeypot = 1
                    break
            
            companies_worked = {h.get('company', '').lower() for h in history}
            only_consulting = 1 if (len(companies_worked) > 0 and companies_worked.issubset(consulting_firms)) else 0
            
            candidates_data.append({
                'candidate_id': cand.get('candidate_id'),
                'years_of_experience': years_exp,
                'recruiter_response_rate': signals.get('recruiter_response_rate', 0.0),
                'notice_period_days': signals.get('notice_period_days', 90),
                'interview_completion_rate': signals.get('interview_completion_rate', 0.0),
                'is_honeypot': is_honeypot,
                'only_consulting_flag': only_consulting
            })

    print(f"\nGenerating embeddings for {len(embeddings_text)} candidates...")
    embeddings = model.encode(embeddings_text, batch_size=256, show_progress_bar=True)
    
    print("\nNormalizing vectors for Cosine Similarity...")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / norms
    np.save("candidate_vectors.npy", embeddings_normalized)
    
    print("Saving structured features to Parquet...")
    df = pd.DataFrame(candidates_data)
    df.to_parquet("candidate_features.parquet", index=False)
    
    print("Generating Job Description target embedding...")
    ideal_candidate_text = """
    Senior AI Engineer with 5 to 9 years of experience. Strong production background in ML systems, 
    embeddings, retrieval, ranking, vector databases like Pinecone or FAISS, and fine-tuning LLMs. 
    Proven experience shipping end-to-end ranking or recommendation systems at product companies. 
    Fast execution, scrappy product engineering attitude.
    """
    target_embedding = model.encode([ideal_candidate_text])
    target_norm = np.linalg.norm(target_embedding, axis=1, keepdims=True)
    target_embedding_normalized = target_embedding / target_norm
    np.save("jd_target_vector.npy", target_embedding_normalized)
    
    print("\nPhase 1 Complete! Artifacts are ready.")

if __name__ == "__main__":
    build_offline_artifacts()
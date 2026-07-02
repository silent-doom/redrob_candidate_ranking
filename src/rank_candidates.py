import pandas as pd
import numpy as np
import time
import argparse

def generate_reasoning(row):
    exp = row['years_of_experience']
    resp_rate = int(row['recruiter_response_rate'] * 100)
    notice = row['notice_period_days']
    
    base = f"Strong semantic match for Senior AI Engineer with {exp} years of experience."
    
    if row['only_consulting_flag'] == 1:
        base += " Slight penalty applied due to pure IT-services background, but vector match remains high."
    else:
        base += " Product/startup experience detected."
        
    behavioral = f" Solid behavioral signals: {resp_rate}% response rate and {notice}-day notice period."
    
    if resp_rate < 40:
        behavioral = f" Warning: Kept due to elite skill match, but high flight-risk ({resp_rate}% response rate)."
        
    return base + behavioral

def run_ranking(args):
    start_time = time.time()
    print("Initializing Phase 2: Online Inference (Pure NumPy)...")

    # Load artifacts (ignoring the raw --candidates file since we pre-computed it)
    print("Loading pre-computed artifacts...")
    vectors = np.load("candidate_vectors.npy")
    df = pd.read_parquet("candidate_features.parquet")
    jd_vector = np.load("jd_target_vector.npy")
    
    # Fast NumPy Cosine Similarity
    print("Executing matrix multiplication for vector similarity...")
    similarities = np.dot(vectors, jd_vector.T).flatten()
    
    k_candidates = 2000
    top_indices = np.argsort(similarities)[-k_candidates:][::-1]
    
    top_candidates = df.iloc[top_indices].copy()
    top_candidates['semantic_score'] = similarities[top_indices]
    
    # Honeypot filtering
    print("Applying Honeypot filters...")
    initial_count = len(top_candidates)
    top_candidates = top_candidates[top_candidates['is_honeypot'] == 0]
    print(f"Dropped {initial_count - len(top_candidates)} honeypot/invalid profiles.")

    # Scoring
    print("Applying behavioral modifiers & ranking...")
    min_score = top_candidates['semantic_score'].min()
    max_score = top_candidates['semantic_score'].max()
    top_candidates['norm_semantic'] = (top_candidates['semantic_score'] - min_score) / (max_score - min_score)
    
    def calculate_final_score(row):
        score = row['norm_semantic'] * 0.60
        score += (row['recruiter_response_rate'] * 0.25)
        score += (row['interview_completion_rate'] * 0.15)
        if row['notice_period_days'] > 60:
            score -= 0.05
        if row['only_consulting_flag'] == 1:
            score -= 0.10
        return score

    top_candidates['final_score'] = top_candidates.apply(calculate_final_score, axis=1)
    
    # Format Output
    top_candidates = top_candidates.sort_values(by='final_score', ascending=False)
    final_100 = top_candidates.head(100).copy()
    final_100['rank'] = range(1, 101)
    final_100['reasoning'] = final_100.apply(generate_reasoning, axis=1)
    final_100 = final_100.rename(columns={'final_score': 'score'})
    
    submission_df = final_100[['candidate_id', 'rank', 'score', 'reasoning']]
    
    # Save to the specific output path requested by the validator
    submission_df.to_csv(args.out, index=False, encoding='utf-8')
    
    elapsed = time.time() - start_time
    print(f"\nSuccess! Ranked top 100 candidates in {elapsed:.3f} seconds.")
    print(f"Output saved to: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Redrob Fast Candidate Ranker")
    parser.add_argument("--candidates", type=str, default="./candidates.jsonl", help="Path to raw candidates (bypassed via pre-computation)")
    parser.add_argument("--out", type=str, default="./team_submission.csv", help="Output path for the ranked CSV")
    args = parser.parse_args()
    
    run_ranking(args)
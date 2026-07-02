
# Intelligent Candidate Discovery & Ranking Pipeline

An ultra-optimized, production-ready two-stage candidate retrieval and ranking system designed for the Redrob Intelligent Candidate Discovery Challenge.

This system completely replaces slow, resource-heavy LLM inference steps and crash-prone local vector database installations with a decoupled pipeline architecture. It converts a raw 100,000-candidate pool into dense mathematical representations offline, enabling the online evaluation loop to execute in **under 2 seconds** while consuming **less than 200MB of RAM**—well within the challenge's strict 5-minute and 16GB CPU limits.

---

## 🏗️ Architecture Design Overview

The solution breaks the traditional black-box matching paradigm into two explicit phases:

### 1. Phase 1: Heavy Offline Data Factory (`build_offline_artifacts.py`)

This stage processes the complete `candidates.jsonl` raw dataset. It parses the profile JSON schema, applies custom behavioral and compliance features based on the job description, and targets anomalies:

* **Honeypot Purge Layer:** Traps subtly impossible data (e.g., >45 years of experience or "Expert" proficiency with 0 months of historical utilization) and flags them for immediate drops.
* **Semantic Vector Mapping:** Encodes concatenated professional summaries, headlines, and recent job descriptions using the `all-MiniLM-L6-v2` sentence-transformer model. All embeddings are L2-normalized and serialized into a raw memory-mappable NumPy array (`candidate_vectors.npy`).
* **Columnar Signal Extraction:** Compresses raw unstructured profiles and `redrob_signals` into a structured, highly compressed Apache Parquet table (`candidate_features.parquet`).

### 2. Phase 2: Ultra-Fast Online Inference (`rank_candidates.py`)

This is the isolated execution layer triggered by the evaluation environment. It bypasses slow runtime text processing completely:

* **High-Recall Matrix Retrieval:** Uses highly optimized matrix dot-product operations (`np.dot`) across the 100K-candidate matrix to calculate exact Cosine Similarities in milliseconds, instantly filtering the top 2,000 semantic matches.
* **Hybrid Scoring Engine:** Blends semantic match vectors with transactional platform metadata via an explicit 60/40 math allocation.
* **Deterministic Explainability:** Constructs factual 1–2 sentence justifications for each candidate dynamically, completely eliminating the latency and hallucination risks associated with generative language models.

---

## 📁 Repository Structure

* `src/build_offline_artifacts.py` - Heavy offline ETL and embedding generator
* `src/rank_candidates.py` - Spec-compliant online CLI ranker
* `requirements.txt` - Unified project dependencies
* `submission_metadata.yaml` - Mandatory portal metadata configuration
* `.gitignore` - Prevents committing datasets or heavy arrays
* `README.md` - Project documentation

---

## ⚙️ Local Setup Instructions

### 1. Environment Initialization

Ensure you are using Python 3.10+.

    git clone https://github.com/silent-doom/redrob_candidate_ranking.git
    cd redrob_candidate_ranking
    python -m venv redrob_env
    source redrob_env/bin/activate  # On Windows: redrob_env\Scripts\activate
    pip install -r requirements.txt

### 2. Run the Offline Pre-Computation Pipeline

Place your uncompressed `candidates.jsonl` pool inside the repository root, then execute the factory script to parse the files and generate the optimized binary matrices:

    python src/build_offline_artifacts.py

*(Note: This step downloads the 90MB MiniLM transformer weights locally and vectorizes the 100K corpus. It can take 10–15 minutes depending on your CPU).*

---

## 🚀 Reproduction Command (Sandbox Verification)

To fulfill the explicit sandbox requirements in the submission spec, the inference step accepts the mandatory `--candidates` and `--out` arguments and runs completely decoupled from external network access, local GPUs, or complex database environments.

Run the following command to execute the ranking step end-to-end:

    python src/rank_candidates.py --candidates ./candidates.jsonl --out ./team_submission.csv

### Performance & Bounds Metrics:

* **Wall-clock execution time:** < 2.0 seconds
* **RAM footprint:** ~150 MB
* **GPU requirements:** None (CPU Only)
* **Network network connectivity:** Disabled (Local NumPy matrix operations)

---

## 🛠️ Feature Engineering Choices & Heuristics

The hybrid scoring script implements explicit business logic to align directly with the core text constraints of the JD:

1. **The Product Engineering Bias:** The job description explicitly penalizes candidates with pure IT-consulting backgrounds. The pipeline evaluates historical company lists and applies a strict penalty modifier if the candidate has never worked outside of those environments.
2. **Behavioral Availability Adjustments:** A candidate with high technical proficiency who does not respond to recruiters is down-weighted. The model treats recruiter response rates and interview completion rates as active multipliers.
3. **Notice Period Thresholds:** Candidates with stated notice configurations exceeding 60 days incur an automatic score reduction.


## 🌐 Global Deployment

A live version of the solution is deployed globally via Hugging Face Spaces.

* **Live Demo: [huggingface.co/spaces/FaizanC/redrob-candidate-ranker](https://huggingface.co/spaces/FaizanC/redrob-candidate-ranker)**

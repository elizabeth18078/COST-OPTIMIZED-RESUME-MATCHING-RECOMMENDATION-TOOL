from embedding import load_resumes_from_folder
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer


def rank_resumes(job_description, resume_names, index, model, top_k=3):
    jd_embedding = model.encode([job_description])
    distances, indices = index.search(np.array(jd_embedding), top_k)

    results = []
    for i in range(top_k):
        resume_name = resume_names[indices[0][i]]
        score = 1 / (1 + distances[0][i])
        results.append((resume_name, round(score * 100, 2)))

    return results




def generate_dynamic_explanation(job_description, resume_text, resume_name):
    jd_words = set(job_description.lower().split())
    resume_words = set(resume_text.lower().split())

    matched_skills = jd_words.intersection(resume_words)

    top_matches = list(matched_skills)[:5]

    explanation = f"""
The resume '{resume_name}' is ranked highly because it matches important keywords 
from the job description such as: {', '.join(top_matches)}.

This indicates strong alignment between the candidate's profile 
and the job requirements based on semantic similarity analysis.
"""
    return explanation



if __name__ == "__main__":

    # Load FAISS index
    index = faiss.read_index("faiss_index.index")

    # Load resume names
    with open("resume_names.pkl", "rb") as f:
        resume_names = pickle.load(f)

    resume_data = load_resumes_from_folder("data/resumes")

    # Load embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    job_description = input("\nEnter Job Description:\n")

    # IMPORTANT LINE (you were missing this)
    top_resumes = rank_resumes(job_description, resume_names, index, model)

    print("\nTop Matching Resumes:")

    # for rank, (name, score) in enumerate(top_resumes, 1):
    #     print(f"\n{rank}. {name} — Match Score: {score}%")
    #     explanation = generate_explanation(job_description, name)
    #     print("Explanation:", explanation)

    for rank, (name, score) in enumerate(top_resumes, 1):
        print(f"\n{rank}. {name} — Match Score: {score}%")
        resume_text = resume_data[name]
        explanation = generate_dynamic_explanation(job_description, resume_text, name)
        print("Explanation:", explanation)

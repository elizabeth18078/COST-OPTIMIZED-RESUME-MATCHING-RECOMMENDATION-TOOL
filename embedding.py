import pickle
import os
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


# -------- PDF EXTRACTION --------
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                
                page_text = page.extract_text()
                if page_text:
                    text += page_text + " "
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text.strip()


def load_resumes_from_folder(folder_path):
    resumes = {}
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".pdf"):
            file_path = os.path.join(folder_path, file_name)
            print(f"Processing: {file_name}")
            text = extract_text_from_pdf(file_path)
            if text:
                resumes[file_name] = text
    return resumes


# -------- EMBEDDING FUNCTION --------
def create_embeddings(resume_dict):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    resume_names = list(resume_dict.keys())
    resume_texts = list(resume_dict.values())

    print("\nGenerating embeddings...")
    embeddings = model.encode(resume_texts)

    return resume_names, embeddings


# -------- FAISS STORAGE --------
def store_in_faiss(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    return index


# -------- MAIN --------
if __name__ == "__main__":
    folder_path = "data/resumes"
    resume_data = load_resumes_from_folder(folder_path)

    print("\nTotal Resumes Loaded:", len(resume_data))

    resume_names, embeddings = create_embeddings(resume_data)

    index = store_in_faiss(embeddings)

    print("\nFAISS index created successfully!")
    print("Total vectors stored:", index.ntotal)
    # Save FAISS index
    faiss.write_index(index, "faiss_index.index")

# Save resume names
    with open("resume_names.pkl", "wb") as f:
        pickle.dump(resume_names, f)

    print("Embeddings saved successfully!")


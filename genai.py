import requests

def generate_ai_evaluation(job_description, resume_text):

    prompt = f"""
You are an HR expert.

Job Description:
{job_description}

Resume:
{resume_text}

Provide:
1. Matching strengths
2. Missing skills
3. Suitability score out of 10
4. 3 technical interview questions
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "phi3",
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"]

    except Exception as e:
        return f"Error connecting to Ollama: {e}"

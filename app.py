from flask import Flask, render_template, request
import ranking  # your ranking logic file
import requests

app = Flask(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"

def generate_ai_response(prompt):
    response = requests.post(OLLAMA_URL, json={
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    })
    return response.json()["response"]

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        job_desc = request.form["job_desc"]

        # Get top resumes
        top_resumes = ranking.get_top_matches(job_desc)

        prompt = f"""
        Job Description:
        {job_desc}

        Top Candidates:
        {top_resumes}

        Explain why these candidates are suitable.
        """

        ai_response = generate_ai_response(prompt)

        return render_template("index.html", result=ai_response)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
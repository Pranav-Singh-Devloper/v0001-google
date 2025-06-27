import os
import json
from openai import OpenAI
from oid_to_str import oid_to_str

def analyze_match(jobs: list, student_data: list) -> str:
    """
    Analyze job matches for a given student using Together AI and DeepSeek model.
    """

    TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
    MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

    if not TOGETHER_API_KEY:
        return "❌ TOGETHER_API_KEY is missing."

    client = OpenAI(
        base_url="https://api.together.xyz/v1",
        api_key=TOGETHER_API_KEY
    )

    # ✅ Sanitize inputs and simplify if needed
    jobs_clean = [oid_to_str(job) for job in jobs]
    student_clean = oid_to_str(student_data[0])

    if not jobs_clean:
        return "❌ No jobs to analyze."

    # ✅ Prepare JSON payloads from cleaned versions only
    student_name = " ".join(filter(None, [student_clean.get("first_name"), student_clean.get("last_name")]))
    jobs_json = json.dumps(jobs_clean, indent=2)
    student_json = json.dumps(student_clean, indent=2)

    # Prompt Construction
    system_prompt = f"""
You are an expert career advisor and the world’s most accurate job matcher. You will analyze the following JSON array of job postings and a JSON student profile.

Steps:
1. Evaluate job relevance based on role type, domain, skills, etc.
2. Assign a Match Score (0–100%) for how well the student fits each job.
3. Provide a friendly report for the top jobs:
   • Title, Company, Score
   • ✅ Why it’s a good fit
   • ⚠️ Mismatches

Use bullet points, emojis, and structure the output cleanly.
""" + "\n\nJob Postings:\n" + jobs_json

    user_prompt = "\n\nStudent Profile:\n" + student_json

    try:
        print("📡 Sending to LLM...")
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        print("✅ LLM response received.")
        return resp.choices[0].message.content

    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ LLM processing failed: {repr(e)}"

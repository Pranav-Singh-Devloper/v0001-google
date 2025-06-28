import os
import json
import logging
from openai import OpenAI, NotFoundError
from oid_to_str import oid_to_str

logger = logging.getLogger("llm_analyzer")
logging.basicConfig(level=logging.INFO)

def analyze_match(jobs: list, student_data: list) -> str:
    print("▶️ Starting analyze_match")

    # 1. Load API key
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    print(f"🔑 OPENROUTER_API_KEY: {openrouter_key!r}")
    if not openrouter_key:
        print("❌ Missing OPENROUTER_API_KEY")
        return "❌ OPENROUTER_API_KEY is missing."

    # 2. Sanitize input
    print(f"🗂 Received {len(jobs)} job(s) and {len(student_data)} student record(s)")
    if not student_data:
        print("❌ student_data empty")
        return "❌ No student data provided."
    jobs_clean = [oid_to_str(job) for job in jobs if isinstance(job, dict)]
    print(f"🧹 jobs_clean length: {len(jobs_clean)}")
    if not jobs_clean:
        print("❌ No valid jobs after sanitize")
        return "❌ No jobs to analyze."
    student_clean = oid_to_str(student_data[0])
    print(f"👤 student_clean keys: {list(student_clean.keys())}")

    # 3. Build prompts
    jobs_json    = json.dumps(jobs_clean, indent=2, separators=(",", ":"))
    student_json = json.dumps(student_clean, indent=2, separators=(",", ":"))
    system_prompt = (
        "You are an expert career advisor and the world’s most accurate job matcher.\n"
        f"Job Postings:\n{jobs_json}"
    )
    user_prompt = f"Student Profile:\n{student_json}"
    print("✍️ Prompts built (system & user)")

    # 4. Instantiate client WITH all required headers (including Authorization)
    print("🔧 Instantiating OpenRouter client with full headers")
    client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_key,
    default_headers={
        "Authorization": f"Bearer {openrouter_key}",
        "HTTP-Referer": "https://v0001-google-production.up.railway.app",
        "X-Title": "JobMatch AI"
        }
    )


    primary_model  = "deepseek/deepseek-r1:free"
    fallback_model = "tngtech/deepseek-r1t-chimera:free"

    # 5. Try primary
    try:
        print(f"📡 Sending to OpenRouter (model: {primary_model})")
        resp = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user",   "content":user_prompt}
            ],
            temperature=0.1
        )
        print("✅ Primary model call succeeded")
        return resp.choices[0].message.content

    except NotFoundError as e:
        print(f"⚠️ Primary model not found: {e}")

    except Exception as e:
        print(f"❌ Primary model call exception: {type(e).__name__}: {e}")
        return "❌ Primary model failed."

    # 6. Try fallback
    try:
        print(f"🔁 Retrying with fallback model: {fallback_model}")
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role":"system", "content":system_prompt},
                {"role":"user",   "content":user_prompt}
            ],
            temperature=0.1
        )
        print("✅ Fallback model call succeeded")
        return resp.choices[0].message.content

    except Exception as e:
        print(f"❌ Fallback model exception: {type(e).__name__}: {e}")
        return "❌ Both primary and fallback models failed."

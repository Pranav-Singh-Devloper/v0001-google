import os
import json
import logging
from openai import OpenAI, NotFoundError
from oid_to_str import oid_to_str

logger = logging.getLogger("llm_analyzer")
logging.basicConfig(level=logging.INFO)

def analyze_match(jobs: list, student_data: list) -> str:
    """
    Analyze job matches for a given student using DeepSeek models via OpenRouter.
    """
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        return "❌ OPENROUTER_API_KEY is missing."

    # Sanitize input
    if not student_data:
        return "❌ No student data provided."
    jobs_clean = [oid_to_str(job) for job in jobs if isinstance(job, dict)]
    if not jobs_clean:
        return "❌ No jobs to analyze."
    student_clean = oid_to_str(student_data[0])

    # Prepare prompts
    jobs_json = json.dumps(jobs_clean, indent=2, separators=(",", ":"))
    student_json = json.dumps(student_clean, indent=2, separators=(",", ":"))

    system_prompt = (
        "You are an expert career advisor and the world’s most accurate job matcher. "
        "Analyze the following JSON array of job postings and a JSON student profile.\n\n"
        "1. Evaluate relevance (role, domain, skills).\n"
        "2. Assign a Match Score (0–100%).\n"
        "3. Report the top fits with:\n"
        "   • Title, Company, Score\n"
        "   • ✅ Why it’s a good fit\n"
        "   • ⚠️ Mismatches\n\n"
        f"Job Postings:\n{jobs_json}"
    )
    user_prompt = f"Student Profile:\n{student_json}"

    base_url = "https://openrouter.ai/api/v1"
    client = OpenAI(base_url=base_url, api_key=openrouter_key)

    primary_model = "deepseek/deepseek-r1:free"
    fallback_model = "deepseek/deepseek-v3-base:free"

    # Try primary
    try:
        logger.info(f"📡 Sending to OpenRouter (model: {primary_model})...")
        resp = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        logger.info("✅ Primary model response received.")
        return resp.choices[0].message.content

    except NotFoundError:
        logger.warning(f"⚠️ {primary_model} not available, trying {fallback_model}...")

    except Exception:
        logger.exception("❌ Primary model call failed.")
        return "❌ Primary model failed."

    # Try fallback
    try:
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1
        )
        logger.info("✅ Fallback model response received.")
        return resp.choices[0].message.content

    except Exception:
        logger.exception("❌ Fallback model call failed.")
        return "❌ Both primary and fallback models failed."

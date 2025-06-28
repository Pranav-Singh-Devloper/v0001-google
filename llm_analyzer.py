import os
import json
import logging
from openai import OpenAI, OpenAIError
from oid_to_str import oid_to_str

logger = logging.getLogger("llm_analyzer")
logging.basicConfig(level=logging.INFO)

def analyze_match(jobs: list, student_data: list) -> str:
    """
    Analyze job matches for a given student using DeepSeek models via OpenRouter.
    """
    logger.info("▶️ Starting analyze_match")

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    logger.info(f"🔑 OPENROUTER_API_KEY starts with: {openrouter_key[:10] if openrouter_key else 'None'}")
    if not openrouter_key:
        return "❌ OPENROUTER_API_KEY is missing."

    if not student_data:
        return "❌ No student data provided."
    logger.info(f"🗂 Received {len(jobs)} job(s) and {len(student_data)} student record(s)")

    jobs_clean = [oid_to_str(job) for job in jobs if isinstance(job, dict)]
    if not jobs_clean:
        return "❌ No jobs to analyze."
    logger.info(f"🧹 jobs_clean length: {len(jobs_clean)}")

    student_clean = oid_to_str(student_data[0])
    logger.info(f"👤 student_clean keys: {list(student_clean.keys())}")

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
    logger.info("✍️ Prompts built (system & user)")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=openrouter_key
    )
    logger.info("🔧 Instantiating OpenRouter client")

    primary_model = "deepseek/deepseek-r1:free"
    fallback_model = "tngtech/deepseek-r1t-chimera:free"

    # Try primary model
    try:
        logger.info(f"📡 Sending to OpenRouter (model: {primary_model})")
        resp = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        logger.info("✅ Primary model response received.")
        return resp.choices[0].message.content

    except OpenAIError as e:
        logger.error(f"❌ Primary model call exception: {type(e).__name__}: {e}")

    # Try fallback model
    try:
        logger.warning(f"⚠️ Trying fallback model: {fallback_model}")
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        logger.info("✅ Fallback model response received.")
        return resp.choices[0].message.content

    except Exception as e:
        logger.exception("❌ Both primary and fallback model calls failed.")
        return "❌ Both primary and fallback models failed."

import os
import json
import logging
from openai import OpenAI, OpenAIError
from oid_to_str import oid_to_str

logger = logging.getLogger("llm_analyzer")
logging.basicConfig(level=logging.INFO)

def analyze_match(jobs: list, student_data: list) -> str:
    logger.info("▶️ Starting analyze_match")

    # 1) Read the OpenRouter key
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    logger.info(f"🔑 OPENROUTER_API_KEY starts with: {openrouter_key[:8]!r}" if openrouter_key else "🔑 No key")
    if not openrouter_key:
        return "❌ OPENROUTER_API_KEY is missing."

    # 2) Also set it as OPENAI_API_KEY so the SDK picks it up unconditionally
    os.environ["OPENAI_API_KEY"] = openrouter_key

    # 3) Sanitize inputs
    if not student_data:
        return "❌ No student data provided."
    logger.info(f"🗂 Received {len(jobs)} job(s) and {len(student_data)} student(s)")
    jobs_clean = [oid_to_str(job) for job in jobs if isinstance(job, dict)]
    if not jobs_clean:
        return "❌ No jobs to analyze."
    logger.info(f"🧹 jobs_clean length: {len(jobs_clean)}")
    student_clean = oid_to_str(student_data[0])
    logger.info(f"👤 student_clean keys: {list(student_clean.keys())}")

    # 4) Build prompts
    jobs_json    = json.dumps(jobs_clean, indent=2, separators=(",", ":"))
    student_json = json.dumps(student_clean, indent=2, separators=(",", ":"))
    system_prompt = (
        "You are an expert career advisor and the world’s most accurate job matcher.\n\n"
        f"Job Postings:\n{jobs_json}"
    )
    user_prompt = f"Student Profile:\n{student_json}"
    logger.info("✍️ Prompts built")

    # 5) Instantiate client (SDK will use OPENAI_API_KEY automatically)
    client = OpenAI(base_url="https://openrouter.ai/api/v1")
    logger.info("🔧 OpenRouter client instantiated")

    primary_model  = "deepseek/deepseek-r1:free"
    fallback_model = "deepseek/deepseek-v3-base:free"

    # 6) Try primary
    try:
        logger.info(f"📡 Sending to OpenRouter (model={primary_model})")
        resp = client.chat.completions.create(
            model=primary_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        logger.info("✅ Primary response received")
        return resp.choices[0].message.content

    except OpenAIError as e:
        logger.error(f"❌ Primary model call failed: {e}")

    # 7) Try fallback
    try:
        logger.warning(f"⚠️ Retrying with fallback model={fallback_model}")
        resp = client.chat.completions.create(
            model=fallback_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1024
        )
        logger.info("✅ Fallback response received")
        return resp.choices[0].message.content

    except Exception as e:
        logger.exception("❌ Both primary & fallback calls failed")
        return "❌ Both primary and fallback models failed."

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import json
from dotenv import load_dotenv
from datetime import datetime
from supabase import create_client
from mangum import Mangum
import logging
from pymongo import MongoClient
from llm_analyzer import analyze_match
from oid_to_str import oid_to_str
from fastapi.responses import JSONResponse

# Load environment
load_dotenv()

# MongoDB client setup
MONGO_URI = os.getenv("MONGODB_URI")
client = MongoClient(MONGO_URI)
db = client["test"]
jobs_collection = db["v0001-collection"]

# Initialize Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ——— Your existing BM25 endpoint ———

bm25 = None
job_index = None
jobs = None

class ProfileRequest(BaseModel):
    intern_name: str
    students: List[Dict[str, Any]]
    interests: str

def simplify_job(job):
    return {
        "title": job.get("title"),
        "companyName": job.get("companyName"),
        "jobDescription": job.get("jobDescription"),
        "tagsAndSkills": job.get("tagsAndSkills"),
        "location": job.get("location"),
        "jobType": job.get("jobType"),
        "jdURL": job.get("jdURL"),
        "score": job.get("score")
    }

@app.get("/")
def read_root():
    return {"message": "🎉 FastAPI app is live!"}

@app.get("/health")
def health_check():
    return {"status": "ok"}


# ——— New Atlas Search endpoint ———
def build_internship_pipeline(interests: List[str],
                              skills: List[str],
                              preferred_locations: List[str]):
    # join interests & skills into text queries
    interests_q = " ".join(interests)
    skills_q    = ", ".join(skills)

    return [
        {
            "$search": {
                "index": "default",
                "compound": {
                    "should": [
                        {
                            "text": {
                                "query": interests_q,
                                "path": [
                                    "title",
                                    "jobDescription",
                                    "tagsAndSkills",
                                    "companyName",
                                    "ambitionBoxData.Url"
                                ],
                                "fuzzy": { "maxEdits": 1 }
                            }
                        },
                        {
                            "text": {
                                "query": skills_q,
                                "path": ["tagsAndSkills", "jobDescription"],
                                "fuzzy": { "maxEdits": 1 }
                            }
                        }
                    ],
                    "minimumShouldMatch": 1
                }
            }
        },
        {
            # hybrid internship filter:
            "$match": {
                "$or": [
                    { "jobType":         { "$regex": "intern", "$options": "i" } },
                    { "title":           { "$regex": "intern", "$options": "i" } },
                    { "jobDescription":  { "$regex": "intern", "$options": "i" } },
                    { "jdURL":           { "$regex": "intern", "$options": "i" } },
                    { "companyJobsUrl":  { "$regex": "intern", "$options": "i" } },
                    { "tagsAndSkills":   { "$regex": "intern", "$options": "i" } }
                ],
                # location filter
                "location": { "$in": preferred_locations }  
            }
        },
        {
            "$addFields": {
                "score": { "$meta": "searchScore" }
            }
        },
        { "$sort":  { "score": -1 } },
        { "$limit": 10 }
    ]

@app.post("/search-mdb")
def search_mdb(request: ProfileRequest) -> List[Dict[str, Any]]:
    """
    Performs an Atlas Search across your jobs_collection,
    using the student's interests and skills, filters by:
      • preferred_locations
      • employment_type (e.g. Internship)
    and returns each full document plus `job_id` and `score`.
    """
    student    = request.students[0]
    prefs      = student.get("job_preferences", {})
    locations  = prefs.get("preferred_locations", [])
    employment = prefs.get("employment_type", [])  # e.g. ["Internship"]
    interests  = prefs.get("interests", []) or [request.interests]
    skills     = student.get("skills", [])
    skills_q   = ", ".join(skills)

    pipeline = build_internship_pipeline(
        interests=request.students[0]["job_preferences"]["interests"],
        skills=request.students[0]["skills"],
        preferred_locations=request.students[0]["job_preferences"]["preferred_locations"],
    )
    try:
        docs = list(jobs_collection.aggregate(pipeline))

        # 1) Convert ObjectIds to strings
        clean_docs = [oid_to_str(doc) for doc in docs]

        # 2) Rename _id to job_id
        for d in clean_docs:
            if "_id" in d:
                d["job_id"] = d.pop("_id")

        # 3) Simplify jobs (optional)
        copy = [simplify_job(job) for job in clean_docs[:15]]
        simplified_docs = copy[:10]

        # 4) Run LLM analyzer
        analysis = analyze_match(simplified_docs, request.students)

        return JSONResponse(content={"analysis": analysis, "mongodb_result": copy})

    except Exception as e:
        logging.exception("Atlas Search failed")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/mongo-only")
def mongo_only_results(request: ProfileRequest) -> List[Dict[str, Any]]:
    """
    Returns raw MongoDB Atlas Search results (simplified),
    without LLM analysis.
    """
    try:
        student    = request.students[0]
        prefs      = student.get("job_preferences", {})
        interests  = prefs.get("interests", []) or [request.interests]
        skills     = student.get("skills", [])
        locations  = prefs.get("preferred_locations", [])

        pipeline = build_internship_pipeline(
            interests=interests,
            skills=skills,
            preferred_locations=locations,
        )

        docs = list(jobs_collection.aggregate(pipeline))

        # Convert ObjectIds to strings
        clean_docs = [oid_to_str(doc) for doc in docs]

        # Rename _id to job_id
        for d in clean_docs:
            if "_id" in d:
                d["job_id"] = d.pop("_id")

        # Simplify for frontend
        simplified = [simplify_job(job) for job in clean_docs[:10]]

        return JSONResponse(content={"mongodb_result": simplified})

    except Exception as e:
        logging.exception("Mongo-only search failed")
        raise HTTPException(status_code=500, detail=str(e))


# AWS Lambda support (Mangum)
handler = Mangum(app)

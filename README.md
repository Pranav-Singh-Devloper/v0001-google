# 📚 Backend Documentation

> In‑depth guide to the FastAPI LLM backend: structure, endpoints, integrations, and deployment.

---

## 🔎 Table of Contents

1. [Overview](#-overview)  
2. [Architecture & Components](#-architecture--components)  
3. [Prerequisites](#-prerequisites)  
4. [Installation & Setup](#-installation--setup)  
   - [Clone & Dependencies](#clone--dependencies)  
   - [Environment Variables](#environment-variables)  
5. [Code Structure](#-code-structure)  
6. [API Endpoints](#-api-endpoints)  
7. [LLM Integration](#-llm-integration)  
8. [External Services & APIs](#-external-services--apis)  
9. [Deployment (Railway)](#-deployment-railway)  
10. [Local Development & Testing](#-local-development--testing)  
11. [Troubleshooting & FAQs](#-troubleshooting--faqs)  
12. [Contributing](#-contributing)  
13. [License](#-license)  

---

## 🌟 Overview

This backend powers a student–job matching service. It:

- Accepts student profiles via REST.  
- Searches a MongoDB Atlas jobs collection using Atlas Search.  
- Optionally runs BM25 (via `rank-bm25`) for keyword matches.  
- Enriches top results with LLM‑driven match analysis.  
- Returns both raw search hits and AI‑generated evaluations.  

All built on FastAPI with Pydantic models for input validation.

---

## 🏗 Architecture & Components

```text
[Client] → FastAPI App
               │
         ┌─────┴─────┐
         │ Middleware│
         └─────┬─────┘
               │
     ┌─────────┴─────────┐
     │   /search-mdb     │───► MongoDB Atlas (Atlas Search)
     │   /mongo-only     │───►
     │   (health, root)  │
     └─────────┬─────────┘
               │
        LLM Analyzer Module
         (OpenRouter API)
               │
        Supabase (for other data ops)
               │
        Deployed on Railway
               │
         (UVicorn + Mangum)
```

Key modules:

- **`main.py`**: route definitions, Mongo & Supabase clients, CORS, BM25/Atlas Search logic.  
- **`llm_analyzer.py`**: wraps OpenRouter/OpenAI SDK calls to deepseek models.  
- **`oid_to_str.py`**: helper to convert Mongo ObjectIds to strings.

---

## 📋 Prerequisites

- **Python** ≥ 3.9  
- **MongoDB Atlas** cluster (with Atlas Search index “default”)  
- **Railway** account (free tier gives 1 GB RAM VM trial for one month)  
- **OpenRouter API Key** (set as `OPENROUTER_API_KEY`)  
- **Supabase** project (for any supplementary data; `SUPABASE_URL` & `SUPABASE_KEY`)  

---

## ⚙️ Installation & Setup

### Clone & Dependencies

```bash
git clone https://github.com/your‑org/your‑backend.git
cd your‑backend
pip install -r requirements.txt
```

`requirements.txt` includes:

- FastAPI, Uvicorn, Mangum  
- Pydantic, python-dotenv  
- supabase-py, pymongo, openai (OpenRouter)  
- rank-bm25, nltk, beautifulsoup4  

### Environment Variables

Create a `.env` file:

```dotenv
MONGODB_URI=<your_mongodb_atlas_uri>
SUPABASE_URL=<your_supabase_url>
SUPABASE_KEY=<your_supabase_key>
OPENROUTER_API_KEY=<your_openrouter_api_key>
```

Railway will auto‑inject `$PORT` and propagate your Git‑committed `railway.json` config.

---

## 📂 Code Structure

```
.
├── main.py             # FastAPI routes & search logic
├── llm_analyzer.py     # OpenRouter LLM orchestration
├── oid_to_str.py       # BSON ObjectId → string helper
├── requirements.txt    # Python deps
├── railway.json        # Railway build/deploy config
└── .env                # Local env variables
```

---

## 🚀 API Endpoints

| Method | Path            | Description                                                     |
| :----- | :-------------- | :-------------------------------------------------------------- |
| `GET`  | `/`             | Root health check: 🎉 “App is live!”                             |
| `GET`  | `/health`       | Returns `{ "status": "ok" }`                                    |
| `POST` | `/search-mdb`   | Atlas Search + LLM analysis. Returns `{ analysis, mongodb_result }` |
| `POST` | `/mongo-only`   | Atlas Search only. Returns raw `{ mongodb_result }`             |

### Request Schema

```json
{
  "intern_name": "Alice",
  "students": [
    {
      "name": "Alice",
      "skills": ["Python","FastAPI"],
      "job_preferences": {
        "interests":["AI","Web3"],
        "preferred_locations":["Bangalore"],
        "employment_type":["Internship"]
      }
    }
  ],
  "interests": "AI"
}
```

---

## 🤖 LLM Integration

- **Provider:** OpenRouter (wrapping OpenAI SDK)  
- **Primary model:** `deepseek/deepseek-r1:free`  
- **Fallback model:** `tngtech/deepseek-r1t-chimera:free`  
- **Endpoint:** `https://openrouter.ai/api/v1`  
- **Logic:** build system/user prompts, call chat completion with `temperature=0.4`, parse JSON‑array output.

---

## 🔗 External Services & APIs

- **MongoDB Atlas** for job postings & Atlas Search queries.  
- **Supabase** client initialized but used for future extensions.  
- **OpenRouter** (OpenAI-compatible) for career‐match analysis.  
- **Rank‑BM25 & NLTK** for optional keyword‐based ranking.  
- **Mangum** adapter to support AWS Lambda environments (optional).

---

## 🚢 Deployment (Railway)

Railway uses `railway.json`:

```json
{
  "build": {
    "builder": "NIXPACKS",
    "nixpacksConfig": { "install": { "cmds": ["pip install -r requirements.txt"] } }
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "runtime": "V2",
    "numReplicas": 1,
    "restartPolicyType":"ON_FAILURE"
  }
}
```

- **Free trial:** 1 GB RAM VM for 30 days.  
- Push to GitHub, connect Railway, set env vars in the Railway dashboard, and deploy.

---

## 🛠 Local Development & Testing

```bash
uvicorn main:app --reload
```

- Navigate to `http://localhost:8000/docs` for interactive Swagger UI.  
- Add tests under `tests/` and run with `pytest`.

---

## ❓ Troubleshooting & FAQs

- **“OPENROUTER_API_KEY missing”**: ensure `.env` is loaded and Railway secrets are set.  
- **Atlas Search errors**: verify your Atlas Search index name and cluster permissions.  
- **Model call failures**: check logs in `llm_analyzer.py`.

---

## 🤝 Contributing

1. Fork & clone.  
2. Create a feature branch.  
3. Write code + tests.  
4. Submit a PR, referencing related issues.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](./LICENSE).

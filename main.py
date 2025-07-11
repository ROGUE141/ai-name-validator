from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os
import json

# ──────────────────────────────────────────────────────────
# FastAPI setup
# ──────────────────────────────────────────────────────────
app = FastAPI()

# Load your API Key from Render’s environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"          # You can change to gpt-4o or gpt-3.5-turbo

# ──────────────────────────────────────────────────────────
# Pydantic schema for incoming requests
# ──────────────────────────────────────────────────────────
class NameValidationRequest(BaseModel):
    names: List[str]

# ──────────────────────────────────────────────────────────
# Helper – decide if human review needed
# ──────────────────────────────────────────────────────────
def needs_human_review(score: int) -> bool:
    """Flag review if confidence < 7."""
    return score < 7

# ──────────────────────────────────────────────────────────
# POST /validate  →  returns list of validation results
# ──────────────────────────────────────────────────────────
@app.post("/validate")
def validate_names(payload: NameValidationRequest) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []

    for raw_name in payload.names:
        try:
            # ‼️ Prompt instructs the model to reply ONLY in JSON
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an expert in American first-name validation. "
                        "Reply ONLY in valid JSON, no extra text."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "Assess the following string strictly as an American "
                        f"first name: \"{raw_name}\" - Return JSON with keys "
                        '{"name": str, "valid": bool, "score": int}. '
                        "Score 0-10 (10 = certain)."
                    ),
                },
            ]

            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0.0,   # deterministic
            )

            # Assistant’s reply (string)
            content = response.choices[0].message.content.strip()

            # Parse it safely
            data = json.loads(content)

            # Build final record
            results.append(
                {
                    "input": raw_name,
                    "name": data.get("name", raw_name),
                    "valid": data.get("valid", False),
                    "score": data.get("score", 0),
                    "human_review": needs_human_review(data.get("score", 0)),
                }
            )

        except Exception as exc:
            # Capture any parsing / API errors
            results.append(
                {"input": raw_name, "name": raw_name, "error": str(exc)}
            )

    return {"results": results}

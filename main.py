from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os
import json
import re

app = FastAPI()

# Load API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

class NameValidationRequest(BaseModel):
    names: List[str]

@app.post("/validate")
def validate_names(data: NameValidationRequest):
    results = []

    for full_input in data.names:
        # Extract just the first name
        name_only = extract_first_name(full_input)

        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that validates if a given string is a real American first name. Respond ONLY in strict JSON like this: {\"name\": \"Dylan\", \"valid\": true, \"score\": 9}"},
                    {"role": "user", "content": f"Is '{name_only}' a valid American first name? Respond only with JSON."}
                ]
            )

            ai_output = response["choices"][0]["message"]["content"]

            # Try to parse JSON safely
            parsed = json.loads(ai_output)

            # Add human_review flag based on confidence
            score = parsed.get("score", 0)
            human_review = score < 7

            results.append({
                "input": full_input,
                "name": parsed.get("name", name_only),
                "valid": parsed.get("valid", False),
                "score": score,
                "human_review": human_review
            })

        except Exception as e:
            results.append({
                "input": full_input,
                "name": name_only,
                "error": str(e)
            })

    return {"results": results}

def extract_first_name(raw: str) -> str:
    """
    Cleans and extracts the first name from a full string.
    Handles cases like "John Smith", "Smith, John", "Dylan and Shaya"
    """
    # Split on comma
    if ',' in raw:
        parts = raw.split(',')
        return parts[1].strip() if len(parts) > 1 else parts[0].strip()

    # Split on "and" or space and take first
    parts = re.split(r'\s+and\s+|\s+', raw)
    return parts[0].strip()

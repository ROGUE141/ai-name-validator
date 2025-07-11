from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os
import re

app = FastAPI()

# Load API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

class NameValidationRequest(BaseModel):
    names: List[str]

def split_names(name: str) -> List[str]:
    # Splits on comma, " and ", or space if it looks like two names
    if ',' in name:
        return [part.strip() for part in name.split(',')]
    elif ' and ' in name:
        return [part.strip() for part in name.split(' and ')]
    elif len(name.split()) == 2:
        return name.split()
    else:
        return [name]

def requires_human_review(score: int) -> bool:
    return score < 7

@app.post("/validate")
def validate_names(data: NameValidationRequest):
    results = []
    for original in data.names:
        subnames = split_names(original)
        for name in subnames:
            try:
                response = openai.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert in first name validation."},
                        {"role": "user", "content": f"Is '{name}' a valid American first name? Respond only in this JSON format: {{\"name\": \"{name}\", \"valid\": true/false, \"score\": number (0-10)}}"}
                    ]
                )
                content = response.choices[0].message.content.strip()
                parsed = eval(content)  # if needed, use json.loads(content) with proper format
                parsed["input"] = original
                parsed["human_review"] = requires_human_review(parsed["score"])
                results.append(parsed)
            except Exception as e:
                results.append({"input": original, "name": name, "error": str(e)})
    return {"results": results}

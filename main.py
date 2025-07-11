from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os

app = FastAPI()

# Use the latest OpenAI client format (>= 1.0.0)
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"

class NameValidationRequest(BaseModel):
    names: List[str]

@app.post("/validate")
def validate_names(data: NameValidationRequest):
    results = []

    for input_name in data.names:
        name_results = []

        # Split multiple names (comma, "and", space-delimited)
        cleaned = (
            input_name.replace(",", " and ")
                      .replace("&", " and ")
                      .replace("  ", " ")
                      .strip()
        )
        name_parts = [n.strip() for part in cleaned.split(" and ") for n in part.split() if n]

        for name in name_parts:
            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": "You are an expert in American first names."},
                        {"role": "user", "content": f"Is '{name}' a valid American first name? Respond only with:\nvalid: yes/no\nscore: (0-10)\nhuman_review: true/false"}
                    ]
                )

                answer = response.choices[0].message.content.strip()
                parsed = parse_validation_response(answer)

                name_results.append({
                    "name": name,
                    "valid": parsed.get("valid"),
                    "score": parsed.get("score"),
                    "human_review": parsed.get("human_review")
                })

            except Exception as e:
                name_results.append({"name": name, "error": str(e)})

        results.append({
            "input": input_name,
            "names": name_results
        })

    return {"results": results}


def parse_validation_response(text: str) -> Dict[str, Any]:
    result = {
        "valid": None,
        "score": None,
        "human_review": None
    }

    for line in text.splitlines():
        line = line.strip().lower()
        if "valid" in line:
            result["valid"] = "yes" in line
        elif "score" in line:
            try:
                result["score"] = float(line.split(":")[1].strip())
            except:
                result["score"] = None
        elif "human_review" in line:
            result["human_review"] = "true" in line

    return result

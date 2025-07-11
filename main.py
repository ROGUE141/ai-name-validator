from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os

app = FastAPI()

# Load API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

class NameValidationRequest(BaseModel):
    names: List[str]

@app.post("/validate")
async def validate_names(data: NameValidationRequest):
    results = []

    for name in data.names:
        try:
            messages = [
                {"role": "system", "content": "You are an expert in first name validation."},
                {"role": "user", "content": f"Is '{name}' a valid American first name? Respond ONLY in this format as JSON: {{\"name\": \"{name}\", \"valid\": true/false, \"score\": 1-10, \"human_review\": true/false}}"}
            ]

            response = openai.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=0
            )

            answer = response.choices[0].message.content.strip()

            # Attempt to parse the assistant's JSON response
            parsed = eval(answer)  # use json.loads(answer) if output is strict JSON
            parsed["input"] = name
            results.append(parsed)

        except Exception as e:
            results.append({"input": name, "name": name, "error": str(e)})

    return {"results": results}

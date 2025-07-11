from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os
import json

app = FastAPI()

openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

class NameValidationRequest(BaseModel):
    names: List[str]

@app.post("/validate")
def validate_names(data: NameValidationRequest):
    results = []

    for name in data.names:
        try:
            response = openai.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a strict first name validator. Only respond in raw JSON. Format:\n{\n  \"name\": \"Name\",\n  \"valid\": true or false,\n  \"score\": number (0-10),\n  \"human_review\": true or false\n}"},
                    {"role": "user", "content": f"Name: {name}"}
                ],
                response_format="json"
            )

            # Parse the JSON safely
            content = response.choices[0].message.content
            result = json.loads(content)
            result["input"] = name
            results.append(result)

        except Exception as e:
            results.append({"input": name, "name": name, "error": str(e)})

    return {"results": results}

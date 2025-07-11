from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import openai
import os

app = FastAPI()

# Load API Key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o"

class NameValidationRequest(BaseModel):
    names: List[str]

@app.post("/validate")
def validate_names(data: NameValidationRequest):
    results = []
    for name in data.names:
        try:
            response = openai.ChatCompletion.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert in first name validation."},
                    {"role": "user", "content": f"Is '{name}' a valid American first name? Respond with yes or no, a confidence score (0-10), and a short comment."}
                ]
            )
            answer = response["choices"][0]["message"]["content"]
            results.append({"input": name, "response": answer})
        except Exception as e:
            results.append({"input": name, "error": str(e)})
    return {"results": results}

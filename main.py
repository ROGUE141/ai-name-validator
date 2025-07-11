from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from typing import List
from pydantic import BaseModel
import re
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class NameRequest(BaseModel):
    names: List[str]

def extract_individual_names(input_name: str) -> List[str]:
    input_name = input_name.strip()
    splitters = [',', ' and ', '&', '/']
    for splitter in splitters:
        if splitter in input_name:
            return [n.strip() for n in input_name.split(splitter) if n.strip()]
    return [input_name]

def call_openai(name: str):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You're an expert in validating first names for U.S.-based customers."},
                {"role": "user", "content": f"Validate the name '{name}' on whether it's a common or accepted U.S. first name. Respond ONLY in this JSON format:\n{{\"name\": \"{name}\", \"valid\": true/false, \"score\": 1-10, \"human_review\": true/false}}"}
            ],
            temperature=0.2
        )
        result_text = response.choices[0].message.content.strip()
        return json.loads(result_text)
    except Exception as e:
        return {"name": name, "error": str(e)}

@app.post("/validate")
async def validate_names(request: NameRequest):
    output = []
    for raw_input in request.names:
        names_to_check = extract_individual_names(raw_input)
        name_results = [call_openai(name) for name in names_to_check]
        output.append({
            "input": raw_input,
            "names": name_results
        })
    return {"results": output}

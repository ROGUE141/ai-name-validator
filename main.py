from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any
import openai
import os
import requests

app = FastAPI()

client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
MODEL = "gpt-4o"

GOOGLE_APPS_SCRIPT_WEBHOOK_URL = "https://script.google.com/macros/s/AKfycbxhGyMtVKEzcdz0PovIwzHigpOvkL2ZMw2O9EuMvwqQx9DKnLJ7xgcMgxAwuJLLHI6x/exec"

# --- Data Models ---
class NameEntry(BaseModel):
    row: int
    name: str

class NameValidationRequest(BaseModel):
    sheetId: str
    sheetName: str
    names: List[NameEntry]

# --- Main Endpoint ---
@app.post("/validate")
def validate_names(data: NameValidationRequest):
    print(f"🔵 Received request for tab: {data.sheetName} with {len(data.names)} names.")

    results_for_sheet = []
    results_for_api = []

    for entry in data.names:
        row_number = entry.row
        input_name = entry.name
        name_results = []

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
                print(f"🟢 AI response for '{name}':", parsed)

                score = parsed.get("score")
                human_review_flag = score is None or float(score) < 6

                name_results.append({
                    "name": name,
                    "valid": parsed.get("valid"),
                    "score": score,
                    "human_review": human_review_flag
                })

            except Exception as e:
                print(f"🔴 Error validating '{name}': {e}")
                name_results.append({
                    "name": name,
                    "valid": "error",
                    "score": None,
                    "human_review": True
                })

        top_name = name_results[0] if name_results else {
            "name": "", "valid": "No", "score": 0, "human_review": True
        }

        results_for_sheet.append({
            "row": row_number + 1,  # ✅ Shift up one row to correct offset
            "input": input_name,
            "name": top_name.get("name", ""),
            "valid": "Yes" if top_name.get("valid") in [True, "yes"] else "No",
            "score": top_name.get("score") if isinstance(top_name.get("score"), (int, float)) else "",
            "human_review": top_name.get("human_review")
        })

        results_for_api.append({
            "input": input_name,
            "row": row_number,
            "names": name_results
        })

    print("📤 Posting results to Web App:", results_for_sheet)

    try:
        response = requests.post(
            GOOGLE_APPS_SCRIPT_WEBHOOK_URL,
            json={
                "sheetId": data.sheetId,
                "sheetName": data.sheetName,
                "results": results_for_sheet
            }
        )
        print("✅ POST to Google Sheets:", response.status_code, response.text)
    except Exception as e:
        print("🔴 Error posting to Google Sheets:", str(e))

    return {"results": results_for_api}

# --- Helper Function ---
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

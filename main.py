# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import os
from openai import OpenAI

# -------------------------------------------------------------------
# 1) Create FastAPI app
# -------------------------------------------------------------------
app = FastAPI()

# -------------------------------------------------------------------
# 2) Load OpenAI key & client (new SDK style)
#    • OPENAI_API_KEY is read from your Render env-var
# -------------------------------------------------------------------
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)
MODEL = "gpt-4o"        # or gpt-4o-mini

# -------------------------------------------------------------------
# 3) Pydantic request / response models
# -------------------------------------------------------------------
class NameValidationRequest(BaseModel):
    names: List[str]

# -------------------------------------------------------------------
# 4) POST endpoint
# -------------------------------------------------------------------
@app.post("/validate")
def validate_names(data: NameValidationRequest):
    results = []

    for first_name in data.names:
        try:
            # NEW SDK call
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in American first-name validation. "
                            "Answer strictly in JSON."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Is '{first_name}' a valid American first name? "
                            "Respond as JSON with keys: "
                            "name, valid (true/false), score (1-10), comment."
                        )
                    }
                ],
                temperature=0.1,
            )

            answer = response.choices[0].message.content.strip()
            results.append({"input": first_name, "response": answer})

        except Exception as e:
            # Capture any API / network error
            results.append({"input": first_name, "error": str(e)})

    return {"results": results}

# -------------------------------------------------------------------
# 5) Run locally OR under Render ("python main.py")
# -------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)

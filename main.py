# main.py
import os
import time
import hashlib
import threading
from typing import List, Dict, Any, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import openai
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel

# -------------------------
# App & Clients
# -------------------------
app = FastAPI()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY env var")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Your Apps Script Web App endpoint (can override in Render env)
GOOGLE_APPS_SCRIPT_WEBHOOK_URL = os.getenv(
    "GOOGLE_APPS_SCRIPT_WEBHOOK_URL",
    "https://script.google.com/macros/s/AKfycbxhGyMtVKEzcdz0PovIwzHigpOvkL2ZMw2O9EuMvwqQx9DKnLJ7xgcMgxAwuJLLHI6x/exec",
)

# Concurrency (tune in Render → Environment)
VALIDATOR_WORKERS = int(os.getenv("VALIDATOR_WORKERS", "20"))

# -------------------------
# Health / Warm-up endpoints
# -------------------------
@app.get("/", response_class=PlainTextResponse)
def root_health():
    # Simple warm-up / health endpoint for Render & Apps Script pings
    return "OK"

@app.get("/healthz", response_class=PlainTextResponse)
def healthz():
    return "OK"

# -------------------------
# In-memory idempotency cache (prevents double writes on retry)
# -------------------------
_IDEMPOTENCY: Dict[str, float] = {}
_ID_LOCK = threading.Lock()
_ID_TTL_SECONDS = 10 * 60  # 10 minutes

def _make_idempotency_key(sheet_id: str, sheet_name: str, rows: List[int]) -> str:
    if not rows:
        base = f"{sheet_id}:{sheet_name}:empty"
    else:
        base = f"{sheet_id}:{sheet_name}:{min(rows)}-{max(rows)}:{len(rows)}"
    return hashlib.sha1(base.encode("utf-8")).hexdigest()

def _already_processed(key: str) -> bool:
    """Return True if we've seen this key recently; else remember it."""
    now = time.time()
    with _ID_LOCK:
        # Purge old entries
        for k, ts in list(_IDEMPOTENCY.items()):
            if now - ts > _ID_TTL_SECONDS:
                _IDEMPOTENCY.pop(k, None)
        if key in _IDEMPOTENCY:
            return True
        _IDEMPOTENCY[key] = now
        return False

# -------------------------
# Request models
# -------------------------
class NameEntry(BaseModel):
    row: int
    name: str

class NameValidationRequest(BaseModel):
    sheetId: str
    sheetName: str
    names: List[NameEntry]

# -------------------------
# Helpers
# -------------------------
def parse_validation_response(text: str) -> Dict[str, Any]:
    """
    Expected lines:
      valid: yes/no
      score: 0-10
      human_review: true/false
    """
    out = {"valid": None, "score": None, "human_review": None}
    for line in (text or "").splitlines():
        low = line.strip().lower()
        if low.startswith("valid"):
            out["valid"] = "yes" in low
        elif low.startswith("score"):
            try:
                out["score"] = float(low.split(":", 1)[1].strip())
            except Exception:
                out["score"] = None
        elif low.startswith("human_review"):
            out["human_review"] = "true" in low
    return out

def pick_first_token_as_name(input_name: str) -> str:
    """
    Keep original intent: validate the first plausible first-name token.
    Splits by 'and', '&', commas, whitespace; returns the first token.
    """
    if not input_name:
        return ""
    cleaned = (
        input_name.replace(",", " and ")
        .replace("&", " and ")
        .replace("  ", " ")
        .strip()
    )
    for part in cleaned.split(" and "):
        tokens = [t for t in part.strip().split() if t]
        if tokens:
            return tokens[0]
    return ""

def validate_one_name(name_str: str) -> Dict[str, Any]:
    """
    One OpenAI call per name_str. Returns normalized result dict.
    """
    if not name_str:
        return {
            "normalized_name": "",
            "valid": False,
            "score": 0,
            "human_review": True,
        }

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in American first names."},
                {
                    "role": "user",
                    "content": (
                        f"Is '{name_str}' a valid American first name? Respond only with:\n"
                        "valid: yes/no\nscore: (0-10)\nhuman_review: true/false"
                    ),
                },
            ],
        )
        answer = (resp.choices[0].message.content or "").strip()
        parsed = parse_validation_response(answer)
        score = parsed.get("score")
        # if score missing or low, flag for human review
        human_review = score is None or float(score) < 6
        return {
            "normalized_name": name_str,
            "valid": bool(parsed.get("valid") in (True, "yes")),
            "score": score if isinstance(score, (int, float)) else 0,
            "human_review": bool(human_review),
        }
    except Exception:
        # On error, be safe: mark invalid and require review
        return {
            "normalized_name": name_str,
            "valid": False,
            "score": 0,
            "human_review": True,
        }

def validate_batch_parallel(entries: List[NameEntry], max_workers: int) -> Tuple[List[Dict[str, Any]], int]:
    """
    Validate the first plausible token for each entry in parallel.
    Returns (results_for_sheet, validate_ms)
    """
    t0 = time.time()
    candidates = [(e.row, pick_first_token_as_name(e.name), e.name) for e in entries]

    results_for_sheet: List[Dict[str, Any]] = []

    # Thread pool for parallel OpenAI calls
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(validate_one_name, cand_name): (row, raw_input)
                   for (row, cand_name, raw_input) in candidates}
        for fut in as_completed(futures):
            row, raw_input = futures[fut]
            try:
                r = fut.result()
            except Exception:
                r = {"normalized_name": "", "valid": False, "score": 0, "human_review": True}

            results_for_sheet.append(
                {
                    "row": row,
                    "input": raw_input,
                    "name": r.get("normalized_name") or "",
                    "valid": "Yes" if r.get("valid") else "No",
                    "score": r.get("score", 0),
                    "human_review": bool(r.get("human_review", False)),
                }
            )

    results_for_sheet.sort(key=lambda x: x["row"])
    validate_ms = int((time.time() - t0) * 1000)
    return results_for_sheet, validate_ms

def post_results_once(
    sheet_id: str,
    sheet_name: str,
    results_for_sheet: List[Dict[str, Any]],
    idempotency_key: str,
) -> Dict[str, Any]:
    """
    Single POST to the Apps Script Web App with light retries/backoff.
    """
    # Skip duplicate writes (process-once semantics)
    if _already_processed(idempotency_key):
        return {"status": "duplicate_skipped", "idempotency_key": idempotency_key, "http_status": 200}

    payload = {"sheetId": sheet_id, "sheetName": sheet_name, "results": results_for_sheet}
    headers = {"X-Idempotency-Key": idempotency_key}

    base_delay = 1.0
    last_exc = None
    for attempt in range(1, 4):
        try:
            t1 = time.time()
            resp = requests.post(
                GOOGLE_APPS_SCRIPT_WEBHOOK_URL,
                json=payload,
                headers=headers,
                timeout=60,
            )
            elapsed_ms = int((time.time() - t1) * 1000)
            return {
                "status": "ok",
                "http_status": resp.status_code,
                "elapsed_ms": elapsed_ms,
                "body": (resp.text or "")[:200],
                "idempotency_key": idempotency_key,
            }
        except Exception as e:
            last_exc = e
            time.sleep(base_delay)
            base_delay *= 2  # 1s -> 2s -> 4s
    return {"status": "post_failed", "error": str(last_exc), "idempotency_key": idempotency_key}

# -------------------------
# Main endpoint
# -------------------------
@app.post("/validate")
def validate_names(data: NameValidationRequest):
    """
    Pipeline:
      1) Validate N names in parallel (first token per row).
      2) Single POST to Apps Script Web App (idempotent).
      3) Return structured timings.
    """
    count = len(data.names)
    print(f"🔵 Received request for tab: {data.sheetName} with {count} names.")

    # 1) Validate in parallel
    results_for_sheet, validate_ms = validate_batch_parallel(data.names, VALIDATOR_WORKERS)

    # 2) Single POST to Web App (idempotent)
    rows = [r["row"] for r in results_for_sheet]
    idem_key = _make_idempotency_key(data.sheetId, data.sheetName, rows)
    webapp_info = post_results_once(data.sheetId, data.sheetName, results_for_sheet, idem_key)

    # 3) Build API-style detail (optional; handy for debugging)
    results_for_api = [
        {
            "row": r["row"],
            "input": r["input"],
            "names": [
                {
                    "name": r["name"],
                    "valid": r["valid"] == "Yes",
                    "score": r.get("score", 0),
                    "human_review": r.get("human_review", False),
                }
            ],
        }
        for r in results_for_sheet
    ]

    return {
        "status": "ok" if webapp_info.get("status") in ("ok", "duplicate_skipped") else "partial",
        "sheet": data.sheetName,
        "count": count,
        "timings": {
            "validate_ms": validate_ms,
            "write_ms": webapp_info.get("elapsed_ms", None),
            "total_ms": (validate_ms + (webapp_info.get("elapsed_ms") or 0)),
        },
        "webapp": webapp_info,
        "idempotency_key": idem_key,
        "results": results_for_api,
    }

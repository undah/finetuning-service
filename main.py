"""
Fine-tune data service — Railway deployment
POST /generate  →  pulls Supabase, builds JSONL, uploads to OpenAI, returns job IDs
GET  /status    →  returns run history from log
GET  /health    →  Railway health check
"""

import json
import os
import io
import logging
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(title="Finetune Data Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to your dashboard domain in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Config (set as Railway env vars) ─────────────────────────────────────────
SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]
OPENAI_API_KEY  = os.environ["OPENAI_API_KEY"]
SETUPS_TABLE    = os.getenv("SETUPS_TABLE",   "trading_setups")
MISTAKES_TABLE  = os.getenv("MISTAKES_TABLE", "trading_mistakes")
TRAIN_RATIO     = float(os.getenv("TRAIN_RATIO", "0.85"))
LOG_PATH        = "/tmp/finetune_log.json"

# ── System prompts ────────────────────────────────────────────────────────────

SETUPS_SYSTEM_PROMPT = """You are a professional trading analyst specializing in technical analysis and price action. Your job is to evaluate trade setups and determine whether they are valid or invalid based on market structure, momentum, risk/reward, and confluence of signals.

When evaluating a setup, reason step by step through the following criteria:
1. Trend direction and structure (higher highs/lows, lower highs/lows)
2. Key level confluence (support/resistance, previous highs/lows, supply/demand zones)
3. Entry trigger quality (candle pattern, breakout, pullback)
4. Risk/reward ratio (minimum 1:2 preferred)
5. Momentum confirmation (volume, dominant directional move, conviction)

Respond with:
- A verdict: VALID or INVALID
- A confidence score from 1–10
- A concise reason referencing the specific criteria above

Be strict. A setup that meets fewer than 3 of the 5 criteria should be marked INVALID."""

MISTAKES_SYSTEM_PROMPT = """You are a professional trading analyst reviewing past errors in trade setup evaluation. Your job is to understand what went wrong in a previous analysis and explain the correct interpretation of the chart.

When reviewing a mistake, clearly state:
1. What the incorrect call was
2. Why it was wrong
3. What the correct reading of the chart should have been
4. What to watch for to avoid this mistake in the future

Be direct and educational. The goal is to build better pattern recognition over time."""

# ── Helpers ───────────────────────────────────────────────────────────────────

def fetch_all_rows(client, table: str) -> list[dict]:
    rows, start, batch = [], 0, 1000
    while True:
        res = client.table(table).select("*").range(start, start + batch - 1).execute()
        rows.extend(res.data)
        if len(res.data) < batch:
            break
        start += batch
    return rows


def build_setup_example(row: dict) -> dict:
    label   = "VALID" if row.get("is_valid_setup") else "INVALID"
    notes   = (row.get("notes") or "").strip()
    session = (row.get("session") or "unknown").replace("_", " ").title()
    url     = row.get("tradingview_url", "")
    return {
        "messages": [
            {"role": "system", "content": SETUPS_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": f"Session: {session}\nEvaluate this trade setup."}
            ]},
            {"role": "assistant", "content": f"{label} — {notes}" if notes else label}
        ]
    }


def build_mistake_example(row: dict) -> dict:
    mistake = (row.get("mistake") or "").strip()
    reason  = (row.get("reason")  or "").strip()
    url     = row.get("screenshot_url", "")
    return {
        "messages": [
            {"role": "system", "content": MISTAKES_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": url}},
                {"type": "text", "text": f'Review this chart. The mistake made was: "{mistake}"\nExplain what went wrong and what the correct read should have been.'}
            ]},
            {"role": "assistant", "content": f"Mistake: {mistake}\n\nCorrection: {reason}"}
        ]
    }


def examples_to_bytes(examples: list[dict]) -> bytes:
    return "\n".join(json.dumps(ex, ensure_ascii=False) for ex in examples).encode("utf-8")


def split_train_val(examples: list[dict]) -> tuple[list, list]:
    cut = int(len(examples) * TRAIN_RATIO)
    return examples[:cut], examples[cut:]


def upload_to_openai(client: OpenAI, data: bytes, filename: str) -> str:
    """Upload bytes as a JSONL fine-tuning file and return the file ID."""
    res = client.files.create(
        file=(filename, io.BytesIO(data), "application/json"),
        purpose="fine-tune"
    )
    return res.id


def read_log() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return json.load(f)


def write_log(entry: dict):
    history = read_log()
    history.append(entry)
    with open(LOG_PATH, "w") as f:
        json.dump(history, f, indent=2)

# ── State (tracks in-progress run) ────────────────────────────────────────────

run_state: dict = {"status": "idle", "detail": None}

# ── Background job ─────────────────────────────────────────────────────────────

def run_pipeline():
    global run_state
    run_state = {"status": "running", "detail": "Connecting to Supabase..."}
    run_at    = datetime.now()
    datestamp = run_at.strftime("%Y%m%d")

    try:
        sb     = create_client(SUPABASE_URL, SUPABASE_KEY)
        openai = OpenAI(api_key=OPENAI_API_KEY)

        # ── Setups ────────────────────────────────────────────────────────────
        run_state["detail"] = "Fetching setups table..."
        rows = fetch_all_rows(sb, SETUPS_TABLE)
        rows = [r for r in rows if r.get("tradingview_url") and r.get("is_valid_setup") is not None]
        setup_examples = [build_setup_example(r) for r in rows]
        train_s, val_s = split_train_val(setup_examples)

        run_state["detail"] = f"Uploading setups to OpenAI ({len(setup_examples)} examples)..."
        n = len(setup_examples)
        s_train_id = upload_to_openai(openai, examples_to_bytes(train_s), f"setups_{datestamp}_{n}rows_train.jsonl")
        s_val_id   = upload_to_openai(openai, examples_to_bytes(val_s),   f"setups_{datestamp}_{n}rows_val.jsonl")

        # ── Mistakes ──────────────────────────────────────────────────────────
        run_state["detail"] = "Fetching mistakes table..."
        mrows = fetch_all_rows(sb, MISTAKES_TABLE)
        mrows = [r for r in mrows if r.get("screenshot_url") and r.get("mistake") and r.get("reason")]
        mistake_examples = [build_mistake_example(r) for r in mrows]
        train_m, val_m = split_train_val(mistake_examples)

        run_state["detail"] = f"Uploading mistakes to OpenAI ({len(mistake_examples)} examples)..."
        mn = len(mistake_examples)
        m_train_id = upload_to_openai(openai, examples_to_bytes(train_m), f"mistakes_{datestamp}_{mn}rows_train.jsonl")
        m_val_id   = upload_to_openai(openai, examples_to_bytes(val_m),   f"mistakes_{datestamp}_{mn}rows_val.jsonl")

        # ── Start fine-tune jobs ───────────────────────────────────────────────
        run_state["detail"] = "Starting fine-tune jobs on OpenAI..."
        setups_job = openai.fine_tuning.jobs.create(
            training_file=s_train_id,
            validation_file=s_val_id,
            model="gpt-4o-2024-08-06",
            hyperparameters={"n_epochs": 3}
        )
        mistakes_job = openai.fine_tuning.jobs.create(
            training_file=m_train_id,
            validation_file=m_val_id,
            model="gpt-4o-2024-08-06",
            hyperparameters={"n_epochs": 3}
        )

        # ── Log ───────────────────────────────────────────────────────────────
        entry = {
            "run_at": run_at.strftime("%Y-%m-%d %H:%M:%S"),
            "setups": {
                "total_rows": n, "train": len(train_s), "val": len(val_s),
                "openai_train_file_id": s_train_id,
                "openai_val_file_id":   s_val_id,
                "finetune_job_id":      setups_job.id,
                "status":               setups_job.status,
            },
            "mistakes": {
                "total_rows": mn, "train": len(train_m), "val": len(val_m),
                "openai_train_file_id": m_train_id,
                "openai_val_file_id":   m_val_id,
                "finetune_job_id":      mistakes_job.id,
                "status":               mistakes_job.status,
            }
        }
        write_log(entry)

        run_state = {"status": "done", "detail": entry}
        log.info("Pipeline complete: %s", entry)

    except Exception as e:
        run_state = {"status": "error", "detail": str(e)}
        log.exception("Pipeline failed")

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"ok": True}


@app.post("/generate")
def generate(background_tasks: BackgroundTasks):
    if run_state["status"] == "running":
        raise HTTPException(409, "A run is already in progress.")
    background_tasks.add_task(run_pipeline)
    return {"status": "started"}


@app.get("/run-status")
def run_status():
    return run_state


@app.get("/history")
def history():
    return read_log()

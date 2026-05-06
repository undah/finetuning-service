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

SETUPS_SYSTEM_PROMPT = """You are a professional trading analyst trained on the TPSS (Trade Setup Scoring System) classification framework. Your job is to evaluate trade setups by analyzing price action strictly between the WHITE LINE (start of window) and the YELLOW LINE (entry cutoff). Everything before the white line is completely irrelevant — treat it as if it does not exist.

Apply the following three-step mental checklist in order. Stop at the first failure.

STEP 1 — STATE CLARITY
Ask: does this chart tell a clear, obvious story from the white line to the yellow line?
- If you have to think hard about whether a trend exists, the state is not clear enough → BAD
- Multiple direction changes with no dominant direction → BAD immediately
- Ambiguous or uncertain bias at the yellow line → BAD, even if steps 2 and 3 would pass

STEP 2 — CONVINCING DIRECTIONAL MOVE
Must occur at any point between the white line and approximately 1 hour before the yellow line.
- Price may range or chop for the entire first part of the window — this is fine
- The move must show conviction: strong impulse candles, decisive structure break, clear follow-through
- A consistent grind where one direction clearly dominates the other is equally valid — overall story matters more than candle size
- Ranging before a breakout is fine — if the breakout is convincing, step 2 passes
- Liquidity grab = NOT valid: move barely takes a level then immediately reverses
- Conviction break = valid: closes well beyond the level with momentum continuing
- Direction does not matter — long and short bias are equally valid

STEP 3 — VALID PAUSE BEFORE YELLOW LINE
After the directional move, there must be a visible slowdown, consolidation, or pullback before the yellow line.
- Even a small pause of a few candles is sufficient
- The pause is NON-NEGOTIABLE — a perfect trend that runs straight into the yellow line with no pause = BAD
- STRUCTURAL RULE: the pullback is valid only if it does NOT break the structural low (for longs) or structural high (for shorts) that originated the move
- The structural low/high is the ORIGIN POINT of the move — the last significant low/high before the impulse started. Intermediate lows that form DURING the move are not the structural low
- If pullback holds above the structural low → structure intact → valid pause
- If pullback breaks that structural level → structure gone → BAD
- CONFIRMATION: price pulling back TO the structural level and respecting it = smart money defending the level = increases confidence significantly

VERDICTS
- GOOD: all three steps pass clearly
- BAD: any one step fails
- UNSURE: any step is borderline — move has some conviction but immediately reverses after taking a level, momentum is weak but present, pause is questionable, or state is mostly clear with one element of ambiguity. Never force GOOD or BAD when evidence is mixed.

ADDITIONAL RULES
- Price opening at the white line already moving aggressively with no subsequent pause = BAD (Rule 12)
- A sharp or aggressive-looking pullback is not automatically BAD — evaluate the structural level, not the visual appearance (Rule 4)
- When anything is ambiguous, flag it immediately. Saying UNSURE is always better than a wrong call (Rule 14)

Respond with:
- Verdict: GOOD, BAD, or UNSURE
- Confidence: 1–10
- Reasoning: walk through each of the three checklist steps explicitly, referencing what you see on the chart between the white and yellow lines"""

MISTAKES_SYSTEM_PROMPT = """You are a professional trading analyst trained on the TPSS classification framework. You are reviewing a chart where an incorrect verdict was previously given. Your job is to identify exactly which rule was violated or misapplied and explain the correct reading.

The TPSS framework evaluates price action ONLY between the WHITE LINE (start) and YELLOW LINE (entry cutoff). Everything before the white line is irrelevant.

The three-step checklist is:
1. State clarity — does the chart tell an obvious story?
2. Convincing directional move — with real conviction, not a liquidity grab
3. Valid pause — that does not break the structural origin low/high

When reviewing a mistake, structure your response as:
- INCORRECT CALL: what verdict was given and why it was wrong
- RULE VIOLATED: which specific TPSS rule or checklist step was misapplied (reference by number if possible)
- CORRECT READ: what the verdict should have been and why, walking through all three checklist steps
- WATCH FOR: the specific visual pattern or reasoning error to avoid repeating this mistake

Common mistake categories to reference where relevant:
- Counting pre-white-line moves as part of the setup (Rule 1)
- Calling a liquidity grab a conviction break (Rule 7)
- Misidentifying the structural low — using an intermediate low instead of the origin point (Rule 15)
- Calling an aggressive pullback BAD without checking the structural level (Rule 4)
- Forcing GOOD or BAD on a borderline setup instead of calling UNSURE (Rule 16)
- Missing that a grind counts as conviction — not just large candles (Rule 17)
- Ignoring ambiguous state at the yellow line (Rule 6)
- Accepting a setup with no pause before the yellow line (Rule 8)

Be direct and educational. The goal is sharper pattern recognition over time."""

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

"""
LangChain 1.x tool-calling agent for fraud investigation.

Root cause fix: response_format=FraudVerdict creates a two-phase LangGraph where
the model must explicitly decide to stop calling tools. Gemini Flash loops forever.
Solution: remove response_format, ask the model to output JSON in its final message,
parse with _parse_fallback. Works reliably with any model.

Other optimizations:
- Singleton agent (compiled once).
- Pre-filled context in prompt saves 3 tool-call round-trips.
- Hard timeout via concurrent.futures.
"""

import os
import re
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langfuse import observe
from tools import AGENT_TOOLS

load_dotenv()

AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT", "60"))

SYSTEM_PROMPT = """You are an expert financial fraud investigator at MirrorPay bank (year 2087).

## Context already provided
The transaction brief below already contains pre-computed context:
- User profile and phishing susceptibility
- Recipient IBAN history
- Phishing messages found near this transaction

You do NOT need to call lookup_user, get_iban_history, or get_phishing_proximity again
unless you want to verify a specific detail.

## When to call additional tools
- `get_user_spending_baseline` — if the amount looks anomalous vs user habits.
- `get_location_at` — if the transaction is in-person and city matters.
- `scan_for_phishing_signals` — to confirm a specific text snippet is phishing.

Make at most 2-3 additional tool calls, then stop and output your verdict.

## Contextual Skepticism (High-Volume Datasets)
You are analyzing datasets with thousands of transactions. In this environment:
- **Novel Merchants** and **High Amounts** can be legitimate (e.g., travel, electronics, business expenses).
- Do NOT flag FRAUD based on a novel merchant or high amount ALONE.
- Reaching a FRAUD verdict with a confidence > 0.8 requires a **smoking gun**:
    1. Recipient IBAN instability (account redirection).
    2. Strong phishing match (homoglyph/urgency) within the critical window.
    3. Severe location mismatch (e.g., ATM withdrawal in another country).

## Output format (REQUIRED)
When done investigating, your FINAL message must be exactly this JSON and nothing else:
{"verdict": "FRAUD", "confidence": 0.95, "reasoning": "brief explanation"}

verdict must be exactly "FRAUD" or "LEGIT".
confidence is 0.0 to 1.0.
"""


# ── Singleton ─────────────────────────────────────────────────────────────────

_agent_singleton = None


def _get_agent():
    global _agent_singleton
    if _agent_singleton is None:
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPEN_ROUTER"),
            model=os.getenv("LLM_MODEL", "google/gemini-2.0-flash-001"),
            temperature=0,
        )
        # No response_format — model outputs JSON in final message
        _agent_singleton = create_agent(
            llm,
            AGENT_TOOLS,
            system_prompt=SYSTEM_PROMPT,
        )
        print("  [agent] Compiled LangChain agent (once).")
    return _agent_singleton


# ── Prompt builder ────────────────────────────────────────────────────────────

def _build_brief(
    tx: dict,
    triage_score: str,
    triage_reasons: list,
    pre_context: dict,
) -> str:
    user = pre_context.get("user") or {}
    iban_hist = pre_context.get("iban_history", [])
    phishing = pre_context.get("phishing_events", [])

    user_block = (
        json.dumps(user, ensure_ascii=False, indent=2) if user
        else "User profile not found."
    )

    if len(iban_hist) > 1:
        iban_block = (
            f"WARNING — {len(iban_hist)} different IBANs for recipient "
            f"{tx.get('recipient_id')}:\n" + "\n".join(f"  {i}" for i in iban_hist)
        )
    elif iban_hist:
        iban_block = f"Recipient uses one IBAN consistently: {iban_hist[0]} (no instability)."
    else:
        iban_block = "No IBAN history for this recipient."

    if phishing:
        phishing_block = f"{len(phishing)} phishing message(s) found near this transaction:\n"
        for strength, diff_days, snippet in phishing:
            phishing_block += f"  [{strength}] {diff_days:.1f}d before | {snippet[:200]}\n"
    else:
        phishing_block = "No phishing messages found near the transaction date."

    return (
        f"## Transaction\n"
        f"ID:             {tx.get('transaction_id')}\n"
        f"Amount:         €{tx.get('amount')}\n"
        f"Type:           {tx.get('transaction_type')}\n"
        f"Timestamp:      {tx.get('timestamp')}\n"
        f"Sender IBAN:    {tx.get('sender_iban')}\n"
        f"Sender ID:      {tx.get('sender_id')}\n"
        f"Recipient ID:   {tx.get('recipient_id')}\n"
        f"Recipient IBAN: {tx.get('recipient_iban')}\n"
        f"Payment method: {tx.get('payment_method')}\n"
        f"Description:    {tx.get('description')}\n"
        f"Location:       {tx.get('location', 'N/A')}\n\n"
        f"## Pre-screening\n"
        f"Triage score:   {triage_score}\n"
        f"Signals:        {'; '.join(triage_reasons) if triage_reasons else 'none'}\n\n"
        f"## User profile (pre-computed)\n{user_block}\n\n"
        f"## Recipient IBAN history (pre-computed)\n{iban_block}\n\n"
        f"## Phishing proximity (pre-computed, 14-day window)\n{phishing_block}\n\n"
        'Review the context, call additional tools only if needed, then output your '
        'verdict as JSON: {"verdict": "FRAUD" or "LEGIT", "confidence": 0.0-1.0, "reasoning": "..."}'
    )


def _parse_fallback(output: str) -> tuple:
    """Extract verdict/confidence/reasoning from free-text agent output."""
    clean = re.sub(r"```(?:json)?", "", output).strip().rstrip("```")
    json_match = re.search(r'\{[^{}]*"verdict"[^{}]*\}', clean, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            verdict = str(parsed.get("verdict", "LEGIT")).upper()
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", output[:300]))
            if verdict not in ("FRAUD", "LEGIT"):
                verdict = "LEGIT"
            return verdict, reasoning, confidence
        except Exception:
            pass
    if "FRAUD" in output.upper() and "LEGIT" not in output.upper():
        return "FRAUD", output[:400], 0.55
    return "LEGIT", output[:400], 0.55


@observe(as_type="generation")
def analyze_transaction(
    tx: dict,
    triage_score: str = "RED",
    triage_reasons: list = None,
    pre_context: dict = None,
) -> tuple:
    """
    Run the fraud investigation agent.
    Returns (verdict, reasoning, confidence).
    """
    if not os.getenv("OPEN_ROUTER"):
        return "LEGIT", "No OPEN_ROUTER key configured", 0.0

    agent = _get_agent()
    brief = _build_brief(tx, triage_score, triage_reasons or [], pre_context or {})

    def _invoke():
        return agent.invoke({"messages": [HumanMessage(content=brief)]})

    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_invoke)
            result = future.result(timeout=AGENT_TIMEOUT_SECONDS)

        messages = result.get("messages", [])
        if messages:
            last = messages[-1].content
            return _parse_fallback(last if isinstance(last, str) else str(last))

        return "LEGIT", "No output from agent", 0.0

    except FuturesTimeoutError:
        tx_id = tx.get("transaction_id", "unknown")
        print(f"  TIMEOUT ({AGENT_TIMEOUT_SECONDS}s) for {tx_id} — skipping")
        return "LEGIT", f"Agent timed out after {AGENT_TIMEOUT_SECONDS}s", 0.0

    except Exception as e:
        print(f"  Agent error for {tx.get('transaction_id')}: {e}")
        return "LEGIT", f"Agent error: {e}", 0.0

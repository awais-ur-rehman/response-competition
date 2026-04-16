"""
Main fraud-detection pipeline.
Triage → [IBAN short-circuit] → Agent (with pre-filled context, limited iterations).

Speed optimizations:
1. IBAN instability with no other ambiguity → auto-flag FRAUD, skip agent entirely.
2. Pre-computed context passed to agent → saves 3 tool-call round-trips per transaction.
3. Agent singleton + recursion_limit=8 → ~4 tool calls max per transaction.
4. Hard per-transaction timeout → pipeline never hangs.
"""

import os
import json
import re
import pandas as pd
import ulid
from dotenv import load_dotenv
from langfuse import observe, propagate_attributes
from tqdm import tqdm

from data_loader import load_transactions, load_sms, load_mails, load_locations
from triage import get_triage_score, _is_strong_phishing, _is_phishing_pattern
from llm_judge import analyze_transaction
from tools import init_store

load_dotenv()


# ── Helpers ───────────────────────────────────────────────────────────────────

def generate_session_id() -> str:
    team = os.getenv("TEAM_NAME", "Default-Team").replace(" ", "-")
    return f"{team}-{ulid.new().str}"


def _build_iban_history(tx_df: pd.DataFrame) -> dict:
    history: dict = {}
    for _, tx in tx_df.iterrows():
        rid = tx.get("recipient_id")
        riban = tx.get("recipient_iban")
        if pd.notna(rid) and pd.notna(riban):
            history.setdefault(rid, set()).add(riban)
    return history


def _build_recipient_tx_count(tx_df: pd.DataFrame) -> dict:
    counts: dict = {}
    for _, tx in tx_df.iterrows():
        rid = tx.get("recipient_id")
        if pd.notna(rid) and str(rid).lower() not in ("none", "nan", ""):
            counts[rid] = counts.get(rid, 0) + 1
    return counts


def _build_sender_baselines(tx_df: pd.DataFrame, users_list: list) -> dict:
    """
    Computes mean transaction amount per sender_iban.
    Fallbacks to 5% of monthly salary if no history.
    """
    amounts: dict = {}
    for _, tx in tx_df.iterrows():
        iban = tx.get("sender_iban")
        if pd.notna(iban):
            amounts.setdefault(iban, []).append(float(tx.get("amount") or 0))
    
    baselines = {iban: sum(vals)/len(vals) for iban, vals in amounts.items()}
    
    # Fallback for users not in transactions or with few transactions
    for u in users_list:
        iban = u.get("iban")
        salary = float(u.get("salary") or 0)
        monthly_cap = (salary / 12) * 0.30 # 30% of monthly salary as a safe baseline
        if iban not in baselines or len(amounts.get(iban, [])) < 3:
            baselines[iban] = max(baselines.get(iban, 100.0), monthly_cap, 100.0)
            
    return baselines


def _precompute_phishing_events(users_list: list, sms: list, mails: list) -> dict:
    """
    Returns dict: sender_iban → list of (timestamp, strength)
    strength = "STRONG" (homoglyph domain) | "WEAK" (urgency + https://)
    """
    events_by_name: dict = {}

    for s in sms:
        content = s.get("sms", "")
        if _is_strong_phishing(content):
            strength = "STRONG"
        elif _is_phishing_pattern(content):
            strength = "WEAK"
        else:
            continue
        match = re.search(r"Date: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", content)
        if not match:
            continue
        try:
            ts = pd.to_datetime(match.group(1))
        except Exception:
            continue
        for u in users_list:
            fname = u.get("first_name", "")
            if fname and fname in content:
                events_by_name.setdefault(fname, []).append((ts, strength, content[:200]))

    for m in mails:
        content = m.get("mail", "")
        if _is_strong_phishing(content):
            strength = "STRONG"
        elif _is_phishing_pattern(content):
            strength = "WEAK"
        else:
            continue
        ts = None
        for pattern in [
            r"(\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2})",
            r"(\d{2} \w{3} \d{4} \d{2}:\d{2}:\d{2})",
        ]:
            match = re.search(pattern, content)
            if match:
                try:
                    ts = pd.to_datetime(match.group(1))
                    break
                except Exception:
                    pass
        if ts is None:
            continue
        for u in users_list:
            fname = u.get("first_name", "")
            if fname and fname in content:
                events_by_name.setdefault(fname, []).append((ts, strength, content[:300]))

    result: dict = {}
    for u in users_list:
        fname = u.get("first_name", "")
        iban = u.get("iban", "")
        if fname and iban and fname in events_by_name:
            result[iban] = events_by_name[fname]
    return result


def _get_phishing_events_for_tx(tx: dict, phishing_by_iban: dict) -> list:
    """Return (ts, strength) list for triage."""
    sender_iban = tx.get("sender_iban", "")
    return [(ts, strength) for ts, strength, _ in phishing_by_iban.get(sender_iban, [])]


def _build_pre_context(tx: dict, users_list: list, iban_history: dict,
                        phishing_by_iban: dict) -> dict:
    """
    Build the pre-computed context dict that gets embedded directly in the agent prompt,
    saving 3 tool-call round-trips per transaction.
    """
    sender_iban = tx.get("sender_iban", "")
    recipient_id = tx.get("recipient_id", "")
    tx_ts = pd.to_datetime(tx.get("timestamp"))

    # User profile
    user = next((u for u in users_list if u.get("iban") == sender_iban), None)

    # IBAN history for recipient
    iban_hist = sorted(iban_history.get(recipient_id, set()))

    # Phishing events within 14 days before transaction
    raw_events = phishing_by_iban.get(sender_iban, [])
    nearby_phishing = []
    for p_ts, strength, snippet in raw_events:
        diff_days = (tx_ts - p_ts).total_seconds() / 86400
        if 0 <= diff_days <= 14:
            nearby_phishing.append((strength, diff_days, snippet))

    return {
        "user": user,
        "iban_history": iban_hist,
        "phishing_events": nearby_phishing,
    }


def _fraud_confidence_threshold(amount: float, triage_score: str) -> float:
    """
    Returns the confidence threshold required to flag a transaction as FRAUD.
    High-value YELLOW escalations require higher confidence (0.85) to avoid FPs.
    """
    if triage_score == "YELLOW":
        return 0.85 # Very strict for weak initial signals
    
    if amount >= 500:
        return 0.45
    if amount >= 200:
        return 0.60
    if amount >= 100:
        return 0.70
    if amount >= 30:
        return 0.80
    return 0.90


# ── Pipeline ──────────────────────────────────────────────────────────────────

@observe()
def run_pipeline(dataset_dir: str, output_file: str) -> None:
    session_id = generate_session_id()
    print(f"Session ID: {session_id}")
    with open("current_session.txt", "w") as f:
        f.write(session_id)

    with propagate_attributes(session_id=session_id, user_id=os.getenv("TEAM_NAME")):
        print(f"Loading data from {dataset_dir} ...")
        tx_df = load_transactions(os.path.join(dataset_dir, "transactions.csv"))

        with open(os.path.join(dataset_dir, "users.json")) as f:
            users_list = json.load(f)

        sms = load_sms(os.path.join(dataset_dir, "sms.json"))
        mails = load_mails(os.path.join(dataset_dir, "mails.json"))
        locations = load_locations(os.path.join(dataset_dir, "locations.json"))

        iban_history = _build_iban_history(tx_df)
        recipient_tx_count = _build_recipient_tx_count(tx_df)
        sender_baselines = _build_sender_baselines(tx_df, users_list)

        print("Pre-computing phishing events ...")
        phishing_by_iban = _precompute_phishing_events(users_list, sms, mails)
        for iban, events in phishing_by_iban.items():
            strong = sum(1 for _, s, _ in events if s == "STRONG")
            weak = sum(1 for _, s, _ in events if s == "WEAK")
            print(f"  {iban}: {strong} STRONG + {weak} WEAK phishing events")

        init_store(users_list, sms, mails, locations, iban_history, tx_df)

        flagged_data = []
        skipped_green = 0
        skipped_yellow = 0
        auto_flagged = 0   # IBAN short-circuit (no agent call)
        agent_calls = 0

        print(f"\nScreening {len(tx_df)} transactions ...")
        for _, row in tqdm(tx_df.iterrows(), total=len(tx_df)):
            tx = row.to_dict()
            amount = float(tx.get("amount") or 0)

            phishing_events = _get_phishing_events_for_tx(tx, phishing_by_iban)

            sender_iban = tx.get("sender_iban", "")
            triage_score, triage_reasons = get_triage_score(
                tx, iban_history, phishing_events,
                recipient_tx_count=recipient_tx_count,
                user_baseline=sender_baselines.get(sender_iban, 50.0)
            )

            # ── Escalation Rules ─────────────────────────────────────────────
            
            should_skip = False
            if triage_score == "GREEN":
                should_skip = True
                skipped_green += 1
            elif triage_score == "YELLOW":
                # Change 3: High-value YELLOW escalates to Agent
                # Adaptive: Only escalate if amount is exceptional for THIS user
                baseline = sender_baselines.get(sender_iban, 100.0)
                hv_threshold = max(500.0, 5 * baseline)
                if amount >= hv_threshold:
                    tqdm.write(f"  ESCALATING YELLOW [€{amount:.2f} >= €{hv_threshold:.1f}]: {tx.get('transaction_id')}")
                    pass # Continue to agent
                else:
                    should_skip = True
                    skipped_yellow += 1
            
            if should_skip:
                continue

            # ── Agent with pre-filled context + iteration cap ────────────────
            agent_calls += 1
            pre_context = _build_pre_context(tx, users_list, iban_history, phishing_by_iban)

            verdict, reasoning, confidence = analyze_transaction(
                tx,
                triage_score=triage_score,
                triage_reasons=triage_reasons,
                pre_context=pre_context,
            )

            threshold = _fraud_confidence_threshold(amount, triage_score)
            if verdict == "FRAUD" and confidence >= threshold:
                tqdm.write(
                    f"  FLAGGED [{confidence:.2f} >= {threshold:.2f}]: "
                    f"{tx.get('transaction_id')} | €{amount:.2f} | {reasoning[:100]}"
                )
                flagged_data.append({
                    "transaction_id": tx["transaction_id"],
                    "amount": amount,
                    "confidence": confidence,
                    "triage": triage_score,
                    "reasoning": reasoning,
                })

        print(f"\n{'─'*60}")
        print(f"GREEN (recurring) skipped:    {skipped_green}")
        print(f"YELLOW (weak signal) skipped: {skipped_yellow}")
        print(f"Auto-flagged (IBAN, no agent):{auto_flagged}")
        print(f"Agent calls made:             {agent_calls}")
        print(f"Total flagged:                {len(flagged_data)}")
        print(f"{'─'*60}\n")

        with open(output_file, "w") as f:
            for item in flagged_data:
                f.write(f"{item['transaction_id']}\n")

        review_file = output_file.replace(".txt", "_review.csv")
        if flagged_data:
            pd.DataFrame(flagged_data).to_csv(review_file, index=False)
            print(f"Human review: {review_file}")

    print(f"Submission:  {output_file}")
    print(f"Session ID:  current_session.txt")


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "The Truman Show - train"
    output = sys.argv[2] if len(sys.argv) > 2 else "flagged_transactions.txt"
    run_pipeline(dataset, output)

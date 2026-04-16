"""
Deterministic pre-screening — no LLM, no API calls.
Runs on every transaction to bucket it GREEN / YELLOW / RED
before deciding whether to invoke the agent.
"""

import re
import pandas as pd


# ── Phishing detection — two precision levels ─────────────────────────────────

def _is_strong_phishing(text: str) -> bool:
    """
    HIGH-PRECISION: only matches homoglyph domains
    (digits substituting letters in a brand name before a suspicious action suffix).
    Examples: paypa1-secure.net, amaz0n-verify.com, ub3r-login.net

    Used for the tight 3-day RED window in triage — very few false positives.
    """
    return bool(re.search(
        r'\b(?:[a-z]*[0-9][a-z]+[a-z0-9]*|[a-z]+[0-9]+)[-.]'
        r'(?:secure|verify|login|bill|alert|support|confirm|account|update)\b',
        text.lower(),
    ))


def _is_phishing_pattern(text: str) -> bool:
    """
    BROADER: homoglyph domain OR (urgency language + a real https:// URL).
    Requires an actual https:// link — not just any .com mention — to avoid
    false positives from email addresses, company names, etc.

    Used for the wider 7-day YELLOW window and for context building.
    """
    if _is_strong_phishing(text):
        return True
    text_lower = text.lower()
    urgency = any(kw in text_lower for kw in [
        "urgent", "immediate", "action required", "suspended", "verify now",
        "unusual login", "account locked", "click here", "click the link",
        "limited time", "confirm your identity",
    ])
    # Require an actual https:// URL — not just .com anywhere in the text
    has_https_url = bool(re.search(r"https?://\S+", text_lower))
    return urgency and has_https_url


# Public alias used across the codebase
is_phishing = _is_phishing_pattern


# ── Main triage function ──────────────────────────────────────────────────────

def get_triage_score(
    tx,
    iban_history: dict,
    phishing_events: list,
    recipient_tx_count: dict = None,
    user_baseline: float = 50.0,
) -> tuple:
    """
    Fast deterministic pre-screening. No LLM calls.

    Parameters
    ----------
    tx                : dict — transaction row
    iban_history      : dict  recipient_id -> set[IBAN]
    phishing_events   : list of (timestamp, strength) tuples
    recipient_tx_count: dict  recipient_id -> int (total appearances in dataset)
    user_baseline     : float mean transaction amount for this user

    Returns
    -------
    (score, reasons) where score is 'GREEN' | 'YELLOW' | 'RED'
    """
    score = "GREEN"
    reasons = []

    # 1. Known recurring patterns → safe, skip LLM entirely
    desc = str(tx.get("description", "")).lower()
    if "salary payment" in desc or "rent payment" in desc:
        return "GREEN", ["Recognized recurring payment"]

    # 2. IBAN instability → strongest RED signal (account redirection)
    rid = tx.get("recipient_id")
    if pd.notna(rid) and rid in iban_history and len(iban_history[rid]) > 1:
        score = "RED"
        reasons.append(
            f"Recipient {rid} has {len(iban_history[rid])} different IBANs"
        )

    # 3. Off-hours transaction (00:00–05:59) → YELLOW
    is_off_hours = False
    try:
        ts = pd.to_datetime(tx.get("timestamp"))
        if ts.hour < 6:
            is_off_hours = True
            reasons.append(f"Off-hours transaction at {ts.hour:02d}:{ts.minute:02d}")
    except Exception:
        pass

    # 4. Novel merchant → YELLOW
    is_novel_merchant = False
    amount = float(tx.get("amount") or 0)
    if (
        recipient_tx_count is not None
        and pd.notna(rid)
        and str(rid).lower() not in ("none", "nan", "")
        and recipient_tx_count.get(rid, 0) == 1
        and amount >= 80
    ):
        is_novel_merchant = True
        reasons.append(f"Novel merchant {rid} (€{amount:.2f})")

    # 5. High amount signal → YELLOW
    is_high_amount = amount > 3 * user_baseline and amount > 200
    if is_high_amount:
        reasons.append(f"Anomalous amount €{amount:.2f} (>3x baseline €{user_baseline:.2f})")

    # 6. Phishing proximity
    #    Stricter rule: Solo phishing is YELLOW. 
    #    Phishing + (Off-hours OR Novel Merchant OR High Amount) → RED
    has_phishing = False
    try:
        tx_ts = pd.to_datetime(tx.get("timestamp"))
        for p_ts, strength in phishing_events:
            diff_days = (tx_ts - p_ts).total_seconds() / 86400
            if 0 <= diff_days <= 7:
                has_phishing = True
                phishing_desc = "Homoglyph" if strength == "STRONG" else "Urgency+URL"
                reasons.append(f"{phishing_desc} phishing received {diff_days:.1f}d before")
                break
    except Exception:
        pass

    # ── Escalation Logic ──────────────────────────────────────────────────────
    
    # If we already have IBAN RED, we keep it.
    
    # Phishing escalation:
    if has_phishing:
        if is_off_hours or is_novel_merchant or is_high_amount:
            score = "RED"
        else:
            if score != "RED": score = "YELLOW" # Phishing alone is just YELLOW
            
    # Default: if we have reasons but no RED, it's YELLOW
    if reasons and score == "GREEN":
        score = "YELLOW"

    return score, reasons

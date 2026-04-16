"""
Agent investigation tools.
Each function is a LangChain @tool the supervisor agent can call.
Data is injected via init_store() before any agent runs — no global I/O inside tools.

Performance: phishing SMS/mail are pre-indexed by user first name at init time
so the get_phishing_proximity tool is O(k) not O(n).
"""

import re
import json
import pandas as pd
from langchain.tools import tool

_STORE: dict = {}


# ── Phishing detection helpers ────────────────────────────────────────────────

def _is_strong_phishing(text: str) -> bool:
    """Homoglyph domains only (paypa1, amaz0n, ub3r). High precision."""
    return bool(re.search(
        r'\b(?:[a-z]*[0-9][a-z]+[a-z0-9]*|[a-z]+[0-9]+)[-.]'
        r'(?:secure|verify|login|bill|alert|support|confirm|account|update)\b',
        text.lower(),
    ))


def _is_phishing_pattern(text: str) -> bool:
    """Homoglyph OR urgency + real https:// URL. No false positives from .com mentions."""
    if _is_strong_phishing(text):
        return True
    text_lower = text.lower()
    urgency = any(kw in text_lower for kw in [
        "urgent", "immediate", "action required", "suspended", "verify now",
        "unusual login", "account locked", "click here", "click the link",
        "limited time", "confirm your identity",
    ])
    return urgency and bool(re.search(r"https?://\S+", text_lower))


def _phishing_strength(text: str) -> str:
    """Return 'STRONG', 'WEAK', or 'NONE'."""
    if _is_strong_phishing(text):
        return "STRONG"
    if _is_phishing_pattern(text):
        return "WEAK"
    return "NONE"


# ── Store init ────────────────────────────────────────────────────────────────

def init_store(
    users_list: list,
    sms_data: list,
    mail_data: list,
    location_data: list,
    iban_history: dict,
    tx_df: pd.DataFrame,
) -> None:
    """
    Populate the shared data store.
    Also builds a pre-indexed phishing lookup so tools don't re-scan raw data.
    """
    _STORE["users"] = users_list
    _STORE["sms"] = sms_data
    _STORE["mail"] = mail_data
    _STORE["locations"] = location_data
    _STORE["iban_history"] = iban_history
    _STORE["tx_df"] = tx_df

    # Pre-index phishing messages by user first name:
    # first_name → list of (timestamp, strength, snippet)
    phishing_index: dict = {}

    for s in sms_data:
        content = s.get("sms", "")
        strength = _phishing_strength(content)
        if strength == "NONE":
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
                phishing_index.setdefault(fname, []).append(
                    (ts, strength, content[:300])
                )

    for m in mail_data:
        content = m.get("mail", "")
        strength = _phishing_strength(content)
        if strength == "NONE":
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
                phishing_index.setdefault(fname, []).append(
                    (ts, strength, content[:400])
                )

    _STORE["phishing_index"] = phishing_index


# ── LangChain tools ───────────────────────────────────────────────────────────


@tool
def lookup_user(sender_iban: str) -> str:
    """
    Look up a user profile by their sender IBAN.
    Returns name, job, salary, city of residence, and a behavioural description
    that includes phishing susceptibility and spending habits.
    Call this first for any suspicious transaction.
    """
    for u in _STORE.get("users", []):
        if u.get("iban") == sender_iban:
            return json.dumps(u, ensure_ascii=False, indent=2)
    return f"No user found for IBAN {sender_iban}"


@tool
def get_iban_history(recipient_id: str) -> str:
    """
    Return all IBANs ever used by a given recipient ID across all transactions.
    A single recipient appearing with multiple different IBANs is a STRONG signal
    of account-redirection fraud or merchant compromise.
    """
    history = _STORE.get("iban_history", {}).get(recipient_id, set())
    if not history:
        return f"Recipient {recipient_id} has no IBAN history (may be a new or cash recipient)."
    ibans = sorted(history)
    if len(ibans) == 1:
        return f"Recipient {recipient_id} consistently uses one IBAN: {ibans[0]} (no instability)."
    return (
        f"WARNING — Recipient {recipient_id} has used {len(ibans)} different IBANs:\n"
        + "\n".join(f"  {iban}" for iban in ibans)
        + "\nThis is a strong indicator of account redirection or merchant compromise."
    )


@tool
def get_phishing_proximity(first_name: str, tx_timestamp: str, window_days: int = 14) -> str:
    """
    Find phishing SMS messages and emails addressed to a user (by first name)
    within window_days days BEFORE the transaction timestamp.
    Returns message summaries with time deltas and phishing strength (STRONG/WEAK).
    STRONG = homoglyph domain. WEAK = urgency language + https URL.
    A STRONG phishing within 7 days or WEAK within 3 days is a high-risk signal.
    """
    try:
        tx_ts = pd.to_datetime(tx_timestamp)
    except Exception:
        return f"Could not parse timestamp: {tx_timestamp}"

    index = _STORE.get("phishing_index", {})
    events = index.get(first_name, [])

    results = []
    for p_ts, strength, snippet in events:
        diff = (tx_ts - p_ts).total_seconds() / 86400
        if 0 <= diff <= window_days:
            results.append(
                f"[{strength}] {diff:.1f}d before tx | {snippet.replace(chr(10), ' ')}"
            )

    if not results:
        return (
            f"No phishing messages found for '{first_name}' "
            f"in the {window_days} days before {tx_timestamp}."
        )
    return (
        f"Found {len(results)} phishing message(s) targeting '{first_name}':\n"
        + "\n---\n".join(results)
    )


@tool
def get_location_at(biotag: str, timestamp: str) -> str:
    """
    Return the user's GPS location nearest to the given timestamp.
    Useful for verifying whether an in-person transaction city matches the user's
    actual location, or detecting whether the user was travelling abroad.
    """
    try:
        ts = pd.to_datetime(timestamp)
    except Exception:
        return f"Could not parse timestamp: {timestamp}"

    user_locs = [l for l in _STORE.get("locations", []) if l.get("biotag") == biotag]
    if not user_locs:
        return f"No location data found for biotag {biotag}."

    user_locs.sort(
        key=lambda x: abs((pd.to_datetime(x["timestamp"]) - ts).total_seconds())
    )
    c = user_locs[0]
    delta_h = abs((pd.to_datetime(c["timestamp"]) - ts).total_seconds()) / 3600

    return (
        f"Nearest GPS ping for {biotag}:\n"
        f"  City:     {c.get('city', 'unknown')}\n"
        f"  Lat/Lng:  {c.get('lat')}, {c.get('lng')}\n"
        f"  Recorded: {c.get('timestamp')} ({delta_h:.1f}h from transaction time)"
    )


@tool
def scan_for_phishing_signals(text: str) -> str:
    """
    Scan a block of text (SMS body, email body) for homoglyph domains and urgency language.
    A homoglyph domain replaces letters with similar-looking digits (paypa1, amaz0n, ub3r)
    to impersonate a trusted brand. Returns detected signals and an overall risk level.
    """
    homoglyph_re = re.compile(
        r"(?:https?://)?"
        r"([a-z0-9]*[0-9][a-z]+[a-z0-9]*|[a-z]+[0-9][a-z0-9]*)"
        r"[-.](?:secure|verify|login|bill|alert|support|confirm|account|update)"
        r"\.[a-z]{2,4}",
        re.IGNORECASE,
    )
    found_domains = homoglyph_re.findall(text)

    urgency_keywords = [
        "URGENT", "IMMEDIATE", "ACTION REQUIRED", "suspended", "verify now",
        "unusual login", "account locked", "click here", "limited time",
        "confirm your identity", "failure to comply",
    ]
    found_urgency = [w for w in urgency_keywords if w.lower() in text.lower()]

    if found_domains:
        risk = "HIGH"
        detail = f"Homoglyph domains detected: {found_domains}"
    elif found_urgency:
        risk = "MEDIUM"
        detail = "No homoglyph domains but urgency language present"
    else:
        risk = "LOW"
        detail = "No phishing signals detected"

    return f"Risk: {risk}\n{detail}\nUrgency keywords: {found_urgency}"


@tool
def get_user_spending_baseline(sender_iban: str) -> str:
    """
    Summarise the user's non-recurring spending patterns: amount range, typical payment
    methods, and usual transaction hours. Helps detect anomalous amounts or payment
    methods that deviate from the user's established behaviour.
    """
    tx_df = _STORE.get("tx_df")
    if tx_df is None:
        return "Transaction data not available."

    user_txs = tx_df[tx_df["sender_iban"] == sender_iban].copy()
    mask = ~user_txs["description"].str.lower().str.contains(
        "salary payment|rent payment", na=False
    )
    non_recurring = user_txs[mask]

    if non_recurring.empty:
        return f"No non-recurring transactions found for IBAN {sender_iban}."

    amounts = non_recurring["amount"].dropna().tolist()
    methods = non_recurring["payment_method"].dropna().value_counts().to_dict()
    hours = pd.to_datetime(non_recurring["timestamp"]).dt.hour.tolist()

    return (
        f"Non-recurring transactions for IBAN {sender_iban}:\n"
        f"  Count:             {len(amounts)}\n"
        f"  Amount range:      min={min(amounts):.2f}, "
        f"avg={sum(amounts)/len(amounts):.2f}, max={max(amounts):.2f}\n"
        f"  Payment methods:   {methods}\n"
        f"  Transaction hours: {sorted(hours)}"
    )


# Tools list exported to the agent
AGENT_TOOLS = [
    lookup_user,
    get_iban_history,
    get_phishing_proximity,
    get_location_at,
    scan_for_phishing_signals,
    get_user_spending_baseline,
]

# Truman Show Fraud Detection Pipeline

This repository contains the source code for the "Truman Show" fraud detection challenge.

## Architecture
The pipeline uses a multi-stage approach optimized for Precision, Economic Impact, and Agentic Performance:
1. **Deterministic Triage**: Pre-screens every transaction to filter recurring payments (GREEN) and identify suspicious patterns (YELLOW/RED) without API calls.
2. **Spending Baseline**: Calculates per-user historical spending to detect truly anomalous amounts.
3. **Escalation**: 
    - All RED transactions (phishing with companion signals or IBAN instability) go to the Agent.
    - High-value YELLOW transactions (>= €200) are escalated to the Agent.
4. **Agentic LLM Judge**: Uses a LangChain tool-calling agent (Gemini 2.0 Flash) with pre-filled context to investigate and provide a final JSON verdict.

## Key Files
- `fraud_pipeline.py`: Main entry point and orchestration.
- `triage.py`: Scoring logic (Phishing detection, IBAN monitoring).
- `llm_judge.py`: Agent implementation and LLM prompt logic.
- `context_builder.py`: Pre-computes context for the agent.
- `tools.py`: Investigation tools for the agent.
- `data_loader.py`: Dataset ingestion.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `.env` (see `.env.template`).
3. Run the pipeline:
   ```bash
   python3 fraud_pipeline.py "The Truman Show - validation" "flagged_transactions.txt"
   ```

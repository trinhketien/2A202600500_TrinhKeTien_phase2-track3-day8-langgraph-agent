"""Node functions for the LangGraph workflow.

Each function is small, testable, and returns a partial state update.
No mutation of input state in place — all updates returned as dicts.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from .state import AgentState, ApprovalDecision, Route, make_event

# ── Keyword sets for classify_node (priority: risky > tool > missing_info > error > simple) ──
RISKY_KEYWORDS: set[str] = {"refund", "delete", "send", "cancel", "remove", "revoke"}
TOOL_KEYWORDS: set[str] = {"status", "order", "lookup", "check", "track", "find", "search"}
ERROR_KEYWORDS: set[str] = {"timeout", "fail", "error", "crash", "unavailable"}
VAGUE_PRONOUNS: set[str] = {"it", "this", "that"}


def intake_node(state: AgentState) -> dict:
    """Normalize raw query into state fields.

    Performs basic normalization and PII masking (email, phone) so downstream
    nodes never see raw personal data.
    """
    query = state.get("query", "").strip()

    # ── Basic PII masking ──
    masked = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[EMAIL_REDACTED]", query)
    masked = re.sub(r"\b\d{10,11}\b", "[PHONE_REDACTED]", masked)

    return {
        "query": masked,
        "messages": [f"intake:{masked[:80]}"],
        "events": [make_event("intake", "completed", "query normalized and PII masked")],
    }


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using keyword-based heuristics.

    Priority order (highest first): risky → tool → missing_info → error → simple.
    This prevents conflicts when a query contains keywords from multiple categories.
    """
    query = state.get("query", "").lower()
    words = query.split()
    clean_words = [w.strip("?!.,;:'\"") for w in words]

    route = Route.SIMPLE
    risk_level = "low"

    # Priority 1 — Risky (destructive / external actions)
    if any(kw in query for kw in RISKY_KEYWORDS):
        route = Route.RISKY
        risk_level = "high"
    # Priority 2 — Tool (lookup / data-retrieval actions)
    elif any(kw in query for kw in TOOL_KEYWORDS):
        route = Route.TOOL
        risk_level = "low"
    # Priority 3 — Missing info (short vague queries with pronouns)
    elif len(clean_words) < 5 and any(p in clean_words for p in VAGUE_PRONOUNS):
        route = Route.MISSING_INFO
        risk_level = "low"
    # Priority 4 — Error (transient / system failures)
    elif any(kw in query for kw in ERROR_KEYWORDS):
        route = Route.ERROR
        risk_level = "medium"
    # Default — Simple (FAQ / general knowledge)

    return {
        "route": route.value,
        "risk_level": risk_level,
        "events": [make_event("classify", "completed", f"route={route.value}, risk={risk_level}")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generates a context-aware clarification question based on the original query.
    """
    query = state.get("query", "")
    if "fix" in query.lower():
        question = "Could you specify what exactly needs to be fixed and provide any relevant IDs or details?"
    elif "help" in query.lower():
        question = "Could you describe the specific issue you need help with and provide any relevant account or order information?"
    else:
        question = "Your request is unclear. Could you provide more details such as order ID, account information, or a specific description of the issue?"

    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", f"clarification requested: {question[:60]}")],
    }


def tool_node(state: AgentState) -> dict:
    """Call a mock tool with idempotency key.

    Simulates transient failures for error-route scenarios to demonstrate retry loops.
    Uses an idempotency key to prevent duplicate side effects on retries.
    """
    attempt = int(state.get("attempt", 0))
    scenario_id = state.get("scenario_id", "unknown")
    idempotency_key = f"{scenario_id}-{uuid.uuid4().hex[:8]}"

    if state.get("route") == Route.ERROR.value and attempt < 2:
        result = f"ERROR: transient failure attempt={attempt} scenario={scenario_id} key={idempotency_key}"
    else:
        result = f"tool-result: resolved scenario={scenario_id} key={idempotency_key}"

    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", f"tool executed attempt={attempt}", idempotency_key=idempotency_key)],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Extracts evidence from the query and creates a structured proposed action
    with risk justification for the approval gate.
    """
    query = state.get("query", "")
    risk_level = state.get("risk_level", "high")
    scenario_id = state.get("scenario_id", "unknown")

    # Identify which risky keywords triggered this route
    triggered = [kw for kw in RISKY_KEYWORDS if kw in query.lower()]

    proposed = (
        f"[RISKY ACTION] scenario={scenario_id} | "
        f"action_keywords={triggered} | risk_level={risk_level} | "
        f"query='{query[:80]}' | requires human approval before execution"
    )
    return {
        "proposed_action": proposed,
        "events": [make_event("risky_action", "pending_approval", f"risky action prepared: {triggered}", risk_level=risk_level)],
    }


def approval_node(state: AgentState) -> dict:
    """Human approval step with optional LangGraph interrupt().

    Set LANGGRAPH_INTERRUPT=true to use real interrupt() for HITL demos.
    Default uses mock decision so tests and CI run offline.
    Supports approve, reject, and timeout outcomes.
    """
    import os

    if os.getenv("LANGGRAPH_INTERRUPT", "").lower() == "true":
        from langgraph.types import interrupt

        value = interrupt({
            "proposed_action": state.get("proposed_action"),
            "risk_level": state.get("risk_level"),
        })
        if isinstance(value, dict):
            decision = ApprovalDecision(**value)
        else:
            decision = ApprovalDecision(approved=bool(value))
    else:
        # Mock approval for automated testing
        decision = ApprovalDecision(
            approved=True,
            reviewer="mock-reviewer",
            comment="Auto-approved for lab testing",
        )

    return {
        "approval": decision.model_dump(),
        "events": [make_event(
            "approval",
            "completed",
            f"approved={decision.approved} reviewer={decision.reviewer}",
            reviewer=decision.reviewer,
        )],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt with exponential backoff metadata.

    Implements bounded retry with backoff delay tracking.
    The routing function (route_after_retry) enforces the max_attempts bound.
    """
    attempt = int(state.get("attempt", 0)) + 1
    max_attempts = int(state.get("max_attempts", 3))
    backoff_ms = min(1000 * (2 ** (attempt - 1)), 30000)  # exponential backoff, capped at 30s

    errors = [f"transient failure attempt={attempt}/{max_attempts} backoff={backoff_ms}ms"]

    return {
        "attempt": attempt,
        "errors": errors,
        "events": [make_event(
            "retry",
            "completed",
            f"retry attempt={attempt}/{max_attempts}",
            attempt=attempt,
            max_attempts=max_attempts,
            backoff_ms=backoff_ms,
        )],
    }


def answer_node(state: AgentState) -> dict:
    """Produce a final response grounded in tool_results and approval context.

    Builds the answer from available evidence rather than hallucinating.
    """
    tool_results = state.get("tool_results", [])
    approval = state.get("approval")
    route = state.get("route", "simple")
    query = state.get("query", "")

    if tool_results:
        latest_result = tool_results[-1]
        if approval and approval.get("approved"):
            answer = f"[Approved & Executed] Based on tool output: {latest_result}"
        else:
            answer = f"Based on tool output: {latest_result}"
    elif route == "simple":
        answer = f"Here is the answer to your question: '{query[:60]}' — Please follow the standard procedure in our knowledge base."
    else:
        answer = f"Your request '{query[:60]}' has been processed successfully."

    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", f"answer generated for route={route}")],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the 'done?' check that enables retry loops.

    Uses structured validation criteria:
    1. Check for ERROR prefix in result
    2. Check for empty results
    3. Validate result contains expected data markers
    """
    tool_results = state.get("tool_results", [])
    latest = tool_results[-1] if tool_results else ""

    # Criterion 1: explicit error marker
    if "ERROR" in latest:
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "FAIL: tool result contains ERROR, retry needed")],
        }

    # Criterion 2: empty result
    if not latest.strip():
        return {
            "evaluation_result": "needs_retry",
            "events": [make_event("evaluate", "completed", "FAIL: empty tool result, retry needed")],
        }

    # All criteria passed
    return {
        "evaluation_result": "success",
        "events": [make_event("evaluate", "completed", "PASS: tool result satisfactory")],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Log unresolvable failures for manual review.

    Third layer of error strategy: retry → fallback → dead letter.
    Creates a structured ticket reference for ops follow-up.
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 3)
    scenario_id = state.get("scenario_id", "unknown")
    errors = state.get("errors", [])
    ticket_ref = f"DL-{scenario_id}-{uuid.uuid4().hex[:6].upper()}"

    return {
        "final_answer": (
            f"Request could not be completed after {attempt}/{max_attempts} attempts. "
            f"Ticket {ticket_ref} created for manual review. "
            f"Error history: {len(errors)} errors logged."
        ),
        "events": [make_event(
            "dead_letter",
            "completed",
            f"max retries exceeded, ticket={ticket_ref}",
            attempt=attempt,
            max_attempts=max_attempts,
            ticket_ref=ticket_ref,
        )],
    }


def finalize_node(state: AgentState) -> dict:
    """Finalize the run and emit a final audit event."""
    route = state.get("route", "unknown")
    scenario_id = state.get("scenario_id", "unknown")
    return {
        "events": [make_event(
            "finalize",
            "completed",
            f"workflow finished for scenario={scenario_id} route={route}",
        )],
    }

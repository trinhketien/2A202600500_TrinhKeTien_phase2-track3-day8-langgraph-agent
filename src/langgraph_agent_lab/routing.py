"""Routing functions for conditional edges.

Each function receives the current state and returns a string matching a node name.
These are used with `graph.add_conditional_edges()` to implement dynamic routing.
"""

from __future__ import annotations

import logging

from .state import AgentState, Route

logger = logging.getLogger(__name__)


def route_after_classify(state: AgentState) -> str:
    """Map classified route to the next graph node.

    Handles unknown routes safely by logging a warning and defaulting to 'answer'.
    """
    route = state.get("route", Route.SIMPLE.value)
    mapping = {
        Route.SIMPLE.value: "answer",
        Route.TOOL.value: "tool",
        Route.MISSING_INFO.value: "clarify",
        Route.RISKY.value: "risky_action",
        Route.ERROR.value: "retry",
    }
    target = mapping.get(route)
    if target is None:
        logger.warning("Unknown route '%s' in state, defaulting to 'answer'", route)
        return "answer"
    return target


def route_after_retry(state: AgentState) -> str:
    """Decide whether to retry, fallback, or dead-letter.

    Bounded retry: if attempt >= max_attempts, route to dead_letter.
    Otherwise, route back to tool for another attempt.
    """
    attempt = int(state.get("attempt", 0))
    max_attempts = int(state.get("max_attempts", 3))
    if attempt >= max_attempts:
        logger.info("Max retries reached (%d/%d), routing to dead_letter", attempt, max_attempts)
        return "dead_letter"
    return "tool"


def route_after_evaluate(state: AgentState) -> str:
    """Decide whether tool result is satisfactory or needs retry.

    This is the 'done?' check that enables retry loops — a key LangGraph advantage over LCEL.
    """
    if state.get("evaluation_result") == "needs_retry":
        return "retry"
    return "answer"


def route_after_approval(state: AgentState) -> str:
    """Continue only if approved. Rejected actions route to clarify.

    Supports three outcomes:
    - approved=True → proceed to tool execution
    - approved=False → route to clarify for re-evaluation
    - approval missing → route to clarify as safety default
    """
    approval = state.get("approval") or {}
    if approval.get("approved"):
        return "tool"
    logger.info("Approval rejected or missing, routing to clarify")
    return "clarify"

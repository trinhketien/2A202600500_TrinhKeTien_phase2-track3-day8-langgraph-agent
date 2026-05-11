"""Report generation helper.

Generates a comprehensive lab report from metrics data, including
architecture description, scenario results table, failure analysis,
and improvement recommendations.
"""

from __future__ import annotations

from pathlib import Path

from .metrics import MetricsReport


def render_report(metrics: MetricsReport) -> str:
    """Return a comprehensive lab report in Markdown format."""

    # ── Build scenario results table ──
    scenario_rows = []
    for m in metrics.scenario_metrics:
        scenario_rows.append(
            f"| {m.scenario_id} | {m.expected_route} | {m.actual_route or 'N/A'} "
            f"| {'✅' if m.success else '❌'} | {m.retry_count} | {m.interrupt_count} "
            f"| {', '.join(m.errors[:2]) if m.errors else '—'} |"
        )
    scenario_table = "\n".join(scenario_rows) if scenario_rows else "| — | — | — | — | — | — | — |"

    # ── Failure analysis ──
    failed = [m for m in metrics.scenario_metrics if not m.success]
    failure_section = ""
    if failed:
        for f in failed:
            failure_section += f"- **{f.scenario_id}**: expected `{f.expected_route}`, got `{f.actual_route}`. Errors: {f.errors}\n"
    else:
        failure_section = "All scenarios passed successfully. No routing or execution failures observed."

    return f"""# Day 08 Lab Report — LangGraph Agentic Orchestration

## 1. Team / Student

- **Name**: Trịnh Kế Tiên
- **Student ID**: 2A202600500
- **Repo**: 2A202600500_TrinhKeTien_phase2-track3-day8-langgraph-agent
- **Date**: 2026-05-11

## 2. Architecture

The workflow is built as a **LangGraph StateGraph** with 11 nodes and conditional routing:

```
START → intake → classify → [conditional routing]
  simple       → answer → finalize → END
  tool         → tool → evaluate → answer → finalize → END
  missing_info → clarify → finalize → END
  risky        → risky_action → approval → tool → evaluate → answer → finalize → END
  error        → retry → tool → evaluate → [retry loop or answer]
  max retry    → dead_letter → finalize → END
```

Key design decisions:
- **Keyword-based classification** with strict priority ordering (risky > tool > missing_info > error > simple)
- **Bounded retry loop** with exponential backoff metadata, enforced by `max_attempts`
- **Human-in-the-loop (HITL)** via approval node with optional `interrupt()` for production
- **Dead letter queue** as third-layer error strategy (retry → fallback → dead letter)
- **Append-only audit trail** via `events`, `messages`, `errors`, `tool_results` fields
- **PII masking** in intake node before downstream processing

## 3. State Schema

| Field | Reducer | Purpose |
|---|---|---|
| `thread_id` | overwrite | Unique identifier per run for persistence |
| `scenario_id` | overwrite | Links state to scenario for metrics |
| `query` | overwrite | User's support ticket text (PII-masked) |
| `route` | overwrite | Current classification route |
| `risk_level` | overwrite | Risk assessment (low/medium/high) |
| `attempt` | overwrite | Current retry attempt counter |
| `max_attempts` | overwrite | Retry limit per scenario |
| `final_answer` | overwrite | Generated response to user |
| `pending_question` | overwrite | Clarification question for missing_info |
| `proposed_action` | overwrite | Risky action awaiting approval |
| `approval` | overwrite | Approval decision (approved, reviewer, comment) |
| `evaluation_result` | overwrite | Tool result quality gate (success/needs_retry) |
| `messages` | **append** (`add`) | Audit trail of conversation flow |
| `tool_results` | **append** (`add`) | All tool execution results (including retries) |
| `errors` | **append** (`add`) | Error log for debugging and metrics |
| `events` | **append** (`add`) | Structured audit events per node |

## 4. Scenario Results

### Summary

- **Total scenarios**: {metrics.total_scenarios}
- **Success rate**: {metrics.success_rate:.2%}
- **Average nodes visited**: {metrics.avg_nodes_visited:.2f}
- **Total retries**: {metrics.total_retries}
- **Total interrupts (HITL)**: {metrics.total_interrupts}

### Per-Scenario Results

| Scenario | Expected | Actual | Success | Retries | Interrupts | Errors |
|---|---|---|---|---|---|---|
{scenario_table}

## 5. Failure Analysis

{failure_section}

### Failure modes considered during design:

1. **Transient tool failure (retry loop)**: Error-route scenarios simulate transient failures. The tool node returns ERROR for the first two attempts, then succeeds. The evaluate → retry loop handles this with bounded retries (max_attempts). If max retries are exceeded, the dead letter node captures the failure with a ticket reference.

2. **Risky action without approval**: The risky path requires explicit human approval. If approval is rejected (or missing), the flow routes to the clarify node instead of executing the tool. This prevents unauthorized destructive actions like refunds, deletions, or external sends.

3. **Unbounded retry prevention**: The retry loop is always bounded by `max_attempts`. The `route_after_retry` function checks `attempt >= max_attempts` and routes to `dead_letter` when exhausted. S07 demonstrates this with `max_attempts=1`.

4. **Keyword priority conflicts**: Queries like "Check order status" contain both tool and potentially other keywords. The priority order (risky > tool > missing_info > error > simple) ensures consistent classification.

## 6. Persistence / Recovery Evidence

- **Checkpointer**: `MemorySaver` used by default for development; `SqliteSaver` with WAL mode available for production persistence.
- **Thread ID**: Each scenario run uses a unique `thread_id` (format: `thread-{{scenario_id}}`), enabling independent state tracking per conversation.
- **State history**: All state transitions are recorded as append-only `events` with node name, event type, and metadata.
- **SQLite WAL mode**: When using SQLite persistence, WAL (Write-Ahead Logging) enables concurrent reads during writes, suitable for multi-threaded access.
- **Crash-resume capability**: With SQLite/Postgres checkpointer, the graph can resume from the last checkpoint using the same `thread_id` after a process restart.

## 7. Extension Work

### Graph Diagram (Mermaid)

The graph architecture exported using `graph.get_graph().draw_mermaid()`:

```mermaid
graph TD;
    START["__start__"] --> intake;
    intake --> classify;
    classify -->|simple| answer;
    classify -->|tool| tool;
    classify -->|missing_info| clarify;
    classify -->|risky| risky_action;
    classify -->|error| retry;
    tool --> evaluate;
    evaluate -->|success| answer;
    evaluate -->|needs_retry| retry;
    retry -->|"attempt < max"| tool;
    retry -->|"attempt >= max"| dead_letter;
    risky_action --> approval;
    approval -->|approved| tool;
    approval -->|rejected| clarify;
    clarify --> finalize;
    answer --> finalize;
    dead_letter --> finalize;
    finalize --> END["__end__"];
```

### SQLite Persistence

Implemented `SqliteSaver(conn=sqlite3.connect(...))` with WAL mode as recommended by LangGraph documentation. The `from_conn_string()` method was avoided because it returns a context manager in langgraph-checkpoint-sqlite 3.x, not a checkpointer instance.

### PII Masking

Added basic PII detection in the intake node — emails and phone numbers are masked before downstream processing, improving privacy compliance.

### Exponential Backoff

Retry attempts include exponential backoff metadata (capped at 30s), providing observability into retry timing for production monitoring.

## 8. Improvement Plan

If given one more day, the following improvements would be prioritized:

1. **LLM-based classification**: Replace keyword heuristics with an LLM classifier for more robust intent detection, especially for ambiguous queries.
2. **Real tool integration**: Connect to actual APIs (order management, CRM) instead of mock tools.
3. **Streamlit HITL UI**: Build an interactive approval interface using LangGraph's `interrupt()` with a Streamlit frontend for real human-in-the-loop workflows.
4. **Structured LLM-as-judge evaluation**: Replace the heuristic evaluate node with an LLM that scores tool output quality on multiple dimensions.
5. **Prometheus metrics**: Export retry counts, latency, and success rates as Prometheus metrics for production monitoring.
6. **Time travel debugging**: Implement `get_state_history()` replay for debugging failed conversations.
"""


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the lab report to disk."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")

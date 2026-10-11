"""
CRRA — Lab C4: Orchestration, HITL & Audit
==========================================

A LangGraph StateGraph that reviews contracts end to end:

    analysis ──► policy_check ──┬──► hitl ──► report ──► END
                                └──────────► report ──► END   (no trigger fired)

  analysis_node      fetch contract + Anthropic tool-calling loop (search_policy, submit_recommendation)
  policy_check_node  pure Python rules engine — no model call
  hitl_node          blocking terminal approval, captures approver name
  report_node        terminal status: <REC>_AUTO | <REC>_APPROVED | ON_HOLD_REJECTED

Every node writes to the append-only audit trail at logs/audit_trail.jsonl.

Run (contract API running in another terminal):
    python orchestrator/supervisor.py                      # default contracts
    python orchestrator/supervisor.py CTR-1003 CTR-1010     # choose your own
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make the project root importable so `guardrails` resolves when run as
# `python orchestrator/supervisor.py` from the project folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
from typing import Optional, TypedDict

import anthropic
import chromadb
import requests
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from guardrails.audit_logger import AuditLogger

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(PROJECT_ROOT.parent / ".env")

MODEL = "claude-opus-5"
EFFORT = "medium"            # temperature is not supported on this model
MAX_TOKENS = 4096
MAX_ROUNDS = 5

API_BASE = os.getenv("CONTRACT_API_URL", "http://localhost:5001").rstrip("/")
HTTP_TIMEOUT = 10

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "data" / "chroma_db"))
KB_DIR = Path(os.getenv("KB_DIR", PROJECT_ROOT / "data" / "kb"))
COLLECTION_NAME = "crra_policy"

DEFAULT_CONTRACTS = ["CTR-1001", "CTR-1003", "CTR-1005", "CTR-1007", "CTR-1010"]

VALID_RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
VALID_CONFIDENCE = ["HIGH", "MEDIUM", "LOW"]

client = anthropic.Anthropic()
audit = AuditLogger()

# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class ContractState(TypedDict, total=False):
    contract_id: str
    contract: dict
    recommendation: str
    confidence: str
    rationale: str
    policy_citation: str
    estimated_annual_impact_inr: float
    hitl_required: bool
    hitl_reason: str
    hitl_approved: Optional[bool]
    approver: Optional[str]
    final_status: str


# ─────────────────────────────────────────────────────────────────────────────
# Contract field helpers — tolerate small naming differences in the C2 API
# ─────────────────────────────────────────────────────────────────────────────
FIELD_ALIASES = {
    "annual_value_inr": ["annual_value_inr", "annual_value", "value_inr", "contract_value_inr", "annual_cost_inr"],
    "approval_band": ["approval_band", "band"],
    "notice_state": ["notice_state", "notice_status"],
    "owner": ["owner", "contract_owner", "owner_name", "business_owner"],
    "utilisation_pct": ["utilisation_pct", "utilization_pct", "utilisation"],
    "vendor": ["vendor", "vendor_name", "supplier"],
    "category": ["category", "spend_category"],
}


def field(contract: dict, name: str, default=None):
    for key in FIELD_ALIASES.get(name, [name]):
        if key in contract and contract[key] not in (None, ""):
            return contract[key]
    return default


def fmt_inr(value) -> str:
    try:
        return f"₹{float(value):,.0f}"
    except (TypeError, ValueError):
        return "n/a"


# ─────────────────────────────────────────────────────────────────────────────
# Data access
# ─────────────────────────────────────────────────────────────────────────────
def fetch_contract(contract_id: str) -> dict:
    url = f"{API_BASE}/api/contracts/{contract_id}"
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {"error": "CONTRACT_API_UNREACHABLE",
                "message": f"Cannot reach {API_BASE}. Start mcp_server/contract_shim.py in another terminal."}
    except requests.exceptions.RequestException as exc:
        return {"error": "CONTRACT_API_ERROR", "message": str(exc)}
    if resp.status_code == 404:
        return {"error": "NOT_FOUND", "message": f"No contract '{contract_id}'."}
    if not resp.ok:
        return {"error": f"HTTP_{resp.status_code}", "message": resp.text[:300]}
    try:
        data = resp.json()
    except ValueError:
        return {"error": "BAD_RESPONSE", "message": "Contract API returned non-JSON."}
    # Some shims wrap the record: {"contract": {...}}
    return data.get("contract", data) if isinstance(data, dict) else {"error": "BAD_RESPONSE"}


_collection = None


def _split_sections(text: str, source: str) -> list[tuple[str, str]]:
    sections, heading, buf = [], None, []
    for line in text.splitlines():
        m = re.match(r"^#{2,3}\s+(.*)", line)
        if m:
            if heading and "".join(buf).strip():
                sections.append((heading, "\n".join(buf).strip()))
            heading, buf = m.group(1).strip(), []
        else:
            buf.append(line)
    if heading and "".join(buf).strip():
        sections.append((heading, "\n".join(buf).strip()))
    if not sections and text.strip():
        sections.append((Path(source).stem, text.strip()))
    return sections


def get_collection():
    global _collection
    if _collection is not None:
        return _collection
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        _collection = chroma_client.get_collection(COLLECTION_NAME)
    except Exception:
        col = chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
        ids, docs, metas = [], [], []
        for md in sorted(KB_DIR.glob("*.md")):
            for i, (heading, body) in enumerate(_split_sections(md.read_text(encoding="utf-8"), md.name)):
                ids.append(f"{md.stem}-{i}")
                docs.append(f"{heading}\n\n{body}")
                metas.append({"source": md.name, "section": heading})
        if not docs:
            raise RuntimeError(f"No policy .md files in {KB_DIR}. Set KB_DIR or run Lab C1 first.")
        col.add(ids=ids, documents=docs, metadatas=metas)
        print(f"  (built policy KB: {len(docs)} chunks)")
        _collection = col
    return _collection


def _confidence(distance: float, space: str) -> float:
    if space == "cosine":
        return round(max(0.0, min(1.0, 1.0 - distance)), 3)
    if space == "ip":
        return round(max(0.0, min(1.0, -distance)), 3)
    return round(1.0 / (1.0 + distance), 3)


def search_policy(query: str) -> dict:
    """Group hits by source file; return the best section from each of the top 2 files."""
    try:
        col = get_collection()
        space = (col.metadata or {}).get("hnsw:space", "l2")
        res = col.query(query_texts=[query], n_results=min(10, col.count()))
    except Exception as exc:
        return {"error": "POLICY_SEARCH_FAILED", "message": str(exc), "results": []}

    best: dict[str, dict] = {}
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        meta = meta or {}
        source = meta.get("source") or meta.get("file") or "unknown"
        section = meta.get("section") or meta.get("heading") or doc.split("\n", 1)[0][:80]
        if source not in best or dist < best[source]["_d"]:
            best[source] = {"source": source, "section": section,
                            "confidence": _confidence(dist, space), "text": doc, "_d": dist}
    ranked = sorted(best.values(), key=lambda r: r["_d"])[:2]
    for r in ranked:
        r.pop("_d")
    return {"query": query, "results": ranked}


# ─────────────────────────────────────────────────────────────────────────────
# Agent definition
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Renewal Analysis Agent in the Contract Renewal Risk Advisor (CRRA).
The contract record is provided in the user message. Its derived fields (notice_deadline,
notice_state, utilisation_pct, approval_band) are pre-computed — use them as given and do not
redo date arithmetic or threshold comparisons.

1. Call search_policy with focused questions about the rules that apply to this contract.
2. Finish by calling submit_recommendation exactly once. Never answer in plain text instead.

- recommendation is exactly one of RENEW, RENEGOTIATE, CONSOLIDATE, TERMINATE.
- policy_citation must name the source file AND section heading returned by search_policy.
  Never invent a citation.
- rationale must state the specific figures that drove the decision.
- utilisation_pct may be null for AMC or support contracts: that means not applicable, not zero.
- estimated_annual_impact_inr: annual saving (+) or added cost (-) versus renewing unchanged.

Confidence: HIGH when a retrieved policy section directly covers the case and the data is
complete; MEDIUM when judgement or an assumption was needed; LOW when data is missing or the
policy does not clearly address the case. LOW confidence is a valid, honest answer — not a
failure. Do not stretch weak evidence into a confident call; a human reviews LOW results.
"""

TOOLS = [
    {
        "name": "search_policy",
        "description": ("Semantic search over the procurement policy KB. Returns the best section from each "
                        "of the top 2 policy files with source, section, confidence (0-1) and text."),
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation. Call exactly once, as the last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {"type": "string", "enum": VALID_RECOMMENDATIONS},
                "confidence": {"type": "string", "enum": VALID_CONFIDENCE},
                "rationale": {"type": "string"},
                "policy_citation": {"type": "string"},
                "estimated_annual_impact_inr": {"type": "number"},
            },
            "required": ["contract_id", "recommendation", "confidence", "rationale",
                         "policy_citation", "estimated_annual_impact_inr"],
        },
    },
]


def extract_text(response) -> str:
    """First block with a .text attribute — a thinking block may come first."""
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            return text
    return ""


def run_agent(contract_id: str, contract: dict) -> tuple[dict | None, int]:
    messages = [{
        "role": "user",
        "content": (f"Analyse contract {contract_id} and submit a renewal recommendation.\n\n"
                    f"Contract record:\n{json.dumps(contract, indent=2, default=str)}"),
    }]

    for round_no in range(1, MAX_ROUNDS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            #output_config={"effort": EFFORT},
        )
        text = extract_text(response)
        if text:
            print(f"    [round {round_no}] {text.strip()[:240]}")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            messages.append({"role": "user",
                             "content": "Please call submit_recommendation now (LOW confidence is acceptable)."})
            continue

        results = []
        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            print(f"    [round {round_no}] → {block.name}({json.dumps(block.input)[:100]})")
            if block.name == "submit_recommendation":
                rec = dict(block.input)
                if rec.get("recommendation") in VALID_RECOMMENDATIONS and rec.get("confidence") in VALID_CONFIDENCE:
                    return rec, round_no
                result = {"status": "rejected", "problem": "invalid recommendation or confidence value"}
            elif block.name == "search_policy":
                result = search_policy(**block.input)
            else:
                result = {"error": f"Unknown tool {block.name}"}
            results.append({"type": "tool_result", "tool_use_id": block.id,
                            "content": json.dumps(result, default=str)})
        messages.append({"role": "user", "content": results})

    return None, MAX_ROUNDS


# ─────────────────────────────────────────────────────────────────────────────
# Node 1 — analysis
# ─────────────────────────────────────────────────────────────────────────────
def analysis_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    print("\n▶ ANALYSIS AGENT")

    contract = fetch_contract(cid)
    if "error" in contract:
        print(f"  ✗ {contract['message']}")
        audit.log("analysis", cid, "contract_fetch_failed", error=contract["error"], message=contract["message"])
        return {**state, "contract": contract, "recommendation": "UNDETERMINED", "confidence": "LOW",
                "rationale": f"Contract data unavailable: {contract['message']}",
                "policy_citation": "N/A", "estimated_annual_impact_inr": 0}

    print(f"  {field(contract, 'vendor', '?')} · {field(contract, 'category', '?')} · "
          f"{fmt_inr(field(contract, 'annual_value_inr'))}/yr · util {field(contract, 'utilisation_pct', 'n/a')}% · "
          f"band {field(contract, 'approval_band', '?')} · {field(contract, 'notice_state', '?')}")
    audit.log("analysis", cid, "contract_fetched",
              approval_band=field(contract, "approval_band"), notice_state=field(contract, "notice_state"))

    try:
        rec, rounds = run_agent(cid, contract)
    except anthropic.APIError as exc:
        rec, rounds = None, 0
        audit.log("analysis", cid, "model_error", error=str(exc))

    if rec is None:
        audit.log("analysis", cid, "no_submission", rounds=rounds, max_rounds=MAX_ROUNDS)
        return {**state, "contract": contract, "recommendation": "UNDETERMINED", "confidence": "LOW",
                "rationale": f"Agent did not submit a recommendation within {MAX_ROUNDS} rounds.",
                "policy_citation": "N/A", "estimated_annual_impact_inr": 0}

    print(f"  ✓ {rec['recommendation']} ({rec['confidence']}) — {rec['policy_citation']}")
    audit.log("analysis", cid, "recommendation_submitted", actor=MODEL, rounds=rounds,
              recommendation=rec["recommendation"], confidence=rec["confidence"],
              policy_citation=rec["policy_citation"],
              estimated_annual_impact_inr=rec["estimated_annual_impact_inr"],
              rationale=rec["rationale"])
    return {**state, "contract": contract,
            "recommendation": rec["recommendation"], "confidence": rec["confidence"],
            "rationale": rec["rationale"], "policy_citation": rec["policy_citation"],
            "estimated_annual_impact_inr": rec["estimated_annual_impact_inr"]}


# ─────────────────────────────────────────────────────────────────────────────
# Node 2 — policy check (deterministic, no model call)
# ─────────────────────────────────────────────────────────────────────────────
def policy_check_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    contract = state.get("contract") or {}
    print("\n▶ POLICY CHECK")

    reasons = []
    band = str(field(contract, "approval_band", "") or "").strip().upper()
    if band in ("B", "C"):
        reasons.append(f"Approval band {band} requires a named approver")

    if str(field(contract, "notice_state", "") or "").strip().upper() == "INSIDE_WINDOW":
        reasons.append("Notice deadline is inside the window (notice_state=INSIDE_WINDOW)")

    if state.get("recommendation") == "TERMINATE":
        reasons.append("Recommendation is TERMINATE")

    if state.get("confidence") == "LOW":
        reasons.append("Agent confidence is LOW")

    owner = str(field(contract, "owner", "") or "").strip().upper()
    if owner == "UNASSIGNED":
        reasons.append("Contract owner is UNASSIGNED")

    if "error" in contract:
        reasons.append(f"Contract data unavailable ({contract.get('error')})")

    hitl_required = bool(reasons)
    hitl_reason = "; ".join(reasons)

    print(f"  {'⚠ HITL required: ' + hitl_reason if hitl_required else '✓ No policy trigger — auto-complete'}")
    audit.log("policy_check", cid, "triggers_evaluated", hitl_required=hitl_required,
              triggers=reasons, trigger_count=len(reasons))
    return {**state, "hitl_required": hitl_required, "hitl_reason": hitl_reason}


# ─────────────────────────────────────────────────────────────────────────────
# Node 3 — human in the loop
# ─────────────────────────────────────────────────────────────────────────────
def hitl_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    contract = state.get("contract") or {}

    print("\n▶ HUMAN APPROVAL REQUIRED")
    print("  " + "─" * 60)
    print(f"  Contract       : {cid} · {field(contract, 'vendor', '?')}")
    print(f"  Recommendation : {state.get('recommendation')}  (confidence {state.get('confidence')})")
    print(f"  Annual impact  : {fmt_inr(state.get('estimated_annual_impact_inr'))}")
    print(f"  Citation       : {state.get('policy_citation')}")
    print(f"  Rationale      : {state.get('rationale')}")
    print("  Why you're seeing this:")
    for reason in (state.get("hitl_reason") or "").split("; "):
        print(f"    • {reason}")
    print("  " + "─" * 60)

    audit.log("hitl", cid, "approval_requested", reasons=state.get("hitl_reason"))

    try:
        answer = ""
        while answer not in ("y", "n"):
            answer = input("  Approve this recommendation? [y/n] ").strip().lower()
        approved = answer == "y"
        approver = None
        if approved:
            while not approver:
                approver = input("  Approver name: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n  (no interactive input — treating as rejected)")
        approved, approver = False, None

    audit.log("hitl", cid, "approved" if approved else "rejected",
              actor=approver or "approver", recommendation=state.get("recommendation"))
    return {**state, "hitl_approved": approved, "approver": approver}


# ─────────────────────────────────────────────────────────────────────────────
# Node 4 — report
# ─────────────────────────────────────────────────────────────────────────────
def report_node(state: ContractState) -> ContractState:
    cid = state["contract_id"]
    rec = state.get("recommendation", "UNDETERMINED")

    if not state.get("hitl_required"):
        final_status = f"{rec}_AUTO"
    elif state.get("hitl_approved"):
        final_status = f"{rec}_APPROVED"
    else:
        final_status = "ON_HOLD_REJECTED"

    print(f"\n▶ REPORT\n  Final status: {final_status}"
          + (f" (approved by {state.get('approver')})" if state.get("approver") else ""))
    audit.log("report", cid, "final_status", actor=state.get("approver") or "system",
              final_status=final_status, recommendation=rec, confidence=state.get("confidence"),
              hitl_required=state.get("hitl_required"), hitl_approved=state.get("hitl_approved"))
    return {**state, "final_status": final_status}


# ─────────────────────────────────────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────────────────────────────────────
def route_after_policy(state: ContractState) -> str:
    return "hitl" if state.get("hitl_required") else "report"


def build_graph():
    g = StateGraph(ContractState)
    g.add_node("analysis", analysis_node)
    g.add_node("policy_check", policy_check_node)
    g.add_node("hitl", hitl_node)
    g.add_node("report", report_node)

    g.set_entry_point("analysis")
    g.add_edge("analysis", "policy_check")
    g.add_conditional_edges("policy_check", route_after_policy, {"hitl": "hitl", "report": "report"})
    g.add_edge("hitl", "report")
    g.add_edge("report", END)
    return g.compile()


def print_summary(outcomes: list[ContractState]) -> None:
    headers = ["Contract", "Recommendation", "Conf.", "Impact/yr", "HITL", "Approver", "Final status"]
    rows = [[o.get("contract_id", "?"), o.get("recommendation", "?"), o.get("confidence", "?"),
             fmt_inr(o.get("estimated_annual_impact_inr")), "YES" if o.get("hitl_required") else "no",
             o.get("approver") or "—", o.get("final_status", "?")] for o in outcomes]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    rule = "─" * (sum(widths) + 3 * (len(widths) - 1))
    print("\n" + "═" * len(rule) + "\nREVIEW SUMMARY\n" + "═" * len(rule))
    print(" │ ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(rule)
    for r in rows:
        print(" │ ".join(c.ljust(w) for c, w in zip(r, widths)))
    print(rule)
    print(f"Audit trail: {audit.log_path}  (run_id {audit.run_id})")


def main() -> None:
    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to a .env file or your environment.")

    contract_ids = sys.argv[1:] or DEFAULT_CONTRACTS
    get_collection()
    graph = build_graph()
    audit.log("supervisor", "*", "run_started", contracts=contract_ids, 
    model=MODEL)

    outcomes = []
    for cid in contract_ids:
        print("\n" + "═" * 66 + f"\nCONTRACT: {cid}\n" + "═" * 66)
        state: ContractState = {"contract_id": cid, "hitl_required": False,
                                "hitl_approved": None, "approver": None}
        outcomes.append(graph.invoke(state))

    audit.log("supervisor", "*", "run_completed",
              statuses={o["contract_id"]: o.get("final_status") for o in outcomes})
    print_summary(outcomes)


if __name__ == "__main__":
    main()
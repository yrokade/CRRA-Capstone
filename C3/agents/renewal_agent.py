"""
CRRA — Lab C3: Renewal Analysis Agent
=====================================

A standalone tool-calling agent that analyses one vendor contract at a time and
records a policy-grounded recommendation through a schema-constrained tool.

Tools (the agent's only route to data):
  1. get_contract           GET  http://localhost:5001/api/contracts/<id>
  2. search_policy          ChromaDB collection 'crra_policy' (grouped by source file)
  3. find_category_overlap  GET  http://localhost:5001/api/categories
  4. submit_recommendation  enum-constrained final answer

Run (with the contract API running in another terminal):
    python mcp_server/contract_shim.py           # terminal 1
    python agents/renewal_agent.py                # terminal 2 — five default contracts
    python agents/renewal_agent.py CTR-1003 CTR-1007   # or choose your own

Environment:
    ANTHROPIC_API_KEY   required (a .env file in C3/ or the project root also works)
    CONTRACT_API_URL    optional, default http://localhost:5001
    CHROMA_DIR          optional, default <C3>/data/chroma_db
    KB_DIR              optional, default <C3>/data/kb
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import anthropic
import chromadb
import requests
from dotenv import load_dotenv

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
C3_DIR = Path(__file__).resolve().parents[1]          # .../C3
load_dotenv(C3_DIR / ".env")
load_dotenv(C3_DIR.parent / ".env")

MODEL = "claude-opus-5"
EFFORT = "medium"            # temperature is not supported on this model
MAX_TOKENS = 4096
MAX_ROUNDS = 5               # hard cap on the agentic loop

API_BASE = os.getenv("CONTRACT_API_URL", "http://localhost:5001").rstrip("/")
HTTP_TIMEOUT = 10

CHROMA_DIR = Path(os.getenv("CHROMA_DIR", C3_DIR / "data" / "chroma_db"))
KB_DIR = Path(os.getenv("KB_DIR", C3_DIR / "data" / "kb"))
COLLECTION_NAME = "crra_policy"

DEFAULT_TEST_CONTRACTS = ["CTR-1001", "CTR-1002", "CTR-1003", "CTR-1004", "CTR-1005"]

VALID_RECOMMENDATIONS = ["RENEW", "RENEGOTIATE", "CONSOLIDATE", "TERMINATE"]
VALID_CONFIDENCE = ["HIGH", "MEDIUM", "LOW"]

client = anthropic.Anthropic()

# ─────────────────────────────────────────────────────────────────────────────
# System prompt
# ─────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the Renewal Analysis Agent in the Contract Renewal Risk Advisor (CRRA).
You analyse ONE vendor contract and recommend exactly one action:
RENEW, RENEGOTIATE, CONSOLIDATE or TERMINATE.

How to work:
1. Call get_contract to read the contract. Its derived fields (notice_deadline, notice_state,
   utilisation_pct, approval_band) are already computed — use them as given and do not redo
   date arithmetic or threshold comparisons yourself.
2. Call search_policy with a focused question about the rule that applies (for example price
   uplift limits, low utilisation, notice windows, approval thresholds, vendor consolidation).
   You may search more than once if the first result does not address the situation.
3. If consolidation is plausible, call find_category_overlap with the contract's category to see
   competing vendors and their utilisation.
4. Finish by calling submit_recommendation exactly once. Never answer in plain text instead.

Rules for the recommendation:
- policy_citation must name the source file AND section heading returned by search_policy,
  e.g. "pricing_policy.md § Price uplift limits". Never invent a citation.
- rationale must state the specific figures that drove the decision (values, uplift %,
  utilisation %, days to notice deadline, overlapping vendors).
- utilisation_pct may be null (AMC or support contracts have no seats). Null means
  "not applicable", not zero — do not treat it as low usage.
- estimated_annual_impact_inr is the expected annual saving (positive) or added cost (negative)
  versus renewing unchanged. Use 0 for RENEW with no price change. Base it on the data provided;
  do not fabricate figures you cannot derive.
- Set human_approval_required to true when the policy you retrieved reserves the decision for a
  named approver (e.g. high approval band, termination or consolidation of a significant contract,
  a notice deadline already inside the window), or whenever you are unsure. When in doubt, true.

Confidence:
- HIGH: the data is complete and a retrieved policy section directly covers the situation.
- MEDIUM: the policy applies but some judgement or an assumption was needed.
- LOW: the data is incomplete or contradictory, the policy does not clearly address the case,
  or the tools failed. LOW confidence is a valid, honest answer — not a failure. Do not stretch
  weak evidence into a confident call; submit LOW and explain what is missing so a human can decide.
"""

# ─────────────────────────────────────────────────────────────────────────────
# Tool schemas
# ─────────────────────────────────────────────────────────────────────────────
TOOLS = [
    {
        "name": "get_contract",
        "description": (
            "Fetch one contract from the contract API, including derived risk fields "
            "(notice_deadline, notice_state, utilisation_pct, approval_band)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string", "description": "Contract ID, e.g. CTR-1003"}
            },
            "required": ["contract_id"],
        },
    },
    {
        "name": "search_policy",
        "description": (
            "Semantic search over the procurement policy knowledge base. Returns the best "
            "matching section from each of the top 2 policy files, with source, section "
            "heading, confidence (0-1) and text. Low confidence means the policy may not "
            "address the question."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A focused natural-language policy question."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "find_category_overlap",
        "description": (
            "Look up a spend category to see every vendor contract in it and their utilisation, "
            "to judge whether another vendor could absorb this contract's capability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "Category name exactly as shown on the contract."}
            },
            "required": ["category"],
        },
    },
    {
        "name": "submit_recommendation",
        "description": "Record the final recommendation for the contract. Call exactly once, as the last step.",
        "input_schema": {
            "type": "object",
            "properties": {
                "contract_id": {"type": "string"},
                "recommendation": {"type": "string", "enum": VALID_RECOMMENDATIONS},
                "confidence": {"type": "string", "enum": VALID_CONFIDENCE},
                "rationale": {
                    "type": "string",
                    "description": "Plain-language reasoning citing the specific figures that drove the call.",
                },
                "policy_citation": {
                    "type": "string",
                    "description": "Source file and section heading from search_policy.",
                },
                "estimated_annual_impact_inr": {
                    "type": "number",
                    "description": "Annual saving (+) or added cost (-) in INR versus renewing unchanged.",
                },
                "human_approval_required": {
                    "type": "boolean",
                    "description": "True if policy reserves this decision for a named human approver.",
                },
            },
            "required": [
                "contract_id",
                "recommendation",
                "confidence",
                "rationale",
                "policy_citation",
                "estimated_annual_impact_inr",
                "human_approval_required",
            ],
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Tool 1 — get_contract
# ─────────────────────────────────────────────────────────────────────────────
def get_contract(contract_id: str) -> dict:
    url = f"{API_BASE}/api/contracts/{contract_id}"
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
    except requests.exceptions.ConnectionError:
        return {
            "error": "CONTRACT_API_UNREACHABLE",
            "message": (
                f"Could not connect to the contract API at {API_BASE}. Start it with "
                "'python mcp_server/contract_shim.py' in another terminal. Without contract "
                "data, any recommendation must be LOW confidence."
            ),
        }
    except requests.exceptions.Timeout:
        return {"error": "CONTRACT_API_TIMEOUT", "message": f"The contract API at {url} timed out."}
    except requests.exceptions.RequestException as exc:
        return {"error": "CONTRACT_API_ERROR", "message": str(exc)}

    if resp.status_code == 404:
        return {"error": "NOT_FOUND", "message": f"No contract with id '{contract_id}'."}
    if not resp.ok:
        return {"error": f"HTTP_{resp.status_code}", "message": resp.text[:500]}
    try:
        return resp.json()
    except ValueError:
        return {"error": "BAD_RESPONSE", "message": "Contract API returned non-JSON content."}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 2 — search_policy (ChromaDB)
# ─────────────────────────────────────────────────────────────────────────────
_collection = None


def _split_sections(text: str, source: str) -> list[tuple[str, str]]:
    """Split a markdown policy file into (heading, section_text) at ## / ### headings."""
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


def _build_collection(chroma_client) -> "chromadb.Collection":
    """Fallback ingest if Lab C1's kb_setup.py hasn't been run against CHROMA_DIR."""
    print(f"  KB collection '{COLLECTION_NAME}' not found — building it now from {KB_DIR} ...")
    col = chroma_client.get_or_create_collection(COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    ids, docs, metas = [], [], []
    for md in sorted(KB_DIR.glob("*.md")):
        for i, (heading, body) in enumerate(_split_sections(md.read_text(encoding="utf-8"), md.name)):
            ids.append(f"{md.stem}-{i}")
            docs.append(f"{heading}\n\n{body}")
            metas.append({"source": md.name, "section": heading})
    if not docs:
        raise RuntimeError(f"No .md policy files found in {KB_DIR}. Set KB_DIR or run Lab C1 first.")
    col.add(ids=ids, documents=docs, metadatas=metas)
    print(f"  Built KB with {len(docs)} chunks.")
    return col


def _get_collection():
    global _collection
    if _collection is None:
        chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            _collection = chroma_client.get_collection(COLLECTION_NAME)
        except Exception:
            _collection = _build_collection(chroma_client)
    return _collection


def _distance_to_confidence(distance: float, space: str) -> float:
    if space == "cosine":
        return round(max(0.0, min(1.0, 1.0 - distance)), 3)
    if space == "ip":
        return round(max(0.0, min(1.0, -distance)), 3)
    return round(1.0 / (1.0 + distance), 3)          # l2 (Chroma default)


def search_policy(query: str, n_results: int = 10, top_files: int = 2) -> dict:
    try:
        col = _get_collection()
        space = (col.metadata or {}).get("hnsw:space", "l2")
        res = col.query(query_texts=[query], n_results=min(n_results, col.count()))
    except Exception as exc:
        return {"error": "POLICY_SEARCH_FAILED", "message": str(exc), "results": []}

    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]

    # Group by source file, keeping only the best (closest) section per file.
    best_per_file: dict[str, dict] = {}
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        source = meta.get("source") or meta.get("file") or meta.get("filename") or "unknown"
        section = meta.get("section") or meta.get("heading") or meta.get("title") or doc.split("\n", 1)[0][:80]
        if source not in best_per_file or dist < best_per_file[source]["_dist"]:
            best_per_file[source] = {
                "source": source,
                "section": section,
                "confidence": _distance_to_confidence(dist, space),
                "text": doc,
                "_dist": dist,
            }

    ranked = sorted(best_per_file.values(), key=lambda r: r["_dist"])[:top_files]
    for r in ranked:
        r.pop("_dist")
    return {"query": query, "results": ranked}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 3 — find_category_overlap
# ─────────────────────────────────────────────────────────────────────────────
def find_category_overlap(category: str) -> dict:
    url = f"{API_BASE}/api/categories"
    try:
        resp = requests.get(url, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.ConnectionError:
        return {
            "error": "CONTRACT_API_UNREACHABLE",
            "message": f"Could not connect to {API_BASE}. Start 'python mcp_server/contract_shim.py'.",
        }
    except (requests.exceptions.RequestException, ValueError) as exc:
        return {"error": "CATEGORY_LOOKUP_FAILED", "message": str(exc)}

    wanted = category.strip().lower()

    # Accept {"categories": [...]}, a plain list, or a dict keyed by category name.
    if isinstance(data, dict) and "categories" in data:
        data = data["categories"]

    if isinstance(data, dict):
        for name, entry in data.items():
            if name.strip().lower() == wanted:
                return {"category": name, **entry} if isinstance(entry, dict) else {"category": name, "contracts": entry}
        available = list(data.keys())
    else:
        for entry in data:
            name = str(entry.get("category") or entry.get("name") or "")
            if name.strip().lower() == wanted:
                return entry
        available = [str(e.get("category") or e.get("name")) for e in data]

    return {"error": "CATEGORY_NOT_FOUND", "message": f"No category '{category}'.", "available": available}


# ─────────────────────────────────────────────────────────────────────────────
# Tool 4 — submit_recommendation
# ─────────────────────────────────────────────────────────────────────────────
def submit_recommendation(**rec) -> dict:
    """Validate the final answer. The enum in the schema already constrains the model;
    this is a belt-and-braces check before the result leaves the agent."""
    problems = []
    if rec.get("recommendation") not in VALID_RECOMMENDATIONS:
        problems.append(f"recommendation must be one of {VALID_RECOMMENDATIONS}")
    if rec.get("confidence") not in VALID_CONFIDENCE:
        problems.append(f"confidence must be one of {VALID_CONFIDENCE}")
    if not isinstance(rec.get("human_approval_required"), bool):
        problems.append("human_approval_required must be a boolean")
    if problems:
        return {"status": "rejected", "problems": problems}
    return {"status": "recorded", "recommendation": rec}


TOOL_DISPATCH = {
    "get_contract": get_contract,
    "search_policy": search_policy,
    "find_category_overlap": find_category_overlap,
    "submit_recommendation": submit_recommendation,
}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def extract_text(response) -> str:
    """Return the text of the first block that has a .text attribute.
    A thinking block may come first, so content[0].text is not safe."""
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            return text
    return ""


def low_confidence_fallback(contract_id: str, reason: str) -> dict:
    return {
        "contract_id": contract_id,
        "recommendation": "RENEGOTIATE",
        "confidence": "LOW",
        "rationale": f"Agent did not reach a recommendation: {reason}. Human review required.",
        "policy_citation": "N/A — no recommendation submitted",
        "estimated_annual_impact_inr": 0,
        "human_approval_required": True,
        "_fallback": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# The agent loop
# ─────────────────────────────────────────────────────────────────────────────
def analyse_contract(contract_id: str, verbose: bool = True) -> dict:
    messages = [
        {
            "role": "user",
            "content": (
                f"Analyse contract {contract_id} and submit a renewal recommendation "
                "using submit_recommendation."
            ),
        }
    ]

    rounds = 0
    while rounds < MAX_ROUNDS:
        rounds += 1
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
            #output_config={"effort": EFFORT},
        )

        text = extract_text(response)
        if verbose and text:
            print(f"  [round {rounds}] {text.strip()[:300]}")

        if response.stop_reason != "tool_use":
            # The model answered in prose instead of submitting. Nudge it once per round.
            messages.append({"role": "assistant", "content": response.content})
            messages.append(
                {
                    "role": "user",
                    "content": "Please record your answer by calling submit_recommendation "
                    "(LOW confidence is acceptable if the evidence is weak).",
                }
            )
            continue

        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        submitted = None

        for block in response.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            fn = TOOL_DISPATCH.get(block.name)
            if verbose:
                print(f"  [round {rounds}] → {block.name}({json.dumps(block.input)[:120]})")
            try:
                result = fn(**block.input) if fn else {"error": f"Unknown tool {block.name}"}
            except Exception as exc:  # never let a tool crash the loop
                result = {"error": "TOOL_EXCEPTION", "message": str(exc)}

            if block.name == "submit_recommendation" and result.get("status") == "recorded":
                submitted = result["recommendation"]

            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                    **({"is_error": True} if "error" in result or result.get("status") == "rejected" else {}),
                }
            )

        if submitted:
            submitted["_rounds"] = rounds
            return submitted

        messages.append({"role": "user", "content": tool_results})

    return low_confidence_fallback(contract_id, f"MAX_ROUNDS ({MAX_ROUNDS}) reached without submission")


# ─────────────────────────────────────────────────────────────────────────────
# Test harness
# ─────────────────────────────────────────────────────────────────────────────
def _inr(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    sign = "-" if v < 0 else ""
    return f"{sign}₹{abs(v):,.0f}"


def print_summary(results: list[dict]) -> None:
    headers = ["Contract", "Recommendation", "Conf.", "Annual impact", "Approval", "Rounds", "Citation"]
    rows = [
        [
            r.get("contract_id", "?"),
            r.get("recommendation", "?") + (" *" if r.get("_fallback") else ""),
            r.get("confidence", "?"),
            _inr(r.get("estimated_annual_impact_inr")),
            "YES" if r.get("human_approval_required") else "no",
            str(r.get("_rounds", MAX_ROUNDS if r.get("_fallback") else "?")),
            (r.get("policy_citation") or "")[:45],
        ]
        for r in results
    ]
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = "─" * (sum(widths) + 3 * (len(widths) - 1))

    print("\n" + "═" * len(line))
    print("SUMMARY")
    print("═" * len(line))
    print(" │ ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for row in rows:
        print(" │ ".join(c.ljust(w) for c, w in zip(row, widths)))
    print(line)
    if any(r.get("_fallback") for r in results):
        print("* fallback: agent did not submit within MAX_ROUNDS — treated as LOW confidence.")
    needing = sum(1 for r in results if r.get("human_approval_required"))
    print(f"{len(results)} contracts analysed · {needing} require human approval")


def main() -> None:
    contract_ids = sys.argv[1:] or DEFAULT_TEST_CONTRACTS

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to a .env file or your environment.")

    _get_collection()  # build/open the KB once up front

    results = []
    for cid in contract_ids:
        print("\n" + "═" * 62)
        print(f"ANALYSING: {cid}")
        print("═" * 62)
        try:
            rec = analyse_contract(cid)
        except anthropic.APIError as exc:
            print(f"  API error: {exc}")
            rec = low_confidence_fallback(cid, f"Anthropic API error: {exc.__class__.__name__}")
        results.append(rec)
        print(f"  ✓ {rec['recommendation']} ({rec['confidence']}) — {rec['rationale'][:200]}")

    print_summary(results)


if __name__ == "__main__":
    main()
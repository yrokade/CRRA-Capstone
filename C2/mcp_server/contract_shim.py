"""contract_shim.py - Flask REST shim over data/contracts.csv on port 5001.
Derived fields use a simulated date of 2025-04-01. PATCH edits are in-memory only.
Run from the project root:  python .\\mcp_server\\contract_shim.py
"""
import csv
from collections import defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from flask import Flask, jsonify, request

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "contracts.csv"
SIM_DATE = date(2025, 4, 1)
APPROACHING_DAYS = 30
DATE_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d")  # ISO first, then DD-MM-YYYY

# Column names in contracts.csv - change these if your headers differ.
ID_COL, VALUE_COL, CATEGORY_COL, VENDOR_COL = "contract_id", "annual_value", "category", "vendor"
PATCHABLE = {"status", "owner", "proposed_uplift_pct"}

app = Flask(__name__)

def _num(value, cast=float, default=0):
    try:
        return cast(str(value).replace(",", "").strip())
    except (TypeError, ValueError):
        return default

def parse_date(text: str) -> date:
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date '{text}' - expected one of {DATE_FORMATS}")

def enrich(row: dict) -> dict:
    renewal = parse_date(row["renewal_date"])
    deadline = renewal - timedelta(days=_num(row.get("notice_days"), int))
    days_to_deadline = (deadline - SIM_DATE).days
    state = ("EXPIRED" if renewal < SIM_DATE
             else "INSIDE_WINDOW" if days_to_deadline < 0
             else "APPROACHING" if days_to_deadline <= APPROACHING_DAYS
             else "OPEN")
    purchased = _num(row.get("seats_purchased"), int)
    active = _num(row.get("seats_active"), int)
    value = _num(row.get(VALUE_COL))
    row.update({
        "renewal_date": renewal.isoformat(),  # normalised to YYYY-MM-DD
        VALUE_COL: value,
        "notice_deadline": deadline.isoformat(),
        "notice_state": state,
        "utilisation_pct": round(active / purchased * 100, 1) if purchased else None,
        "approval_band": "A" if value < 1_000_000 else "B" if value <= 5_000_000 else "C",
    })
    return row

def load_contracts() -> dict:
    with CSV_PATH.open(newline="", encoding="utf-8-sig") as f:
        return {str(r[ID_COL]).strip(): enrich(dict(r)) for r in csv.DictReader(f)}

CONTRACTS = load_contracts()  # loaded once at startup

def not_found(contract_id):
    return jsonify(error="not_found", message=f"Contract '{contract_id}' not found"), 404

@app.get("/health")
def health():
    return jsonify(status="ok", contracts=len(CONTRACTS), simulated_date=SIM_DATE.isoformat())

@app.get("/api/contracts")
def list_contracts():
    category = request.args.get("category", "").strip().lower()
    band = request.args.get("band", "").strip().upper()
    state = request.args.get("notice_state", "").strip().upper()
    result = [c for c in CONTRACTS.values()
              if (not category or str(c.get(CATEGORY_COL, "")).lower() == category)
              and (not band or c["approval_band"] == band) and (not state or c["notice_state"] == state)]
    return jsonify(count=len(result), contracts=result)

@app.get("/api/contracts/expiring")
def expiring():
    try:
        days = int(request.args.get("days", 90))
    except ValueError:
        return jsonify(error="bad_request", message="'days' must be an integer"), 400
    rows = [{**c, "days_to_renewal": (parse_date(c["renewal_date"]) - SIM_DATE).days}
            for c in CONTRACTS.values()]
    result = sorted((c for c in rows if 0 <= c["days_to_renewal"] <= days),
                    key=lambda c: c["days_to_renewal"])
    return jsonify(days=days, count=len(result), contracts=result)

@app.get("/api/contracts/<contract_id>")
def get_contract(contract_id):
    contract = CONTRACTS.get(contract_id)
    return jsonify(contract) if contract else not_found(contract_id)

def group_by(field, label, keys=()):
    groups = defaultdict(lambda: {"vendors": set(), "ids": [], "total": 0.0})
    for k in keys:
        groups[k]  # pre-seed so empty groups (e.g. an unused band) still appear
    for c in CONTRACTS.values():
        g = groups[c.get(field) or "Uncategorised"]
        g["vendors"].add(c.get(VENDOR_COL, ""))
        g["ids"].append(c[ID_COL])
        g["total"] += c[VALUE_COL]
    return [{label: k, "vendors": sorted(g["vendors"]), "vendor_count": len(g["vendors"]),
             "contract_ids": g["ids"], "contract_count": len(g["ids"]),
             "total_value": round(g["total"], 2)} for k, g in sorted(groups.items())]

@app.get("/api/categories")
def categories():
    result = group_by(CATEGORY_COL, "category")
    return jsonify(count=len(result), categories=result)

@app.get("/api/contract/band")
def band_summary():
    result = group_by("approval_band", "band", keys=("A", "B", "C"))
    return jsonify(portfolio_total=round(sum(b["total_value"] for b in result), 2),
                   contract_count=len(CONTRACTS), bands=result)

@app.get("/api/category/vendors")
def category_vendor_ranking():
    result = sorted(group_by(CATEGORY_COL, "category"),
                    key=lambda g: (-g["vendor_count"], -g["total_value"]))
    top_n = result[0]["vendor_count"] if result else 0
    return jsonify(top_categories=[g for g in result if g["vendor_count"] == top_n],
                   ranking=result)

@app.patch("/api/contracts/<contract_id>")
def update_contract(contract_id):
    if not (contract := CONTRACTS.get(contract_id)):
        return not_found(contract_id)
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or not body:
        return jsonify(error="bad_request", message="JSON body required"), 400
    if invalid := set(body) - PATCHABLE:
        return jsonify(error="bad_request", message=f"Not updatable: {sorted(invalid)}",
                       allowed=sorted(PATCHABLE)), 400
    if "proposed_uplift_pct" in body:
        try:
            body["proposed_uplift_pct"] = float(body["proposed_uplift_pct"])
        except (TypeError, ValueError):
            return jsonify(error="bad_request", message="proposed_uplift_pct must be a number"), 400
    contract.update(body)
    return jsonify(updated=sorted(body), contract=contract)

if __name__ == "__main__":
    print(f"Loaded {len(CONTRACTS)} contracts from {CSV_PATH}")
    app.run(host="127.0.0.1", port=5001, debug=False)
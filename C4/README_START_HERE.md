# CRRA — Contract Renewal Risk Advisor

A four-lab capstone. You will build an AI system that reviews Zensar's vendor contract
portfolio and recommends what to renew, renegotiate, consolidate or terminate — with a
policy citation for every recommendation and a human approval gate before anything is
committed.

Budget roughly **9 hours** across the four labs.

---

## Set up once, before Lab C1

Create one project folder and work inside it for all four labs. Do not create a
separate folder per lab — later labs reuse earlier labs' files.

```
mkdir C:\CRRA-Training
cd C:\CRRA-Training
python -m venv labenv
labenv\Scripts\activate
```

Unzip Lab C1 into this folder, then install:

```
pip install -r requirements.txt
```

For Labs C3 and C4 you will also need an API key:

```
copy .env.template .env
```

Open `.env` and paste your key. **Never commit `.env` to git** — the included
`.gitignore` already excludes it, but check with `git status` before you push.

---

## The labs, in order

| Lab | You build | Time |
|-----|-----------|------|
| **C1** | A searchable policy knowledge base in ChromaDB | 90 min |
| **C2** | A Flask mock contract API with derived policy fields | 90 min |
| **C3** | A tool-calling agent that recommends and cites policy | 3 hr |
| **C4** | LangGraph orchestration, human approval gate, audit trail | 3 hr |

Each zip contains a Word document with step-by-step instructions and a
**Vibe-Coding Prompt** you can paste into Claude Desktop to generate the code
yourself instead of reading the supplied version. Both routes are valid — try
vibe-coding at least one lab.

---

## Terminal discipline

By Lab C3 you need two things running at once. Use **separate terminal tabs** in
VS Code rather than stopping and restarting:

- Tab 1: `python mcp_server/contract_shim.py` — leave running
- Tab 2: your agent or orchestrator

Both tabs need the virtual environment activated.

---

## If something breaks

**`ModuleNotFoundError: No module named 'guardrails'`** (Lab C4)
Python only puts the script's own folder on the import path, not the project root.
The supplied file already fixes this with a `sys.path.insert` at the top — check it
is still there and that you are running from the project root.

**`Contract API unreachable`** (Labs C3, C4)
The Flask shim is not running, or it is running in a terminal where you closed the
window. Check `http://localhost:5001/health` in a browser.

**ChromaDB fails on first run**
It downloads a small embedding model the first time. This needs a stable network
connection. Retry once before changing any code.

**The agent loops and never finishes**
Check `MAX_ROUNDS` is still in place. Without it, an agent that cannot reach a
confident answer will keep calling tools indefinitely.

**PowerShell mangles `curl`**
Bare `curl` in PowerShell is an alias for `Invoke-WebRequest` and does not accept
the flags you expect. Use a browser for simple GET requests, or `curl.exe`
explicitly.

---

## What to submit

Push your work to a personal GitHub repository and share the link. Include all four
labs' code, plus a short note (half a page is plenty) covering:

- One recommendation your agent got wrong, and why you think it did
- One thing you changed from the supplied code, and what happened
- The extra approval trigger you added in Lab C4, Step 6

The wrong recommendation is the most interesting part. An honest account of a
failure is worth more here than a clean run with nothing learned.

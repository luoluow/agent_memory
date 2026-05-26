# Eval result 2026-05-26 for auto memory of claude-code Sonnet 4.6

## auto-memory.py implementation

### What our `auto_memory.py` does

**Ingest** (`ingest_session`):
- Skips filler sessions outright
- For evidence sessions: calls `claude -p` with a custom `INGEST_SYSTEM_PROMPT` we wrote, asking Claude to output a JSON structure `{"files": [...], "delete": [...]}` describing what to write/delete
- Parses that JSON and writes `.md` files to a temp directory
- Rebuilds `MEMORY.md` index after each write

**Retrieve** (`retrieve`):
- Reads all `.md` files from the temp directory directly
- Strips YAML frontmatter, concatenates bodies
- No LLM call — just file I/O

**Reset**:
- Creates a fresh temp directory per episode so episodes don't contaminate each other

---

### Where it genuinely matches real Claude Code

| Aspect         | Real Claude Code                                 | Our implementation                   |
| -------------- | ------------------------------------------------ | ------------------------------------ |
| File format    | YAML frontmatter + typed `.md` files             | Same format                          |
| Memory types   | user / feedback / project / reference            | Same types                           |
| Index file     | `MEMORY.md` with one bullet per file             | Same structure                       |
| Retrieval      | Files loaded at session start, no extra LLM call | Same — direct file read              |
| Decision maker | Claude decides what's worth saving               | Claude decides (via our ingest call) |

---

### Where it diverges from real Claude Code

**1. The ingest prompt is our approximation, not the real one.**
Real Claude Code's auto-memory system prompt is internal and proprietary. We wrote `INGEST_SYSTEM_PROMPT` based on the publicly documented behavior. The real prompt likely has more nuance around when to merge vs create files, how to handle contradictions, etc.

**2. Ingest mechanism is different.**
Real Claude Code fires a **Stop hook** automatically after every session ends — Claude writes to files naturally as a side-effect. We call `claude -p` explicitly and ask for a **JSON diff** (`{"files": [...], "delete": [...]}`), then apply it programmatically. This is an awkward workaround because we can't actually hook into Claude's file-writing tools — we can only call the CLI and parse its stdout.

**3. We skip filler sessions; real Claude Code doesn't.**
The real Stop hook runs after every session unconditionally. We skip ~80% of sessions as an optimization.

**4. Episode isolation is artificial.**
Real Claude Code memory persists indefinitely across all sessions. We reset to a fresh temp dir per episode to prevent cross-contamination between unrelated MeME episodes.

---

### Bottom line

Our implementation captures the **format and philosophy** of Claude Code's auto-memory — typed markdown files, LLM-decides-what-to-save, direct-file retrieval. But the ingest call is a proxy (JSON via subprocess, not a real hook), and the prompt driving the decisions is our approximation of the real one. The eval results measure this proxy, not the exact real system.

## Auto-memory vs In-context baseline (after phase):
  
  ┌─────────┬─────────────────┬─────────────────┬───────┐
  │  Task   │    Baseline     │   Auto-memory   │ Delta │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ ER      │ 5.0%            │ 86.0%           │ +81pp │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ Agg     │ 5.0%            │ 41.0%           │ +36pp │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ Tr      │ 6.0%            │ 14.0%           │ +8pp  │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ Del     │ 34% (real=2%)   │ 54% (real=44%)  │ +20pp │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ Cas     │ 1.8% (real)     │ 37.8% (real=60) │ +36pp │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ Abs     │ 63.8% (trivial) │ 29.2% (real=38) │ −35pp │
  ├─────────┼─────────────────┼─────────────────┼───────┤
  │ Overall │ 19.6%           │ 42.5%           │ +23pp │
  └─────────┴─────────────────┴─────────────────┴───────┘

  Key takeaways:
  - Huge wins on ER (+81pp), Agg (+36pp), Cas (+36pp real) — the memory correctly stores and retrieves specific facts
  - Del improves dramatically in real passes (44 real vs 2 real) 
  - Abs drops numerically but the baseline's 64% was entirely trivial (said "I don't know" because it had no context at all); auto-memory's 29% are all
  genuine real passes
  - Tr (revision history) only improves modestly — memory stores current state but loses historical chains

  CLAUDE.md has been updated with all results. Ready to proceed to approach #3 (LLM Wiki) whenever you are.


## Cost Comparison: In-Context Baseline vs Auto-Memory
  
  How calls are made (from the code)

  In-context: for every question, feeds the full transcript (sessions 0..before_pos or 0..after_pos) + question as one big prompt. No memory — pure context
  window.

  Auto-memory: two-phase —
  1. Ingest: one claude -p call per evidence session (filler skipped), with session text + current memory state → writes/updates .md files
  2. Answer: one call per question with just the memory files + question (no transcript) 

  Token estimates (per 100 episodes, filler32k)
  Our implementation vs real Claude Code auto-memory behavior

  ┌───────────────────────┬────────────────────────────┬────────────────────────────┐
  │                       │     Our auto_memory.py     │      Real Claude Code      │
  ├───────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Sessions processed    │ Evidence only (~5/episode) │ All sessions (~24/episode) │
  ├───────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Ingest calls (100 ep) │ ~500                       │ ~2,400                     │
  ├───────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Avg input per ingest  │ ~1,900 tokens              │ ~1,900 tokens              │
  ├───────────────────────┼────────────────────────────┼────────────────────────────┤
  │ Ingest input tokens   │ ~940k                      │ ~4.6M                      │
  └───────────────────────┴────────────────────────────┴────────────────────────────┘

  Corrected cost estimate (if running on all sessions like real Claude Code)

  ┌───────────────┬─────────────┬────────────────────────┬───────────────────────┐
  │   Component   │ In-context  │ Auto-memory (our impl) │ Auto-memory (real CC) │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Ingest input  │ —           │ ~940k tokens           │ ~4.6M tokens          │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Answer input  │ ~44M tokens │ ~873k tokens           │ ~873k tokens          │
  └───────────────┴─────────────┴────────────────────────┴───────────────────────┘

  Corrected cost estimate (if running on all sessions like real Claude Code)

  ┌───────────────┬─────────────┬────────────────────────┬───────────────────────┐
  │   Component   │ In-context  │ Auto-memory (our impl) │ Auto-memory (real CC) │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Ingest input  │ —           │ ~940k tokens           │ ~4.6M tokens          │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Answer input  │ ~44M tokens │ ~873k tokens           │ ~873k tokens          │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Total input   │ ~44M        │ ~1.8M                  │ ~5.5M                 │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Total output  │ ~240k       │ ~490k                  │ ~1.2M                 │
  ├───────────────┼─────────────┼────────────────────────┼───────────────────────┤
  │ Est. API cost │ ~$136       │ ~$12.75                │ ~$34                  │
  └───────────────┴─────────────┴────────────────────────┴───────────────────────┘


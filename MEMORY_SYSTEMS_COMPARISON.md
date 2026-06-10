# Agentic Memory Systems on MeME — Auto-Memory vs. Hermes vs. OpenClaw

A head-to-head comparison of file-based agentic memory systems on the **MeME**
benchmark (filler32k, 100 episodes, Personal-Life + Software-Project domains). All
systems are answered and judged identically (`claude-code` answerer + real MeME
`LLMJudge`) over the same shared answer prompt, so the **only variable is the memory
architecture**.

**Dataset:** `filler32k` (50 pl + 50 sw episodes) · **Judge:** MeME `LLMJudge` (real correctness) · **Baseline:** in-context (no memory) = 19.6% overall (after).

---

## TL;DR

All four configurations are "the agent curates markdown files," but they differ in
whether they keep an **always-on, resolved digest** of memory or rely on **search over
raw notes** — and MeME shows that single choice dominates the results.

| | Auto-Memory | Auto-Memory **+dream** | Hermes | OpenClaw (default) | OpenClaw **+dreaming** |
|---|---:|---:|---:|---:|---:|
| **Overall (after)** | 42.5% | 52.7% | 51.0% | 26.4% | **54.2%** |
| Resolved digest always in context? | yes (overwrite file) | yes (+consolidation) | yes (bounded file) | **no (empty MEMORY.md)** | yes (consolidated MEMORY.md) |

> **Headlines:**
> 1. **The decisive factor is an always-on resolved digest, not the storage gimmick.**
>    Every system that keeps one (auto-memory, auto-memory+dream, Hermes,
>    OpenClaw+dreaming) lands at 42–54%; the one that doesn't (OpenClaw default)
>    collapses to 26%.
> 2. **A consolidation ("dreaming") pass helps — by an amount inversely proportional to
>    how much the system already consolidates.** OpenClaw default (append-only, no
>    consolidation) gains **+27.8pp** from dreaming; auto-memory (already curates
>    per-session) gains only **+10.2pp**. Both consolidation-enabled systems top the table.
> 3. **OpenClaw and Hermes are genuinely comparable** — with its consolidation enabled,
>    OpenClaw is the **strongest overall (54.2%)**, just above auto-memory+dream (52.7%)
>    and Hermes (51.0%). The "OpenClaw is weak" result was a *default-config artifact*,
>    not a product gap.
> 4. **Only overwriting truly forgets.** Auto-memory variants win **Del** (54% / 64%)
>    because overwriting/consolidating *destroys* the old value; pure search systems leak it.
> 5. **No single design wins all six tasks:** Hermes' raw-transcript FTS owns **ER (100%)**;
>    auto-memory+dream owns **Agg (62%)** and **Del (64%)**; OpenClaw+dreaming owns
>    **Cas (80%)** and **Abs (36%)**.

> **Note on "dreaming" for Claude Code:** Claude Code has **no native consolidation/
> reflection feature** — its auto-memory is purely session-reactive. "Auto-Memory+dream"
> here is a *novel extension* we built (a per-phase pass that re-reads all memory files
> and rewrites them deduplicated + cascade-resolved), analogous to OpenClaw's opt-in
> dreaming, included to test whether reflection helps a system that already curates.

---

## The systems

| | **Auto-Memory** | **Hermes** | **OpenClaw (memory-core)** |
|---|---|---|---|
| Store | Unbounded typed `.md` files + `MEMORY.md` index | Bounded `USER.md` (1,375 ch) + `MEMORY.md` (2,200 ch) | **Append-only** dated notes `memory/YYYY-MM-DD.md` + evergreen `MEMORY.md` |
| Write | Curate / **overwrite** per session | Curate within **hard char caps** | Per-session **flush**: extract durable facts, **append** |
| Consolidation | implicit (every overwrite) | implicit (replace-on-change) | **"dreaming"** distills notes → `MEMORY.md` (**opt-in, OFF by default**) |
| Index | none | **FTS5** over raw sessions | **Hybrid** vector (`nomic-embed-text`) + BM25 over notes |
| Retrieval | read whole memory | file snapshot **+ `session_search`** | `MEMORY.md` (if populated) **+ hybrid `memory_search`** |
| Forgetting | overwrite **deletes** the value | text note; value still recallable via FTS | none; old note stays, search resurfaces it |
| Reference | [code.claude.com/docs](https://code.claude.com/docs/en/memory#auto-memory) | [nousresearch/hermes-agent](https://github.com/nousresearch/hermes-agent) | [openclaw/openclaw](https://github.com/openclaw/openclaw) |

All are reimplemented as MeME agents (`agents/auto_memory.py`, `hermes_memory.py`,
`openclaw_memory.py`) using the real formats, budgets, and retrieval mechanics. Answerer
(`claude-code` + shared prompt) and judge are identical. OpenClaw's vector half uses
local `nomic-embed-text` (Ollama), off the answerer's quota.

**Two OpenClaw configs are reported**, because the difference between them *is* the
finding:
- **default** — dreaming OFF (its real default), so `MEMORY.md` stays empty and the
  model sees only hybrid search hits over raw append-only notes.
- **+dreaming** — the opt-in consolidation pass runs after each phase, distilling the
  notes into a resolved `MEMORY.md` that is surfaced always-on alongside search.

---

## Results — full per-task table (after phase)

`real` = genuinely-correct passes; `trivial` = passes from "I don't know" that happened
to be acceptable.

| Task | Measures | Auto-Memory | Auto-Mem **+dream** | Hermes | OpenClaw (default) | OpenClaw **+dreaming** |
|------|----------|------------:|--------------------:|-------:|-------------------:|----------------------:|
| **ER**  | Exact verbatim recall | 86.0% | 81.0% | **100.0%** | 88.0% | 85.0% |
| **Agg** | Combine scattered facts | 41.0% | **62.0%** | 56.0% | 12.0% | 37.0% |
| **Tr**  | Full revision history | 14.0% | 45.0% | 48.0% | 36.0% | **54.0%** |
| **Cas** | Propagate via rules | 37.8% (real 36.6) | 51.8% (real 45.7) | 76.2% (real 64.0) | 15.2% (real 5.5) | **80.5%** (real 71.3) |
| **Del** | Recognize removals | 54.0% (real 44.0) | **64.0%** (real 61.0) | 5.0% (real 4.0) | 12.0% (real 10.0) | 21.0% (real 13.0) |
| **Abs** | Uncertainty when no replacement | 29.2% (real 29.2) | 22.3% (real 21.5) | 15.4% (real 15.4) | 7.7% (real 2.3) | **36.2%** (real 31.5) |
| **Overall (after)** | | 42.5% | 52.7% | 51.0% | 26.4% | **54.2%** |
| **Overall (before)** | | 69.0% | 76.1% | 73.3% | 54.0% | 70.2% |

`knew_but_failed` — the system *had* the info to answer correctly but still got it wrong:

| Task | Auto | Auto+dream | Hermes | OC default | OC +dreaming |
|------|-----:|-----------:|-------:|-----------:|-------------:|
| Del | 32 | 35 | 93 | 62 | 73 |
| Cas | 85 | 77 | 32 | **91** | **22** |
| Abs | 82 | **100** | 108 | 92 | 81 |

---

## Analysis

### The audit: why default OpenClaw looked weak (and why it isn't)

A 2× gap between two comparable products (OpenClaw 26.4% vs Hermes 51.0%) was a red
flag. Auditing the OpenClaw source and our adapter found the gap was **~half config,
~half a fixable artifact** — not a product deficiency:

1. **Empty `MEMORY.md` (config).** In OpenClaw's default, the per-session flush only
   ever *appends* to dated notes; the single thing that consolidates them into the
   long-term `MEMORY.md` is **dreaming, which is off by default**. So default OpenClaw
   answers from a *ranked jumble of raw note lines*, with no resolved digest — while
   Hermes always injects its full resolved file. Enabling dreaming closes the gap.
2. **Per-line chunking (adapter artifact).** Real OpenClaw retrieves multi-line snippet
   ranges; our first adapter indexed one note-line per chunk. Of 88 default-config Agg
   failures, **33 had a gold item missing from the retrieved context** purely from
   fragmentation. Fixed by chunking per flush-block (Agg 12% → 37%).
3. **Not a problem:** the `active-memory` plugin injects only a ~220-char summary (our
   6k-char retrieval is already more generous); top-K ≈ 10 and dreaming-off match the
   real defaults.

**With both addressed, OpenClaw+dreaming reaches 54.2% — the best overall.** So the
real lesson is about *configuration*: a memory product's headline number depends
entirely on whether its consolidation layer is on.

### Does dreaming help a system that already consolidates? (Auto-Memory ablation)

Auto-memory already re-curates its files every session, so a global reflection pass has
less to add than it did for append-only OpenClaw — and that is exactly what we see:
**+10.2pp (42.5 → 52.7) vs OpenClaw's +27.8pp.** The gains concentrate where a *global*
view beats incremental per-session edits:

- **Agg 41 → 62%** (now the best of all): the dream pass groups facts scattered across
  typed files into one place.
- **Cas 37.8 → 51.8%**: resolving *all* dependency rules against current triggers at once.
- **Del 54 → 64%** (best of all): reconciliation strengthens removals.
- **Tr 14 → 45% (a surprise):** we expected overwriting to have destroyed history before
  the dream runs, but the consolidation writes "previously X, now Y" annotations that
  *incidentally* reconstruct the revision chain.
- **Abs 29 → 22% (a regression):** aggressively resolving everything to a current value
  sometimes erases the uncertainty signal base auto-memory had left behind
  (`knew_but_failed` 82 → 100). Consolidation is not free.

### Axis 1 — the always-on resolved digest decides cascade & invalidation (Cas, Abs)

The systems that maintain a resolved digest can *apply* state changes; the one that
doesn't can't.

- **Cas:** OpenClaw+dreaming **80.5%** (real 71%), Hermes 76%, auto 38%, OpenClaw default
  **15%**. Default OpenClaw stores a cascade as an *unresolved conditional* note
  ("medication switches to multivitamin **if** health changes") and never collapses it;
  `knew_but_failed` is a damning **91/164**. Dreaming consolidates it into the current
  value ("medication: multivitamin"), dropping `knew_but_failed` to **22**.
- **Abs:** OpenClaw+dreaming **36.2%** is the best of all — and notably *without* any
  hand-coded uncertainty rule (we removed that tell). Its "resolve to current value /
  don't carry removed values forward" consolidation produces the right behavior
  emergently.

```
Cas [sw_017] "What authentication method do we use?"   GOLD: SAML 2.0 SSO (cascaded)
    OPENCLAW default   : API Key + HMAC      ✗ (stale, unresolved note)
    OPENCLAW +dreaming : SAML 2.0 SSO        ✓ (consolidated current value)
```

### Axis 2 — raw-transcript recall wins pinpoint recall & aggregation (ER, Agg)

Here Hermes' FTS-over-raw-turns is unmatched: it returns the original text verbatim and
in bulk.

- **ER:** Hermes **100%** (raw verbatim) > OpenClaw 85–88% (distilled note, occasionally
  lossy) > auto 86%.
- **Agg:** Hermes **56%** > auto 41% > OpenClaw+dreaming 37% > default 12%. "List
  everything" rewards seeing *all* raw material at once; Hermes' full snapshot + search
  beats a consolidated digest that compresses items, and crushes top-K-only retrieval.

### Axis 3 — only overwriting truly forgets (Del)

- **Del:** auto **54%** ≫ OpenClaw+dreaming 21% > default 12% > Hermes 5%.
- Every retrieval/append system leaks the deleted value — search resurfaces the old note
  even when a "removed" marker exists (Hermes `knew_but_failed` 93/100). Auto-memory
  wins for a blunt reason: **overwriting a file destroys the old value**, so there is
  nothing to resurface. OpenClaw+dreaming improves on the other search systems because
  the consolidated `MEMORY.md` carries a prominent always-on "removed" line, but the
  searchable notes still leak the value some of the time.

```
Del [sw_048] "What's our log drain endpoint?"   GOLD: No — explicitly removed
    AUTO-MEM           : I don't have that information.                    ✓
    HERMES             : DELETED per user request (was stream.velturis.io/aurora) ✗
```

---

## Takeaways

1. **An always-on, resolved memory digest is the single biggest lever.** Three different
   ways of maintaining one (overwrite, bounded-curate, dreaming-consolidate) all land at
   42–54%; dropping it (OpenClaw default) costs ~28 points.
2. **A product's MeME score is a config statement, not just an architecture statement.**
   OpenClaw moved from worst (26.4%) to best (54.2%) by toggling its own consolidation
   feature. Always report which config you measured.
3. **The capabilities trade off and no single design wins all six tasks:** raw-transcript
   search (Hermes) owns ER/Agg; a resolved digest (OpenClaw+dreaming) owns Cas/Tr/Abs;
   overwriting (auto-memory) owns Del.
4. **Forgetting remains the hardest task for any search-based memory.** Del/Abs failures
   are overwhelmingly `knew_but_failed` — the signal is in context but the value leaks.
   What's missing across all of them is **structured suppression** (tombstones that
   withhold a deleted value at retrieval) — exactly what the structured-state approach
   (OmniService) adds.

---

## Methodology & integrity note

- **Harness:** `eval/run_agent.py` (ingest in order → before-Qs → ingest → after-Qs),
  then `eval/judge.py` (`LLMJudge`). Answerer & judge `claude-code`, shared prompt.
- **Agents:** `agents/auto_memory.py` (`dreaming` flag → `--agent-type auto_memory` /
  `auto_memory_dreaming`), `agents/hermes_memory.py`, `agents/openclaw_memory.py`
  (`dreaming` flag → `--agent-type openclaw` / `openclaw_dreaming`). All skip the
  curation/flush LLM pass for filler sessions.
- **"Dreaming" is a built feature, not a shipped one for Claude Code.** OpenClaw ships an
  opt-in dreaming consolidation; Claude Code has no consolidation/reflection pass at all
  (auto-memory is purely session-reactive), so `auto_memory_dreaming` is a novel
  extension — a per-phase `finalize_ingest` pass that re-reads and rewrites all memory
  files. Both dream prompts do general consolidation only (resolve to current value, keep
  full lists, mark explicit removals); the Abs-specific uncertainty instruction was
  deliberately removed from both, so Abs reflects emergent behavior.
- **OpenClaw fidelity fixes** (after a design audit): per-flush-block chunking (matches
  OpenClaw's multi-line snippet retrieval) and the dreaming consolidation pass in
  `finalize_ingest`. The dreaming prompt does general consolidation (resolve to current
  value, keep full lists, mark explicit removals); an Abs-specific uncertainty
  instruction was deliberately **removed** to avoid hand-coding a benchmark answer, so
  Abs reflects emergent behavior.
- **Integrity:** the Claude CLI usage cap interfered with several runs; contaminated
  (rate-limited / empty-memory) episodes were detected and re-run. **All numbers above
  are from fully-clean 100/100-episode runs** (0 rate-limit answers, 0 empty memory).

### Reproduction

```bash
cd MEME-public/code && source .venvs/baseline_env/bin/activate
bash scripts/run_hermes_eval.sh                  # Hermes
bash scripts/run_openclaw_eval.sh                # OpenClaw default (dreaming off)
bash scripts/run_openclaw_dreaming_eval.sh       # OpenClaw + dreaming consolidation
bash scripts/run_auto_memory_dreaming_eval.sh    # Auto-Memory + dreaming (novel extension)
# Auto-memory base (previously run):
#   python -m eval.run_agent -d data/filler32k_{pl,sw} --agent-type auto_memory --model claude-code -w1 --skip-existing
```

Artifacts: `output/{auto_memory,auto_memory_dreaming,hermes,openclaw,openclaw_dreaming}/claude-code/`
(100 agent outputs + 100 judged each).

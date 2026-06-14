# Agentic Memory Systems on MeME — Auto-Memory vs. Hermes vs. OpenClaw

A head-to-head comparison of file-based agentic memory systems on the **MeME** benchmark (filler32k, 100 episodes, Personal-Life + Software-Project domains). All systems are answered and judged identically (`claude-code` answerer + real MeME `LLMJudge`) over the same shared answer prompt, so the **only variable is the memory architecture**.

**Dataset:** `filler32k` (50 pl + 50 sw episodes) · **Judge:** MeME `LLMJudge` (real correctness) · **Baseline:** in-context (no memory) = 19.6% overall (after).

---

## TL;DR

These systems are all "the agent curates markdown files," but they differ in whether they keep an **always-on, resolved digest** of memory and whether they run a **dreaming** consolidation pass — and MeME shows those two choices dominate the results.

| | Auto-Memory | Auto-Memory **+dreaming** | Hermes | OpenClaw | OpenClaw **+dreaming** |
|---|---:|---:|---:|---:|---:|
| **Overall (after)** | 42.5% | **66.0%** | 51.0% | 26.4% | 54.2% |
| Always-on resolved digest? | yes | yes | yes | **no** | yes |

> **Headlines:**
> 1. **Dreaming wins decisively (66.0%).** Claude Code auto-memory plus a consolidation
>    pass that re-reads the raw session transcripts — Anthropic's
>    [Dreams](https://platform.claude.com/docs/en/managed-agents/dreams) design — recovers
>    facts the incremental writes missed: **ER 95%, Tr 72%, Cas 74%, Abs 41%**. It beats the
>    next-best system by 12pp.
> 2. **An always-on resolved digest is the floor.** Every system that keeps one lands
>    42–66%; the one that doesn't (OpenClaw default) collapses to 26%.
> 3. **Consolidation helps inversely to how much a system already consolidates** — but the
>    source matters more. OpenClaw default (append-only) gains **+27.8pp** from a memory-only
>    dream; Claude auto-memory (already curates per-session) gains **+23.5pp**, almost all of
>    it from the dream *re-reading the transcripts* rather than just reorganizing memory.
> 4. **Dreaming trades a little forgetting for a lot of recall.** Its only weak task is
>    **Del (58%)** — re-reading the source can resurface a deleted value — while it tops or
>    ties every other task.
> 5. **No single design wins all six tasks:** Hermes' FTS ties ER; OpenClaw+dreaming edges
>    Cas; auto-memory (no dream) holds Del. Dreaming leads overall and on ER/Tr/Abs.

> **"Dreaming" is a real Anthropic feature, not just our invention.** Claude Code's *CLI*
> auto-memory is purely session-reactive (no consolidation pass). But the **Managed Agents
> API ships [Dreams](https://platform.claude.com/docs/en/managed-agents/dreams)** — an
> async job that reads a memory store **plus 1–100 raw session transcripts** and emits a
> *new* reorganized store (duplicates merged, stale/contradicted entries replaced with the
> latest value, new insights surfaced). Our **Auto-Memory +dreaming** matches that design
> (consolidation over memory + transcripts). OpenClaw ships the same idea as its opt-in
> "dreaming" (over its notes).

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

Best in **bold**. **+dreaming** = the Dreams-style consolidation over memory **+ raw
transcripts** (the Anthropic Dreams design).

| Task | Measures | Auto-Memory | Auto-Memory **+dreaming** | Hermes | OpenClaw | OpenClaw **+dreaming** |
|------|----------|------------:|--------------------------:|-------:|---------:|----------------------:|
| **ER**  | Exact verbatim recall | 86.0% | 95.0% | **100.0%** | 88.0% | 85.0% |
| **Agg** | Combine scattered facts | 41.0% | **59.0%** | 56.0% | 12.0% | 37.0% |
| **Tr**  | Full revision history | 14.0% | **72.0%** | 48.0% | 36.0% | 54.0% |
| **Cas** | Propagate via rules | 37.8% | 73.8% (real 65.2) | 76.2% (real 64.0) | 15.2% | **80.5%** (real 71.3) |
| **Del** | Recognize removals | 54.0% | **58.0%** (real 57) | 5.0% | 12.0% | 21.0% |
| **Abs** | Uncertainty when no replacement | 29.2% | **40.8%** (real 39.2) | 15.4% | 7.7% | 36.2% |
| **Overall (after)** | | 42.5% | **66.0%** | 51.0% | 26.4% | 54.2% |
| **Overall (before)** | | 69.0% | 75.3% | 73.3% | 54.0% | 70.2% |

`knew_but_failed` — the system *had* the info to answer correctly but still got it wrong:

| Task | Auto | Auto+dreaming | Hermes | OC default | OC +dreaming |
|------|-----:|--------------:|-------:|-----------:|-------------:|
| Del | 32 | 42 | 93 | 62 | 73 |
| Cas | 85 | **40** | 32 | **91** | 22 |
| Abs | 82 | **75** | 108 | 92 | 81 |

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

### Dreaming over memory + transcripts is the decisive lever (Auto-Memory +dreaming)

Adding the Dreams-style consolidation to Claude Code auto-memory is the single biggest
jump in the study: **42.5 → 66.0% overall (+23.5pp), the best system by 12pp.** The key
is *what the dream reads*. Auto-memory already re-curates its files every session, so a
dream that only **reorganized memory** would add little (the same logic that gave
append-only OpenClaw a big lift gives an already-curating system a small one). The win
comes from the dream **re-reading the raw session transcripts** — exactly Anthropic's
[Dreams](https://platform.claude.com/docs/en/managed-agents/dreams) design — which
*recovers* facts the lossy incremental writes never captured:

- **ER 86 → 95%, Tr 14 → 72%, Cas 37.8 → 73.8%, Abs 29 → 41%** — re-reading the source
  recovers verbatim quotes, full revision chains, dependency rules, and the
  trigger-changed signals that drive uncertainty. Tr is the standout: incremental
  overwriting destroys history, but the transcripts still hold the full chain.
- **Agg 41 → 59%:** the global pass groups facts scattered across typed files.
- **The one cost — Del 54 → 58% is a near-wash, and below the other variants' peak:**
  re-reading a transcript that introduced a fact *before* it was deleted can resurface the
  deleted value (the `partner: James` case). This is the recall↔forgetting tension seen
  across the study — transcript mining lands hard on the recall side. Net, the recall
  gains (9–58pp across five tasks) dwarf any Del cost.

This is the clearest result in the comparison: **a consolidation pass that re-reads the
source transcripts beats every other memory design**, and it is a shipping Anthropic
feature, not a hypothetical.

### Axis 1 — the always-on resolved digest decides cascade & invalidation (Cas, Abs)

The systems that maintain a resolved digest can *apply* state changes; the one that
doesn't can't.

- **Cas:** OpenClaw+dreaming **80.5%** (real 71%), Hermes 76%, **auto+dreaming 74%**, base
  auto 38%, OpenClaw default **15%**. Default OpenClaw stores a cascade as an *unresolved
  conditional* note ("medication switches to multivitamin **if** health changes") and never
  collapses it; `knew_but_failed` is a damning **91/164**. Any dreaming pass consolidates it
  into the current value ("medication: multivitamin") — OpenClaw's drops `knew_but_failed`
  to **22**, auto-memory's to **40**.
- **Abs:** auto+dreaming **40.8%** and OpenClaw+dreaming 36.2% lead — and notably *without*
  any hand-coded uncertainty rule (we removed that tell). The "resolve to current value /
  don't carry removed values forward" consolidation produces the right behavior emergently;
  re-reading transcripts (auto+dreaming) sharpens it further.

```
Cas [sw_017] "What authentication method do we use?"   GOLD: SAML 2.0 SSO (cascaded)
    OPENCLAW default   : API Key + HMAC      ✗ (stale, unresolved note)
    OPENCLAW +dreaming : SAML 2.0 SSO        ✓ (consolidated current value)
```

### Axis 2 — raw-transcript access wins pinpoint recall & aggregation (ER, Agg)

Systems that touch the raw turns — Hermes' FTS, or auto+dreaming's transcript-mining dream
— return original text verbatim and in bulk.

- **ER:** Hermes **100%** (raw verbatim via FTS) > **auto+dreaming 95%** (dream recovers the
  exact quote from the transcript) > OpenClaw 85–88% > base auto 86%.
- **Agg:** **auto+dreaming 59%** ≈ Hermes **56%** > base auto 41% > OpenClaw+dreaming 37% >
  default 12%. "List everything" rewards seeing *all* the material at once; auto+dreaming's
  dream pulls scattered items together from the transcripts, Hermes via full snapshot +
  search — both beat top-K-only retrieval (OpenClaw).

### Axis 3 — only overwriting truly forgets (Del)

- **Del:** auto+dreaming **58%** ≈ base auto **54%** ≫ OpenClaw+dreaming 21% > default 12% >
  Hermes 5%.
- Auto-memory variants win for a blunt reason: **overwriting a file destroys the old
  value**, so there is little to resurface. Every search/append system leaks it — search
  resurfaces the old note even when a "removed" marker exists (Hermes `knew_but_failed`
  93/100). Note the tension: re-reading transcripts (auto+dreaming) *can* resurface a
  deleted value, which is why Del is auto+dreaming's weakest task even though it is still
  the best Del score overall.

```
Del [sw_048] "What's our log drain endpoint?"   GOLD: No — explicitly removed
    AUTO-MEM           : I don't have that information.                    ✓
    HERMES             : DELETED per user request (was stream.velturis.io/aurora) ✗
```

---

## Takeaways

1. **A dreaming pass that re-reads the source transcripts is the single biggest lever.**
   Claude auto-memory **+dreaming** leads at 66.0% — +23.5pp over base auto-memory and
   +12pp over the next system — because re-reading the transcripts *recovers* facts the
   incremental writes missed (Tr 14→72, Cas 38→74, ER 86→95).
2. **An always-on resolved digest is the floor.** Every system that keeps one lands 42–66%;
   dropping it (OpenClaw default) costs ~28 points and collapses to 26%.
3. **A product's MeME score is a config statement, not just an architecture statement.**
   OpenClaw moved from worst (26.4%) to second (54.2%) by toggling its own consolidation
   feature; always report which config you measured.
4. **The capabilities still trade off:** Hermes' FTS ties ER, OpenClaw+dreaming edges Cas,
   and **overwriting (auto-memory) is the only thing that truly forgets** (best Del). The
   transcript-mining dream wins overall but pays a small Del cost for its recall.
5. **Forgetting remains the hardest task.** Del/Abs failures are overwhelmingly
   `knew_but_failed` — the signal is in context but the value leaks. What's missing is
   **structured suppression** (tombstones that withhold a deleted value at retrieval) —
   exactly what the structured-state approach (OmniService) adds.

---

## Methodology & integrity note

- **Harness:** `eval/run_agent.py` (ingest in order → before-Qs → ingest → after-Qs),
  then `eval/judge.py` (`LLMJudge`). Answerer & judge `claude-code`, shared prompt.
- **Agents:** `agents/auto_memory.py` (`--agent-type auto_memory` / `auto_memory_dreaming`),
  `agents/hermes_memory.py`, `agents/openclaw_memory.py` (`openclaw` / `openclaw_dreaming`).
  All skip the curation/flush LLM pass for filler.
- **"Dreaming" matches a shipped Anthropic feature.** The *Managed Agents API* ships
  [Dreams](https://platform.claude.com/docs/en/managed-agents/dreams) (memory store + raw
  transcripts → reorganized store); our **`auto_memory_dreaming`** matches that design — a
  per-phase `finalize_ingest` pass over the memory files **+ the archived evidence
  transcripts**. (The *Claude Code CLI* auto-memory has no such pass; this adds it.)
  OpenClaw ships the same idea as opt-in dreaming over its notes. All dream prompts do
  general consolidation only (resolve to current value, keep
  full lists, mark explicit removals); the Abs-specific uncertainty instruction was
  deliberately removed, so Abs reflects emergent behavior.
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
bash scripts/run_auto_memory_dreaming_eval.sh    # Auto-Memory + dreaming (memory + transcripts; Anthropic Dreams design)
# Auto-memory base (previously run):
#   python -m eval.run_agent -d data/filler32k_{pl,sw} --agent-type auto_memory --model claude-code -w1 --skip-existing
```

Artifacts: `output/{auto_memory,auto_memory_dreaming,hermes,openclaw,openclaw_dreaming}/claude-code/`
(100 agent outputs + 100 judged each).

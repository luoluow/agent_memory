# Agent Memory

This project evaluates different approaches to managing agent memory for long-term interactions. The eval framework is [MeME](https://github.com/SeokwonJung-Jay/MEME-public), modified to run on Claude Code using a Claude Pro subscription instead of an API key.

## Approaches

### 1. In-context (baseline)
No external memory. The full episode transcript — including all filler sessions (~42k tokens for filler32k) — is fed directly to the LLM for every question. Nothing is stored or summarized between sessions.

**Limitation:** The model must search a noisy, very long context to answer each question. Works only if the relevant fact is still in the window and not buried.

---

### 2. Auto-memory ([Claude Code built-in](https://code.claude.com/docs/en/memory#auto-memory))
After each evidence session, one LLM call decides what facts are worth saving as typed Markdown files (categorized as `user`, `feedback`, `project`, `reference`). At question time, all files are read directly — no LLM call needed for retrieval.

**Strength:** Extracts and stores facts cleanly. Verbatim values are preserved. At retrieval time the full memory is available, so nothing is missed.

**Limitation:** Memory is append-and-overwrite only. When a fact is deleted or revised, the old value must be explicitly overwritten; if the ingest LLM misses that, stale values linger. No mechanism for synthesizing across multiple entities.

---

### 3. LLM Wiki ([Karpathy](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f))
Maintains a set of interlinked entity pages (one page per person, project, medication, etc.) with timestamped attribute lines and `[[cross-link]]` references. An index page maps entity names to file names.

At question time, a first LLM call reads the index and selects which pages are relevant; those pages are then read and passed to a second LLM call for the answer.

**Strength:** Revision history is preserved as timestamped lines, so the model can reconstruct what changed and when (strong Tr and Agg scores). Cross-links help with multi-entity questions.

**Limitation:** Entity pages paraphrase facts rather than storing verbatim values, hurting ER. The two-LLM retrieval path doubles answer-phase cost and adds latency.

---

### 4. EvoMemory (inspired by [Evo-Memory / ReMem](https://arxiv.org/pdf/2511.20857))
Shares Auto-memory's ingest mechanism but adds a **Refine** pass at the end of each phase. After all evidence sessions are ingested, a single LLM call reads the entire memory store and reorganizes it: merging duplicates, resolving contradictions, and prominently marking deleted or discontinued facts at the top of each file.

At question time, files are read directly (same as Auto-memory) — no extra LLM call.

**Strength:** The Refine step produces a clean, coherent, contradiction-free snapshot. Deletions and cancellations are surfaced explicitly, so the model never returns a stale value when answering Del, Cas, or Abs questions. Best overall score (52.2%).

**Limitation:** Refine optimizes for the current state, discarding revision chains. This makes tracking history (Tr) harder than Wiki or A-Mem. Two extra LLM calls per episode (one Refine per phase).

---

### 5. A-Mem ([Zettelkasten / A-MEM](https://arxiv.org/pdf/2502.12110))
Each evidence session produces one **atomic note** with LLM-generated keywords, tags, and a contextual summary. Notes are embedded (all-MiniLM-L6-v2) and linked to the most similar existing notes via a second LLM call that also **evolves** those neighbors — updating their context if the new note adds relevant information.

At question time, the query is embedded and the top-k most similar notes are retrieved by cosine similarity; their linked notes are included too. No LLM call at retrieval.

**Strength:** The evolving link network naturally aggregates facts scattered across sessions (strong Agg, 82%) and the cumulative evolution of notes preserves revision context (strong Tr, 69%). Embedding-based retrieval is fast and focused.

**Limitation:** Atomic notes summarize rather than transcribe, so verbatim values are lost (ER 59% vs Auto-memory 86%). Deletions are not explicitly marked — an evolved note can silently overwrite a deletion signal, producing the weakest Del (32%) and Abs (12%) scores.

---

## Methodology vs. Results

The six MeME task types probe fundamentally different memory properties. The methodology choices above map directly onto the score gaps:

### Exact Recall (ER) — *does memory preserve verbatim values?*
Auto-memory (86%) wins because it stores the raw fact string into a file without paraphrasing. Wiki (3%) and A-Mem (59%) both summarize, losing the exact value. EvoMemory (76%) preserves facts well through Refine but occasionally rewrites phrasing.

### Aggregation (Agg) — *can memory combine facts across sessions?*
A-Mem (82%) and Wiki (65%) dominate because their architectures are explicitly relational: A-Mem's evolving links cluster related notes; Wiki's entity pages co-locate facts. Auto-memory (41%) stores per-session files without cross-linking, making cross-entity synthesis harder.

### Tracking (Tr) — *can memory reconstruct revision history?*
Wiki (64%) and A-Mem (69%) preserve history: Wiki keeps timestamped attribute lines; A-Mem's evolution log retains prior context. EvoMemory (30%) and Auto-memory (14%) discard old values when overwriting — Refine specifically prunes "superseded" facts.

### Deletion (Del) — *does memory recognize when a fact was removed?*
EvoMemory (97%) dominates because its Refine prompt explicitly instructs the LLM to mark cancelled/deleted facts at the top of the relevant file. Auto-memory (54%) sometimes catches deletions, sometimes overwrites silently. A-Mem (32%) has no deletion-awareness mechanism — an evolved note can absorb a deletion without flagging it.

### Cascade (Cas) — *does memory propagate updates through dependent facts?*
EvoMemory (56%) is strongest because Refine consolidates all memory in one pass, naturally propagating updates. A-Mem (40%) propagates via its link-evolve step but only to linked neighbors. Wiki (22%) maintains per-entity pages but doesn't enforce cross-page consistency.

### Absence (Abs) — *does memory correctly signal uncertainty when no replacement applies?*
EvoMemory (62%) and Auto-memory (29%) can distinguish "I know this was deleted" from "I never knew this." A-Mem (12%) scores lowest because evolved notes don't retain the deletion signal — the model infers a current value where none exists.

---

## Eval Results

**All approaches — after phase (filler32k, 100 episodes)**

| Task | In-context | Auto-memory | LLM Wiki | EvoMemory | A-Mem |
|------|-----------|-------------|----------|-----------|-------|
| ER | 5.0% | **86.0%** | 3.0% | 76.0% | 59.0% |
| Agg | 5.0% | 41.0% | 65.0% | 61.0% | **82.0%** |
| Tr | 6.0% | 14.0% | 64.0% | 30.0% | **69.0%** |
| Del | 34.0% | 54.0% | 60.0% | **97.0%** | 32.0% |
| Cas | 1.8% | 37.8% | 22.0% | **56.1%** | 40.2% |
| Abs | 63.8%* | 29.2% | 32.0% | **62.3%** | 12.3% |
| **Overall** | **19.6%** | **42.5%** | **42.2%** | **52.2%** | **46.7%** |

\* In-context Abs passes are all trivial (model said "I don't know" because it had no memory).

---

## Cost Analysis

**Per 100 episodes, filler32k — priced at Claude Sonnet 4.6 API rates ($3/M input, $15/M output)**

| Approach | LLM Calls/ep | Est. Tokens/ep | $/ep | $/100ep | Wall/ep | Score |
|----------|-------------|---------------|------|---------|---------|-------|
| In-context | 12 | 503k | $1.52 | $152 | ~90s | 19.6% |
| Auto-memory | 17 | 22.5k | $0.11 | $11 | 162s | 42.5% |
| LLM Wiki | 29 | 27.5k | $0.14 | $14 | 215s | 42.2% |
| EvoMemory | 19 | 24.1k | $0.16 | $16 | **104s** | **52.2%** |
| A-Mem | 22 | 35.1k | $0.17 | $17 | 219s | 46.7% |

*Actual runs used `claude -p` (Claude Pro subscription). Token counts are estimated — `claude -p` does not expose token usage.*

**Call breakdown per episode (5 evidence sessions, ~12 questions):**
- **In-context:** 12 × answer (42k tokens each)
- **Auto-memory:** 5 × ingest + 12 × answer
- **LLM Wiki:** 5 × ingest + 12 × retrieve (LLM) + 12 × answer
- **EvoMemory:** 5 × ingest + 2 × Refine + 12 × answer
- **A-Mem:** 5 × note-construct + 5 × link-evolve + 12 × answer

**Takeaway:** All memory approaches are 10–14× cheaper than in-context at filler32k scale. Among memory approaches, costs are within 50% of each other ($11–$17/100ep) while performance ranges from 42% to 52%. EvoMemory offers the best score-per-dollar at 3.3 pp/$, closely followed by Auto-memory at 3.9 pp/$.

---

## How to Run

```bash
cd MEME-public/code
source .venvs/baseline_env/bin/activate

# Run an approach (paced to respect Claude session limits)
python scripts/run_paced.py --agent evomem --domain both --inter-sleep 90

# Judge results
python -m eval.judge \
  -d ../../output/evomem/claude-code \
  -o ../../output/evomem/claude-code/judge \
  --judge-model claude-code -w 1 --check-workers 4 --skip-existing
```

Agents: `auto_memory`, `wiki`, `evomem`, `amem`. See `CLAUDE.md` for full setup and dataset download instructions.

---

## LICENSE & DISCLAIMER

[DISCLAIMER](DISCLAIMER.md) | [LICENSE](LICENSE)

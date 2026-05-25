# Eval Process for Agentic Memory

## Run Agent with the dataset to generate output for judging

### Dataset

  The filler32k dataset is pre-built by the MeME authors. Each episode file (data/filler32k_sw/episode_001.json) contains:
  - episode_id, domain
  - sessions — a list of conversation sessions, each with timestamped turns. The bulk of tokens are "filler" conversations unrelated to the tracked
  entities, making retrieval hard
  - before_questions — questions + gold answers + position_after_session (which session index to stop at)
  - after_questions — same but using the full transcript
  
### Running the agent (in_context_baseline.py)

  For each episode, process_one_episode():

  1. Loads the episode JSON
  2. Builds two transcripts by calling _flatten_sessions():
    - transcript_before — sessions 0 through before_questions.position_after_session
    - transcript_after — sessions 0 through after_questions.position_after_session (the full transcript)
  3. For each before-question, calls _ask() which sends:
  System: "You are answering a question based on a long conversation transcript..."
  User:   "Transcript:\n{transcript_before}\n\nQuestion: {question}"
  3. and records the answer
  4. For each after-question, repeats with transcript_after
  5. Writes agent_sw_001_in_context_claude-code.json with:
  {
    "episode_id": 1,
    "domain": "software_project",
    "before_answers": [{ "question": "...", "entity_values": {...}, "task_type": "ER", "agent_answer": "..." }],
    "after_answers":  [{ ... }],
    "config": { "agent_type": "in_context", "agent_model": "claude-code" }
  }

### The LLM client (claude_code_adapter.py)

  We used the ClaudeCodeAsOpenAI adapter instead of the real OpenAI/Anthropic SDK. It:
  - Implements the same client.chat.completions.create() interface
  - Internally runs claude -p --output-format text --no-session-persistence as a subprocess
  - Passes the user messages as stdin and the system prompt via --system-prompt
  {
    "episode_id": 1,
    "domain": "software_project",
    "before_answers": [{ "question": "...", "entity_values": {...}, "task_type": "ER", "agent_answer": "..." }],
    "after_answers":  [{ ... }],
    "config": { "agent_type": "in_context", "agent_model": "claude-code" }
  }

### Parallelism

  in_context_baseline.py supports -w N workers via ProcessPoolExecutor. Each worker process handles one episode independently — no shared state. We used -w
   1 to avoid rate-limit issues with the Claude CLI.

### Output

  100 agent output files land in output/, one per episode (50 PL + 50 SW). These are the inputs to the judge.

## Judge the output
Here's the full judging workflow:

  1. Input: Agent output files

  Each file (agent_pl_001_in_context_claude-code.json) contains:
  - before_answers — the agent's answers to pre-update questions
  - after_answers — the agent's answers to post-update questions
  - Each answer has: question, entity_values (gold), task_type, agent_answer

  2. Per-question correctness check (u_check / u_check_multi)

  Each question is judged by task type:

  ┌───────────────────┬───────────────────────────────────────────────────────────┐
  │       Task        │                          Method                           │
  ├───────────────────┼───────────────────────────────────────────────────────────┤
  │ ER (Exact Recall) │ Deterministic substring match — no LLM call               │
  ├───────────────────┼───────────────────────────────────────────────────────────┤
  │ Tr (Tracking)     │ LLM checks if agent lists all historical values in order  │
  ├───────────────────┼───────────────────────────────────────────────────────────┤
  │ Agg (Aggregation) │ Single LLM call checks all required entity values at once │
  ├───────────────────┼───────────────────────────────────────────────────────────┤
  │ Cas (Cascade)     │ LLM checks if agent states only the new cascaded value    │
  ├───────────────────┼───────────────────────────────────────────────────────────┤
  │ Abs (Absence)     │ LLM checks if agent expresses uncertainty                 │
  ├───────────────────┼───────────────────────────────────────────────────────────┤
  │ Del (Deletion)    │ LLM checks if agent avoids revealing the deleted value    │
  └───────────────────┴───────────────────────────────────────────────────────────┘

  Before-questions (except ER) use a generic "does the answer contain the correct info?" prompt. After-questions use the task-specific prompts above.

  3. Trivial-pass classification (Cas / Abs / Del only)

  These three tasks can be "passed" trivially by a model that simply says "I don't know" to everything. To distinguish genuine knowledge from lucky
  guesses, the judge cross-references each entity's before-question result:

  ┌─────────────┬────────────┬────────────────────────────────────────────────────────────────┐
  │ Before pass │ After pass │                             Label                              │
  ├─────────────┼────────────┼────────────────────────────────────────────────────────────────┤
  ├─────────────┼────────────┼────────────────────────────────────────────────────────────────┤
  │ ✗           │ ✓          │ trivial — agent got lucky (never knew the value to begin with) │
  │ ✗           │ ✓          │ trivial — agent got lucky (never knew the value to begin with) │
  ├─────────────┼────────────┼────────────────────────────────────────────────────────────────┤
  │ ✓           │ ✗          │ knew_but_failed — agent knew but failed to apply the update    │
  ├─────────────┼────────────┼────────────────────────────────────────────────────────────────┤
  │ ✗           │ ✗          │ never_knew — agent never knew                                  │
  └─────────────┴────────────┴────────────────────────────────────────────────────────────────┘

  In our results, Abs had 83 passes — but all 83 were trivial: the model said "I don't know" not because it understood the upstream changed, but because it
   couldn't find the answer in 32k tokens of filler context.

  4. Output: Eval files

  Each episode produces eval_pl_001_in_context_claude-code_claude-code.json with per-answer pass/fail, trivial-pass labels, and totals.

  5. Aggregation

  A final script sums across all 100 episodes to produce the grand total — before %, after %, and per-task Cas/Abs/Del breakdowns with real vs. trivial
  counts.

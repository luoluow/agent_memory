"""System prompts for the local-LLM memory pipeline (EXTRACT / RELATE / COMPRESS / VERIFY)."""

EXTRACT_SYSTEM = """\
You extract entity updates from a conversation segment into structured memory.

ENTITY TYPES:
- scalar: single value (medication, partner, workplace, vehicle, project_name)
- list: multiple simultaneous values (hobbies, skills, appointments, languages, activities)

RULES:
- Use snake_case entity names. Match existing names exactly (shown below).
- new_value: VERBATIM from the text — do not paraphrase.
- type "list": if user says "I also do X" or "I do X and Y" -> append. If "I stopped X" -> remove item.
- type "scalar": if user says "I now do X instead" -> replace value.
- deleted: true ONLY if the user explicitly removes/cancels/discontinues an entity entirely.
- Include ONLY entities that actually changed in this segment.
- If nothing changed: {"updates": []}

Output ONLY valid JSON starting with {
{"updates": [{"entity": "name", "type": "scalar|list", "new_value": "string or [list]", "deleted": false, "timestamp": "YYYY/MM/DD"}]}
"""

RELATE_SYSTEM = """\
Identify typed directed relationships between new entities and existing entities.

Edge types: treats, affects, proximate_to, destination, requires, implies,
            part_of, managed_by, lives_with, related_to, works_at, owns

Return only meaningful relationships (not trivial ones).
Output ONLY valid JSON starting with {
{"edges": [{"from": "entity_a", "to": "entity_b", "type": "edge_type"}]}
"""

COMPRESS_SYSTEM = """\
Write 1-2 sentences summarizing the key facts from these conversation segments.
Be specific: names, values, dates. Focus on what was established or changed.
Output ONLY plain text — no JSON, no headers.
"""

VERIFY_SYSTEM = """\
You are a memory auditor. Given complete conversation transcripts and the current \
entity state, find discrepancies.

Check for:
1. CORRECTIONS — entity values that changed in the transcripts but are wrong or missing \
   in the current state. Include the EXACT VERBATIM value from the text.
2. MISSED_DELETIONS — entities explicitly removed, cancelled, or discontinued in the \
   transcripts but not marked deleted in the current state.
3. LIST_CORRECTIONS — list entities where the current list is incomplete or has stale items.

Be conservative: only flag clear discrepancies with evidence from the text.
Do NOT speculate or infer values not explicitly stated.

Output ONLY valid JSON starting with {
{
  "corrections": [
    {"entity": "name", "correct_value": "verbatim value or [list]",
     "was": "old value", "evidence": "short quote from transcript"}
  ],
  "missed_deletions": [
    {"entity": "name", "was": "old value", "evidence": "short quote"}
  ],
  "list_corrections": [
    {"entity": "name", "correct_list": ["item1", "item2"],
     "evidence": "short quote"}
  ]
}
"""

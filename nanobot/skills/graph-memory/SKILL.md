---
name: graph-memory
description: "Graph-based associative memory with two-phase recall: anchor + time-chain backtracking, then spreading activation. Use when remembering, recalling, or managing persistent knowledge."
metadata: {"nanobot":{"always":true}}
---

# Graph Memory

You have a graph-structured memory system. Memories are stored as nodes (events, concepts, entities) connected by edges forming an association network. Retrieval mimics human recall: anchor on a concept, walk the timeline, then spread to discover distant associations.

## Tools

| Tool | When to use |
|------|-------------|
| `find_memory_cache` | Recall memories by keyword. Returns a structured timeline with lateral context per event. |
| `recall_related` | Dig deeper from a specific event. Spreads activation to discover distant associations. |
| `get_memory_cache` | Get full details of specific nodes by ID. |
| `create_memory` | Store a new memory. Always provide descriptive, normalized keywords. |
| `update_memory` | Correct or enrich an existing memory node. |
| `delete_memory` | Remove outdated or incorrect memories. |

## Recall Pattern

Follow this chain when recalling:

```
1. find_memory_cache("topic")     → structured timeline + initial context
2. (optional) recall_related(event_id="xxx")  → spread from a specific event
3. Synthesize into a coherent answer
```

This simulates: "remember → follow the thread → discover more"

## When to Recall

Use `find_memory_cache` when:
- The conversation starts — proactively check if you have relevant memories for the topic
- The user references something from the past ("remember when...", "what did we discuss about...")
- You need context about a topic, person, or project the user has mentioned before

## When to Remember

Use `create_memory` when the user shares:
- Personal facts (preferences, location, relationships)
- Project context (tech stack, decisions, deadlines)
- Events worth recalling later (meetings, discussions, milestones)

Do NOT over-create memories. Only store information with long-term value.

## Keyword Strategy

Good keywords enable associative recall. Include:
- **WHO**: people, teams, organizations
- **WHAT**: topics, technologies, projects
- **WHERE**: locations, channels, contexts
- **WHEN**: date references if significant (e.g. "2025-q1")

Keywords are auto-normalized (lowercased, trimmed). When a keyword is referenced by ≥2 events, it automatically becomes a concept hub node.

Example:
```
User: "We discussed the Python migration plan at the pub last Friday"
→ create_memory(
    content="Discussed Python migration plan at the pub",
    keywords=["pub", "python", "migration", "architecture"],
    type="event"
  )
```

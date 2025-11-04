# SABRE - Memory System Plan

## Problem Statement

**Current State:**
- Each conversation starts from scratch
- No knowledge persists between conversations
- Python namespace variables lost on restart
- Same expensive operations repeated (schema queries, topology discovery)
- No learning from past incidents or investigations

**What We Want:**
- Knowledge accumulates across conversations
- Learn from experience (incidents, patterns, solutions)
- Remember structural facts (schemas, topologies, configurations)
- Build expertise over time for each persona
- Fast access to previously learned information

## Core Insight: Persona + Memory = Expertise

Personas provide **example workflows** (how to approach tasks).
Memory provides **accumulated knowledge** (what has been learned).

Together they create an expert that gets smarter with use:
- **Week 1:** Data analyst follows examples, discovers database schema
- **Week 2:** Data analyst remembers schema, focuses on analysis
- **Week 3:** Data analyst has baselines, can spot anomalies instantly

## Memory Architecture

### Five Memory Types (Cognitive Model)

#### 1. **Working Memory** - Python Runtime Namespace

**What it is:** Variables, imports, functions in current conversation

**Scope:** Per-conversation (persists across turns, lost when conversation ends)

**Storage:** Python namespace in memory (already exists!)

**Example:**
```
# Turn 1
<helpers>
schema = Database.get_schema("orders")  # Fetched, stored in namespace
</helpers>

# Turn 2 (same conversation)
<helpers>
print(schema['columns'])  # Still available!
</helpers>

# [Conversation ends]

# Turn 1 (new conversation)
<helpers>
print(schema)  # ❌ NameError - namespace cleared
</helpers>
```

**Duration:** Until conversation ends or server restarts

**No changes needed:** This already works in SABRE's continuation model

---

#### 2. **Semantic Memory** - Facts & Knowledge

**What it is:** Timeless facts (schemas, topologies, definitions, configurations)

**Scope:** Global per-persona (shared across all conversations for that persona)

**Storage:** SQLite with TTL-based expiration

**Schema:**
```sql
CREATE TABLE semantic_facts (
    key TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    value JSON NOT NULL,
    persona TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    access_count INTEGER DEFAULT 0
);
```

**Example:**
```
# Conversation A (data-analyst persona)
<helpers>
# First time: fetch and cache
schema = Database.get_schema("orders")
remember_fact("database.orders.schema", schema, ttl=3600)  # 1 hour
</helpers>

# [Conversation ends, server restarts]

# Conversation B (data-analyst persona, next day)
<helpers>
# Check semantic memory first
schema = recall_fact("database.orders.schema")

if not schema:
    # Cache miss - fetch and store
    schema = Database.get_schema("orders")
    remember_fact("database.orders.schema", schema, ttl=3600)

# Now can use schema immediately
</helpers>
```

**Auto-persistence strategy:**
- Expensive helper calls (>1s execution time) auto-cached
- Structural data (schemas, topologies) marked with `@cacheable` decorator
- LLM can explicitly call `remember_fact()` for important knowledge

**File location:** `~/.local/share/sabre/memory/{persona}/semantic/facts.db`

---

#### 3. **Episodic Memory** - What Happened When

**What it is:** Events, incidents, investigations with temporal context

**Scope:** Global per-persona (can be filtered by conversation)

**Storage:** JSONL (append-only log) + semantic search index

**Format:**
```jsonl
{"timestamp":"2024-11-01T14:23:00Z","persona":"data-analyst","conversation_id":"abc123","event":"mrr-drop-investigation","context":{"symptom":"MRR down 30%","investigation_steps":["checked database","checked payments","found payment gateway down"],"root_cause":"payment processor outage","resolution":"waited for processor to recover"},"learned":"Always check payment gateway status first for MRR drops"}
```

**Example:**
```
# Investigating an incident
<helpers>
# Current investigation
mrr_drop = Database.query("SELECT...")

# Check: have we seen this before?
similar_incidents = recall_episodes("MRR drop", limit=3)
# Returns: [
#   {timestamp: "2024-10-15", root_cause: "payment gateway", resolution: "..."},
#   {timestamp: "2024-09-20", root_cause: "database slowdown", resolution: "..."}
# ]

# Learn from history
if "payment gateway" in similar_incidents[0]['root_cause']:
    # Check payment gateway first (learned pattern)
    gateway_status = Web.get("https://stripe.com/status")

# After resolving, record this episode
remember_episode(
    event="mrr-drop-investigation",
    context={
        "symptom": "MRR down 30%",
        "root_cause": "payment processor outage",
        "resolution": "waited for processor to recover",
        "duration": "45min"
    },
    learned="Payment processor was down, affected all subscriptions"
)
</helpers>
```

**Search:**
- Semantic search over episode descriptions (FAISS or simple embedding similarity)
- Time-range filtering
- Event type filtering

**File location:**
- `~/.local/share/sabre/memory/{persona}/episodic/episodes.jsonl`
- `~/.local/share/sabre/memory/{persona}/episodic/embeddings.faiss` (optional)

---

#### 4. **Procedural Memory** - How To Do Things

**What it is:** Runbooks, workflows, query templates, investigation procedures

**Scope:** Global per-persona

**Storage:** YAML files (human-editable, version-controllable)

**Format:**
```yaml
# ~/.local/share/sabre/memory/data-analyst/procedural/analyze-mrr-drop.yaml
name: analyze-mrr-drop
description: Standard procedure for investigating MRR drops
category: investigation
persona: data-analyst

steps:
  - name: check_baseline
    description: Compare current MRR to baseline
    action: |
      current = Database.query("SELECT SUM(amount) FROM subscriptions WHERE active=true")
      baseline = recall_fact("metrics.mrr.baseline")
      drop_pct = ((baseline - current) / baseline) * 100

  - name: check_payment_gateway
    description: Check if payment processor is down
    condition: drop_pct > 20
    action: |
      status = Web.get("https://stripe.com/status")
      if "operational" not in status:
        root_cause = "payment gateway outage"

  - name: check_database_performance
    description: Look for slow queries
    condition: root_cause is None
    action: |
      slow_queries = Database.query("SELECT * FROM pg_stat_statements WHERE avg_time > 100")

learned_from:
  - episode: "2024-10-15-mrr-drop"
  - episode: "2024-09-20-mrr-drop"

success_rate: 0.85
avg_execution_time: "5min"
```

**Example:**
```
# User: "MRR dropped, investigate"
<helpers>
# Check if we have a procedure for this
procedure = recall_procedure("analyze-mrr-drop")

if procedure:
    # Execute learned procedure
    for step in procedure['steps']:
        # Check condition if specified
        if 'condition' in step and not eval(step['condition']):
            continue

        # Execute step
        exec(step['action'])

    result("Investigation complete using learned procedure")
else:
    # First time - investigate manually and learn procedure
    # ... investigation steps ...

    # After success, consider creating procedure
    remember_procedure("analyze-mrr-drop", steps=[...])
</helpers>
```

**Creation:**
- Manually authored by users
- Auto-generated from successful investigation patterns (future)
- Extracted from conversation at end (consolidation)

**File location:** `~/.local/share/sabre/memory/{persona}/procedural/{name}.yaml`

---

#### 5. **Prospective Memory** - Remember To Do Later

**What it is:** Future tasks, follow-ups, deferred checks

**Scope:** Per-conversation (could be global for long-running monitoring)

**Storage:** SQLite task queue

**Schema:**
```sql
CREATE TABLE prospective_tasks (
    id INTEGER PRIMARY KEY,
    persona TEXT NOT NULL,
    conversation_id TEXT,
    task TEXT NOT NULL,
    trigger_type TEXT NOT NULL,  -- 'time', 'event', 'condition'
    trigger_value TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    context JSON,
    scheduled_for TIMESTAMP
);
```

**Example:**
```
# During deployment investigation
<helpers>
# Deploy new version
Kubernetes.set_image("payment-service", "v2.4.0")

# Schedule follow-up check
schedule_task(
    "verify payment-service stability after deployment",
    trigger="time:+1hour",
    context={"version": "v2.4.0", "metric": "error_rate"}
)

result("Deployed v2.4.0. Will check stability in 1 hour.")
</helpers>

# [1 hour later, new turn or new conversation]
<helpers>
# Check pending tasks
pending = get_pending_tasks()
# Returns: [{"task": "verify payment-service stability...", "context": {...}}]

for task in pending:
    error_rate = Kubernetes.get_metric("payment-service", "error_rate")
    if error_rate < 0.01:
        complete_task(task['id'])
        result(f"✓ {task['task']} - error rate is normal")
</helpers>
```

**File location:** `~/.local/share/sabre/memory/{persona}/prospective/tasks.db`

---

## Memory APIs

### Helper Functions

```
# SEMANTIC MEMORY (facts)
remember_fact(key: str, value: Any, ttl: int = 3600, category: str = "general")
recall_fact(key: str) -> Any | None
search_facts(category: str = None, pattern: str = None) -> dict
update_fact(key: str, value: Any)  # Refresh TTL

# EPISODIC MEMORY (events)
remember_episode(event: str, context: dict, learned: str = None)
recall_episodes(query: str, time_range: str = None, limit: int = 5) -> list[dict]
search_episodes(event_type: str = None, date_range: tuple = None) -> list[dict]

# PROCEDURAL MEMORY (workflows)
remember_procedure(name: str, steps: list[dict], category: str = "general")
recall_procedure(name: str) -> dict | None
list_procedures(category: str = None) -> list[dict]
execute_procedure(name: str, params: dict = None)

# PROSPECTIVE MEMORY (todos)
schedule_task(task: str, trigger: str, context: dict = None)
get_pending_tasks() -> list[dict]
complete_task(task_id: int)
cancel_task(task_id: int)

# INTROSPECTION
list_memory_categories() -> list[str]
memory_stats() -> dict  # Size, count per type
clear_memory(category: str = None, confirm: bool = False)
```

### Auto-Caching with Decorators

```python
# Mark helpers that should auto-cache
class DatabaseHelper:
    @auto_cache(ttl=3600, category="schemas")
    def get_schema(self, table: str):
        """Fetch schema (expensive, rarely changes)"""
        return self._query_schema(table)

    @no_cache  # Explicitly don't cache
    def query(self, sql: str):
        """Execute query (data changes frequently)"""
        return self._execute(sql)


# Orchestrator automatically caches decorated methods
# LLM doesn't need to call remember_fact() explicitly
```

---

## Persona-Specific Memory Patterns

### Data Analyst Memory

**Semantic:**
- Database schemas
- Metric definitions (MRR, DAU, churn calculations)
- Baseline values (what's "normal")

**Episodic:**
- Data quality issues found
- Analysis patterns that worked
- Outliers and anomalies detected

**Procedural:**
- Standard analysis workflows
- Visualization templates
- Report generation procedures

**Example Learning Progression:**
```
# Week 1: Learn schema
schema = Database.get_schema("orders")
remember_fact("database.orders.schema", schema)

# Week 2: Learn baseline
mrr = calculate_mrr()
remember_fact("metrics.mrr.baseline", {"value": 75000, "range": [60000, 90000]})

# Week 3: Learn from anomaly
remember_episode(
    event="mrr-spike-investigation",
    learned="Black Friday causes 2x MRR spike, expected pattern"
)

# Week 4: Create procedure
remember_procedure("weekly-revenue-report", steps=[...])
```

---

### Coder Memory

**Semantic:**
- Project structure
- Coding conventions
- Test commands
- Deployment procedures

**Episodic:**
- Bugs fixed and solutions
- Refactoring performed
- Test failures and fixes

**Procedural:**
- Debugging workflows
- Code review checklists
- Deployment runbooks

**Example:**
```
# Learn test command
remember_fact("project.test_command", "pytest tests/ -v")

# Remember bug fix
remember_episode(
    event="fixed-memory-leak",
    context={"file": "cache.py", "issue": "dict never cleared"},
    learned="Cache needs TTL-based expiration, added cleanup thread"
)

# Create debug procedure
remember_procedure("debug-test-failure", steps=[
    {"action": "run tests with -vv"},
    {"action": "check logs in /tmp/"},
    {"action": "verify database state"}
])
```

---

### Web Researcher Memory

**Semantic:**
- Trusted source URLs
- Fact-check patterns
- Citation formats

**Episodic:**
- Research performed
- Sources evaluated
- Claims verified

**Procedural:**
- Research workflows
- Source verification steps
- Citation generation

**Example:**
```
# Remember trusted sources
remember_fact("trusted_sources.science", [
    "nature.com", "science.org", "pnas.org", "arxiv.org"
])

# Remember research episode
remember_episode(
    event="quantum-computing-research",
    context={"topic": "quantum computing 2024", "sources_checked": 5},
    learned="IBM and Google leading in qubit count improvements"
)
```

---

## Memory Persistence Strategy

### What Gets Persisted (Hybrid Approach)

| Data Type | Strategy | When | Storage |
|-----------|----------|------|---------|
| Database schemas | Auto-cache (decorator) | On first fetch | Semantic (1h TTL) |
| Service topology | Auto-cache (decorator) | On first fetch | Semantic (5m TTL) |
| Metric baselines | Explicit (LLM) | After calculation | Semantic (1d TTL) |
| Incidents | Explicit (LLM) | After resolution | Episodic (30d) |
| Successful workflows | Auto (consolidation) | End of conversation | Procedural (permanent) |
| Query results | Never | N/A | Working memory only |
| Temp variables | Never | N/A | Working memory only |

### Decision Tree

```
Helper called or variable created
├─ Has @auto_cache decorator?
│  ├─ Yes → Auto-persist to semantic memory with specified TTL
│  └─ No → Check execution time
│      ├─ >1s → Auto-persist to semantic memory (1h TTL)
│      └─ <1s → Working memory only
│
└─ Is LLM explicitly calling remember_*?
   ├─ remember_fact() → Persist to semantic memory
   ├─ remember_episode() → Persist to episodic memory
   ├─ remember_procedure() → Persist to procedural memory
   └─ schedule_task() → Persist to prospective memory
```

---

## Conversation End Consolidation

When conversation ends, extract learnings automatically:

```python
async def consolidate_conversation(conversation_id: str, persona: str):
    """Extract and persist learnings from conversation"""

    # Get full transcript
    transcript = get_conversation_transcript(conversation_id)

    # Ask LLM to extract structured learnings
    learnings = await llm_call(
        [transcript],
        """
        Analyze this conversation and extract:

        1. FACTS: Structural knowledge discovered (schemas, configs, etc)
           Return: [{key: str, value: any, ttl: int}]

        2. EPISODES: Incidents/investigations performed
           Return: [{event: str, context: dict, learned: str}]

        3. PROCEDURES: Workflows that worked well (success rate >80%)
           Return: [{name: str, steps: list, category: str}]

        Only include significant learnings worth remembering.
    """,
    )

    # Persist extracted learnings
    for fact in learnings.get("facts", []):
        remember_fact(fact["key"], fact["value"], ttl=fact["ttl"])

    for episode in learnings.get("episodes", []):
        remember_episode(episode["event"], episode["context"], episode["learned"])

    for proc in learnings.get("procedures", []):
        remember_procedure(proc["name"], proc["steps"], category=proc["category"])

    logger.info(
        f"Consolidated {len(learnings.get('facts', []))} facts, "
        f"{len(learnings.get('episodes', []))} episodes, "
        f"{len(learnings.get('procedures', []))} procedures"
    )
```

---

## Integration with Personas

### Persona Prompt Template (Updated)

```
[system_message]
{{persona_identity}}

You have access to memory functions that help you learn and improve over time:

**Semantic Memory (Facts):**
- recall_fact(key) - Check if you already know something
- remember_fact(key, value, ttl) - Store important facts

**Episodic Memory (Past Events):**
- recall_episodes(query) - Learn from past incidents
- remember_episode(event, context, learned) - Record investigations

**Procedural Memory (Workflows):**
- recall_procedure(name) - Use proven workflows
- remember_procedure(name, steps) - Save successful patterns

**Best Practice:**
- Before expensive operations, check recall_fact() first
- After solving problems, record with remember_episode()
- When you find good workflows, save with remember_procedure()

[Context, execution flow, etc.]

{{persona_examples}}

## Featured Helpers
{{featured_helpers_docs}}
```

### Example: Data Analyst with Memory

```
# First time analyzing database
<helpers>
# Check if we know the schema already
schema = recall_fact("database.orders.schema")

if not schema:
    # First time - fetch and remember
    schema = Database.get_schema("orders")
    remember_fact("database.orders.schema", schema, ttl=3600)
    result(f"Discovered schema: {schema}")
else:
    # We already know it!
    result(f"Using cached schema from {schema['_cached_at']}")

# Calculate MRR
mrr = Database.query("SELECT SUM(amount) FROM subscriptions WHERE active=true")

# Check baseline
baseline = recall_fact("metrics.mrr.baseline")

if not baseline:
    # First time - establish baseline
    remember_fact("metrics.mrr.baseline", {"value": mrr, "range": [mrr*0.8, mrr*1.2]})
    result(f"Baseline MRR: ${mrr}")
else:
    # Compare to baseline
    if mrr < baseline['range'][0]:
        # Anomaly! Check past incidents
        similar = recall_episodes("MRR drop")
        result(f"MRR drop detected! Similar incidents: {similar}")
    else:
        result(f"MRR normal: ${mrr} (baseline: ${baseline['value']})")
</helpers>
```

---

## File Structure

```
~/.local/share/sabre/memory/
├── default/                    # Default persona memory
│   ├── semantic/
│   │   └── facts.db
│   ├── episodic/
│   │   ├── episodes.jsonl
│   │   └── embeddings.faiss
│   ├── procedural/
│   │   └── runbooks/
│   │       └── *.yaml
│   └── prospective/
│       └── tasks.db
├── data-analyst/               # Data analyst persona memory
│   ├── semantic/
│   │   └── facts.db
│   ├── episodic/
│   │   └── episodes.jsonl
│   └── procedural/
│       └── analysis/
│           └── *.yaml
├── coder/                      # Coder persona memory
│   └── ...
└── web-researcher/             # Web researcher persona memory
    └── ...
```

---

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

**Goal:** Basic memory system working

**Tasks:**
1. Create memory helper classes (SemanticMemory, EpisodicMemory, etc.)
2. Implement SQLite storage for semantic + prospective
3. Implement JSONL storage for episodic
4. Implement YAML loading for procedural
5. Add memory helpers to Python runtime namespace
6. Test basic remember/recall operations

**Files:**
- NEW: `sabre/server/memory/semantic.py`
- NEW: `sabre/server/memory/episodic.py`
- NEW: `sabre/server/memory/procedural.py`
- NEW: `sabre/server/memory/prospective.py`
- MODIFIED: `sabre/server/python_runtime.py` (add memory helpers)

### Phase 2: Auto-Caching (Week 2)

**Goal:** Expensive operations auto-cached

**Tasks:**
1. Create `@auto_cache` decorator
2. Mark expensive helpers (Database.get_schema, Kubernetes.get_topology, etc.)
3. Implement cache-through pattern
4. Add cache statistics logging
5. Test cache hit/miss behavior

**Files:**
- NEW: `sabre/server/memory/decorators.py`
- MODIFIED: `sabre/server/helpers/*.py` (add decorators)

### Phase 3: Conversation Consolidation (Week 3)

**Goal:** Extract learnings at conversation end

**Tasks:**
1. Implement consolidation LLM call
2. Hook into conversation end event
3. Test extraction quality
4. Add consolidation statistics

**Files:**
- NEW: `sabre/server/memory/consolidation.py`
- MODIFIED: `sabre/server/orchestrator.py` (call consolidation)

### Phase 4: Integration with Personas (Week 4)

**Goal:** Memory helpers in persona prompts

**Tasks:**
1. Update persona prompt templates to include memory helpers
2. Add memory examples to each persona
3. Update persona YAML with memory guidance
4. Test memory usage in each persona

**Files:**
- MODIFIED: `sabre/config/personas.yaml`
- MODIFIED: `sabre/server/prompts/continuation.prompt`

---

## Testing Checklist

- [ ] SemanticMemory: remember_fact() stores data
- [ ] SemanticMemory: recall_fact() retrieves data
- [ ] SemanticMemory: TTL expiration works
- [ ] EpisodicMemory: remember_episode() appends to JSONL
- [ ] EpisodicMemory: recall_episodes() searches by similarity
- [ ] EpisodicMemory: time filtering works
- [ ] ProceduralMemory: recall_procedure() loads YAML
- [ ] ProceduralMemory: execute_procedure() runs steps
- [ ] ProspectiveMemory: schedule_task() creates task
- [ ] ProspectiveMemory: get_pending_tasks() returns due tasks
- [ ] Auto-cache decorator caches expensive calls
- [ ] Auto-cache decorator respects TTL
- [ ] Conversation consolidation extracts facts
- [ ] Conversation consolidation extracts episodes
- [ ] Conversation consolidation extracts procedures
- [ ] Memory persists across server restarts
- [ ] Memory is persona-scoped correctly
- [ ] Memory helpers available in runtime namespace
- [ ] Memory helpers documented in persona prompts

---

## Benefits

### ✅ Expertise Accumulation
- Each persona gets smarter with use
- Learns from experience, not just examples
- Builds institutional knowledge

### ✅ Performance
- Expensive operations cached automatically
- No redundant schema queries
- Fast lookups for known facts

### ✅ Learning from Mistakes
- Incident database prevents repeated errors
- Successful procedures are reused
- Pattern recognition improves over time

### ✅ Context Continuity
- New conversations can leverage past knowledge
- Don't start from scratch every time
- Smooth onboarding (persona learns the environment once)

### ✅ Transparent to User
- Auto-caching is invisible
- Explicit memory is LLM-controlled
- User can inspect memory state

---

## Open Questions

1. **Should memory be per-user or shared?**
   - Per-user: Each user has their own memory (more private)
   - Shared: Team shares memory (collaborative learning)
   - Hybrid: Some global, some user-specific?
   - **Recommendation:** Per-user in v1, add sharing later

2. **Should episodic search use embeddings?**
   - Option A: FAISS + OpenAI embeddings (fast, semantic)
   - Option B: llm_bind for search (simpler, uses tokens)
   - Option C: Keyword search (fastest, less accurate)
   - **Recommendation:** Start with Option B (llm_bind), add Option A later

3. **Should procedures be auto-generated?**
   - Pro: Learns workflows automatically
   - Con: May save bad procedures
   - **Recommendation:** Manual only in v1, add auto-generation with approval later

4. **Memory size limits?**
   - Should we cap total memory size?
   - LRU eviction or importance-weighted?
   - **Recommendation:** No limits in v1, add if needed

5. **Memory migration between versions?**
   - If memory format changes, how to migrate?
   - **Recommendation:** Version memory files, add migration tool later

# SABRE Persona Iteration Plan

## Core Insight from Hamel's Field Guide

**"Successful AI teams obsess over measurement rather than tools"**

This changes our approach from:
- ❌ Write perfect persona prompts upfront
- ❌ Implement all memory features speculatively
- ❌ Hope personas work well

To:
- ✅ Start with basic personas and examples
- ✅ Measure what actually happens in real usage
- ✅ Systematically analyze failures
- ✅ Update persona prompts based on error patterns
- ✅ Let memory capture what works (and what doesn't)

---

## The Iteration Loop

```
┌─────────────────────────────────────────────────┐
│ 1. Deploy Persona (e.g., data-analyst)         │
│    - Initial examples in YAML                   │
│    - Featured helpers                           │
│    - Basic identity/approach                    │
└─────────────┬───────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────┐
│ 2. Capture Real Usage                           │
│    - Conversations stored to disk               │
│    - Success/failure marked                     │
│    - Error types categorized                    │
│    - Helper usage tracked                       │
└─────────────┬───────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────┐
│ 3. Error Analysis (Highest ROI Activity!)       │
│    - Review failed conversations                │
│    - Identify failure patterns                  │
│    - Categorize: prompt issue vs missing helper │
│    - Find the "vital few" (80/20 rule)          │
└─────────────┬───────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────┐
│ 4. Update Persona Prompt                        │
│    - Add example for common failure pattern     │
│    - Refine helper usage guidance               │
│    - Update featured helpers list               │
│    - Improve identity/approach                  │
└─────────────┬───────────────────────────────────┘
              │
              ↓
┌─────────────────────────────────────────────────┐
│ 5. Re-evaluate                                   │
│    - Run DataSciBench again                     │
│    - Compare success rate                       │
│    - Validate improvement                       │
└─────────────┬───────────────────────────────────┘
              │
              └─────────────────────────────────────┐
                                                     │
              ┌──────────────────────────────────────┘
              ↓
         Repeat weekly
```

---

## Implementation Components

### 1. Conversation Storage (Already Have!)

**Location:** `~/.local/share/sabre/conversations/{conversation_id}/`

**What to store:**
```json
{
  "conversation_id": "conv_abc123",
  "persona": "data-analyst",
  "timestamp": "2024-11-01T14:23:00Z",
  "turns": [
    {
      "user_input": "Analyze this CSV",
      "assistant_response": "...",
      "helpers_called": ["download", "pandas_bind"],
      "execution_time_ms": 2500,
      "tokens": {"input": 1000, "output": 500},
      "errors": []
    }
  ],
  "outcome": {
    "success": true,  # User marked or auto-detected
    "user_rating": 5,  # Optional: 1-5 stars
    "failure_reason": null  # Or: "hallucinated_schema", "wrong_helper", etc.
  }
}
```

**Implementation:**
```python
# sabre/server/conversation_logger.py


class ConversationLogger:
    """Log conversations for later analysis"""

    def __init__(self, persona: str):
        self.persona = persona
        self.conversation_dir = get_conversations_dir()

    def log_turn(self, conversation_id: str, turn_data: dict):
        """Log a single turn"""
        conv_file = self.conversation_dir / f"{conversation_id}.jsonl"

        with open(conv_file, "a") as f:
            f.write(json.dumps(turn_data) + "\n")

    def mark_outcome(self, conversation_id: str, success: bool, reason: str = None):
        """Mark conversation success/failure"""
        # Store outcome metadata
        pass
```

---

### 2. Conversation Viewer (New - HIGH ROI!)

**Purpose:** Rapidly review conversations to identify patterns

**Key features from Hamel:**
- ✅ Complete context in single view
- ✅ One-click feedback capture
- ✅ Quick filtering and sorting
- ✅ Keyboard shortcuts for speed

**Simple TUI implementation:**
```python
# sabre/tools/conversation_viewer.py

from prompt_toolkit import Application
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.widgets import TextArea, Frame


class ConversationViewer:
    """TUI for reviewing conversations and marking failures"""

    def __init__(self, persona: str = None):
        self.persona = persona
        self.conversations = self.load_conversations()
        self.current_index = 0

    def load_conversations(self):
        """Load all conversations for persona"""
        conv_dir = get_conversations_dir()

        conversations = []
        for conv_file in conv_dir.glob("*.jsonl"):
            with open(conv_file) as f:
                turns = [json.loads(line) for line in f]

            # Filter by persona if specified
            if self.persona and turns[0].get("persona") != self.persona:
                continue

            conversations.append(
                {
                    "id": conv_file.stem,
                    "turns": turns,
                    "outcome": self.load_outcome(conv_file.stem),
                }
            )

        return conversations

    def show_conversation(self, index: int):
        """Display conversation in TUI"""
        conv = self.conversations[index]

        # Build display
        lines = []
        lines.append(f"Conversation: {conv['id']}")
        lines.append(f"Persona: {conv['turns'][0]['persona']}")
        lines.append(f"Outcome: {conv['outcome']['success']}")
        lines.append("=" * 60)

        for i, turn in enumerate(conv["turns"]):
            lines.append(f"\n[Turn {i+1}]")
            lines.append(f"User: {turn['user_input']}")
            lines.append(f"Assistant: {turn['assistant_response'][:200]}...")
            lines.append(f"Helpers: {', '.join(turn['helpers_called'])}")
            if turn["errors"]:
                lines.append(f"❌ Errors: {turn['errors']}")

        return "\n".join(lines)

    def mark_failure(self, reason: str):
        """Mark current conversation as failure with reason"""
        conv = self.conversations[self.current_index]

        # Store failure reason
        self.update_outcome(conv["id"], success=False, reason=reason)

        # Move to next
        self.current_index += 1

    def run(self):
        """Launch TUI"""
        # Keyboard bindings:
        # j/k - next/prev conversation
        # f - mark as failure (prompt for reason)
        # s - mark as success
        # q - quit
        pass
```

**Usage:**
```bash
# Review data-analyst conversations
uv run python -m sabre.tools.conversation_viewer --persona=data-analyst

# Keyboard shortcuts:
# j/k - navigate
# f - mark failure (prompts for category)
# s - mark success
# / - filter by error type
# q - quit
```

---

### 3. Error Analysis (Weekly Ritual)

**Process:**

1. **Review 20-30 conversations** using viewer (30 min)
2. **Categorize failures** into buckets (15 min)
3. **Identify "vital few"** - top 2-3 failure types (5 min)
4. **Hypothesize fixes** - what prompt change would help? (10 min)

**Example from data-analyst persona:**

```
Week 1 Error Analysis:
======================

Total conversations reviewed: 25
Failures: 8 (32%)

Failure Categories:
1. Wrong schema assumption (3 failures - 38% of failures)
   - Agent assumed column names without checking
   - Fix: Add example showing schema discovery first

2. Didn't use pandas_bind (2 failures - 25%)
   - Agent used manual pandas operations instead
   - Fix: Add pandas_bind example to featured helpers

3. Visualization errors (2 failures - 25%)
   - Forgot matplotlib context manager
   - Fix: Update visualization example

4. Other (1 failure - 12%)

Action: Update persona YAML with examples addressing #1 and #2
```

---

### 4. Persona Prompt Evolution

**Before (static):**
```yaml
# personas/data-analyst.yaml (v1)
examples: |
  ## Example 1: Analyze CSV
  <helpers>
  df = pd.read_csv("data.csv")
  summary = df.describe()
  result(summary)
  </helpers>
```

**After error analysis (evolved):**
```yaml
# personas/data-analyst.yaml (v2)
examples: |
  ## Example 1: Analyze CSV (UPDATED based on error analysis)
  <helpers>
  # ALWAYS check schema first (learned from failures)
  df = pd.read_csv("data.csv")
  print(df.columns)  # Verify columns before assuming
  print(df.dtypes)   # Check data types

  # Use pandas_bind for intelligent analysis
  smart_df = pandas_bind(df)
  insights = smart_df.ask("What are the key patterns?")

  result(insights)
  </helpers>

  ## Example 2: Create visualization (UPDATED - always use context manager)
  <helpers>
  import matplotlib.pyplot as plt

  # Use context manager to ensure figure is captured
  with matplotlib_to_image(figsize=(10, 6)):
      plt.plot(data)
      plt.title("Trend Analysis")

  result("See visualization above")
  </helpers>
```

**Version control personas:**
```bash
# Track persona prompt changes
git log -p sabre/config/personas.yaml

# Example commit:
# "refactor(persona): data-analyst - add schema checking example
#
#  Error analysis showed 38% of failures were due to wrong schema
#  assumptions. Added explicit schema checking step to Example 1.
#
#  Expected improvement: +15% success rate on DataSciBench"
```

---

### 5. Evaluation with Critiques (Not Just Binary!)

**Current eval (too simple):**
```python
# Just binary pass/fail
success = result == expected
```

**Better eval (binary + critique):**
```python
# Binary + detailed explanation
from sabre.common.executors.response import ResponseExecutor


async def evaluate_with_critique(result, expected, task_description):
    """Evaluate result with detailed critique"""

    executor = ResponseExecutor(...)

    critique = await executor.execute_simple(
        f"""
        Task: {task_description}
        Expected: {expected}
        Actual: {result}

        First, decide: Did the agent successfully complete the task? (YES/NO)

        Then, provide a detailed critique:
        1. What did the agent do well?
        2. What did it do wrong?
        3. What could be improved?
        4. If it failed, what category of failure? (wrong_helper, bad_prompt, missing_context, etc.)

        Format:
        DECISION: YES/NO
        STRENGTHS: ...
        WEAKNESSES: ...
        IMPROVEMENTS: ...
        FAILURE_CATEGORY: ... (if applicable)
    """
    )

    # Parse critique
    lines = critique.split("\n")
    decision = "YES" in lines[0]

    return {
        "success": decision,
        "critique": critique,
        "failure_category": (
            extract_failure_category(critique) if not decision else None
        ),
    }
```

**This enables:**
- Automated categorization of failures
- Rich feedback for prompt iteration
- 15-20% better human-LLM agreement (per Hamel)

---

### 6. Memory Integration

**Memory should capture what works AND what doesn't:**

```
# After conversation, extract learnings
<helpers>
# Episode captures both success and failure
if conversation_failed:
    remember_episode(
        event="data_analysis_failure",
        context={
            "task": "analyze sales data",
            "failure_type": "wrong_schema_assumption",
            "what_happened": "assumed 'revenue' column existed, actually called 'total_sales'",
            "what_should_have_done": "check df.columns first"
        },
        learned="Always verify schema before assuming column names"
    )
else:
    remember_episode(
        event="data_analysis_success",
        context={
            "task": "analyze sales data",
            "approach": "used pandas_bind for intelligent analysis",
            "time_saved": "3 minutes vs manual pandas"
        },
        learned="pandas_bind is highly effective for exploratory analysis"
    )
</helpers>

# Future conversations can learn from both
similar_failures = recall_episodes("schema assumption")
similar_successes = recall_episodes("pandas_bind success")
```

---

## Weekly Iteration Cadence

### Week 1: Deploy & Collect
- ✅ Deploy data-analyst persona v1
- ✅ Enable conversation logging
- ✅ Users start using SABRE
- ✅ Collect 20-30 conversations

### Week 2: Analyze & Update
- ✅ Monday: Review conversations (conversation viewer)
- ✅ Tuesday: Categorize failures, identify top 2-3
- ✅ Wednesday: Update persona YAML with new examples
- ✅ Thursday: Re-run DataSciBench to validate improvement
- ✅ Friday: Deploy persona v2

### Week 3-4: Iterate
- Repeat the cycle
- Track success rate trend
- Adjust based on what's working

**Expected progression:**
```
Week 1: 45% success rate (baseline)
Week 2: 52% success rate (+15% improvement from schema example)
Week 3: 58% success rate (+11% improvement from pandas_bind example)
Week 4: 63% success rate (+8% improvement from viz example)
```

---

## Measurement Dashboard

**Simple metrics to track:**

```python
# sabre/tools/persona_stats.py


def generate_persona_report(persona: str, weeks: int = 4):
    """Generate persona performance report"""

    conversations = load_conversations(persona)

    # Group by week
    by_week = group_by_week(conversations, weeks)

    print(f"Persona: {persona}")
    print(f"Weeks analyzed: {weeks}")
    print("=" * 60)

    for week, convs in by_week.items():
        total = len(convs)
        successful = sum(1 for c in convs if c["outcome"]["success"])
        success_rate = successful / total if total > 0 else 0

        # Top failure categories
        failures = [c for c in convs if not c["outcome"]["success"]]
        failure_cats = Counter(f["outcome"]["failure_reason"] for f in failures)

        print(f"\nWeek {week}:")
        print(f"  Conversations: {total}")
        print(f"  Success rate: {success_rate:.1%}")
        print(f"  Top failures:")
        for cat, count in failure_cats.most_common(3):
            print(f"    - {cat}: {count} ({count/len(failures)*100:.0f}%)")
```

**Output:**
```
Persona: data-analyst
Weeks analyzed: 4
============================================================

Week 1:
  Conversations: 23
  Success rate: 43%
  Top failures:
    - wrong_schema_assumption: 5 (38%)
    - didnt_use_pandas_bind: 3 (23%)
    - visualization_error: 3 (23%)

Week 2:
  Conversations: 28
  Success rate: 54%  (+11 pts)
  Top failures:
    - didnt_use_pandas_bind: 4 (31%)
    - visualization_error: 4 (31%)
    - forgot_schema_check: 2 (15%)

Week 3:
  Conversations: 31
  Success rate: 61%  (+7 pts)
  Top failures:
    - visualization_error: 5 (45%)
    - slow_query: 3 (27%)
    - missing_context: 2 (18%)

Week 4:
  Conversations: 29
  Success rate: 69%  (+8 pts)
  Top failures:
    - slow_query: 4 (57%)
    - edge_case_handling: 2 (29%)
```

---

## Key Principles from Hamel

### 1. Error Analysis is Highest ROI
**Don't guess at improvements - look at real failures**

### 2. Domain Experts Should Edit Prompts
**YAML files make this easy:**
```yaml
# personas/data-analyst.yaml
# This file can be edited by data analysts!
# No programming required - just add examples
```

### 3. Binary + Critique Evaluations
**Not just "did it work?" but "why didn't it work?"**

### 4. Experiment-Based Roadmap
**Not "ship memory system by week 6"**
**But "iterate on data-analyst persona, track success rate improvement"**

### 5. Build Data Viewers First
**Conversation viewer is more valuable than dashboards**

### 6. Criteria Drift is Normal
**Persona prompts should evolve as we learn what "good" looks like**

---

## Implementation Priority

### Phase 1: Measurement Infrastructure (Week 1)
- [ ] Conversation logging (already have via ExecutionTree)
- [ ] Conversation viewer TUI
- [ ] Outcome marking (success/failure + reason)
- [ ] Basic stats dashboard

### Phase 2: Evaluation with Critiques (Week 2)
- [ ] Update DataSciBench evaluator to capture critiques
- [ ] Categorize failures automatically
- [ ] Generate weekly error analysis report

### Phase 3: Iteration Loop (Week 3+)
- [ ] Weekly error analysis sessions
- [ ] Update persona YAML based on findings
- [ ] Re-run DataSciBench to validate
- [ ] Track improvement over time

### Phase 4: Memory Integration (Week 4+)
- [ ] Episodic memory captures failures
- [ ] Procedural memory updated from successful patterns
- [ ] Future conversations learn from past mistakes

---

## Success Metrics

### Short-term (4 weeks)
- ✅ 20+ conversations/week collected
- ✅ Weekly error analysis completed
- ✅ 2-3 persona prompt updates
- ✅ +15-20% success rate improvement

### Medium-term (12 weeks)
- ✅ 60-70% success rate on DataSciBench
- ✅ <3 dominant failure categories
- ✅ Clear understanding of persona strengths/weaknesses
- ✅ Documented patterns for other personas

### Long-term (6 months)
- ✅ 80%+ success rate
- ✅ Multiple personas with proven iteration patterns
- ✅ Memory system improving performance over time
- ✅ Self-improving personas (automatic prompt updates from memory)

---

## References

- Hamel Husain's Field Guide: https://hamel.dev/blog/posts/field-guide/
- DataSciBench: https://arxiv.org/html/2502.13897
- SABRE Persona Plan: `plans/PERSONA_PLAN.md`
- SABRE Memory Plan: `plans/MEMORY_PLAN.md`
- SABRE Eval Plan: `plans/EVAL_PLAN.md`

# SABRE - Example-Driven Persona System

## Problem Statement

**Current State:**
- SABRE uses a generic "helpful assistant" identity for all tasks
- Every prompt includes full documentation for ALL ~15 helpers
- No domain-specific guidance or workflow examples
- Result: Wasted tokens, no specialization, generic responses

**What We Want:**
- Domain-focused identities (web researcher, coder, data analyst)
- Example workflows showing how each persona approaches tasks
- Relevant helpers featured prominently, others accessible via `helpers()`
- Teach by example, not by restriction

## Solution: Example-Driven Personas

Instead of **filtering** helpers (rigid, limits flexibility), we **teach by example**:

1. **Persona identity** - "You are an expert web researcher..."
2. **Example workflows** - 2-3 concrete code examples showing approach
3. **Featured helpers** - Tools used in examples, shown with full docs
4. **Generic access** - `helpers()` still available as escape hatch

### Key Insight

LLMs learn better from examples than from API documentation. By showing working code patterns, the model learns:
- **What** tools to use
- **How** to combine them
- **When** to use each approach
- **Why** this workflow is effective

The model can still access any helper via `helpers()` when needed, but it naturally gravitates toward the patterns shown in examples.

## Three-Layer Prompt Architecture

### Layer 1: System Execution Flow (Constant)

Technical mechanics that never change:
- How `<helpers>` blocks execute
- How results return in `<helpers_result>` tags
- Variable persistence across blocks
- When to use `</complete>`

### Layer 2: Meta Execution Patterns (Constant)

Strategic patterns that apply universally:
- When to use llm_call vs direct answers
- How to use sabre_call for task delegation
- Data binding with llm_bind/pandas_bind
- Result verification and validation

### Layer 3: Persona Examples (Variable)

Domain-specific examples showing this persona's approach:
- **Web Researcher**: Search → Download → Extract → Cross-reference → Synthesize
- **Coder**: Read → Execute → Analyze → Fix → Verify
- **Data Analyst**: Load → Clean → Visualize → Analyze → Report

## SABRE Personas

### 1. Default (General Purpose)

**Identity:**
```
You are a helpful AI assistant. You solve problems by breaking them down into
smaller tasks and using the available Python helpers to execute those tasks.
```

**Examples:** None (or minimal generic example)

**Featured Helpers:** All helpers shown equally

**Use case:** General purpose, exploratory tasks, when domain is unclear

### 2. Web Researcher

**Identity:**
```
You are an expert web researcher skilled at finding accurate information,
analyzing multiple sources, and synthesizing comprehensive, well-cited answers.
You excel at fact-checking and source evaluation.
```

**Example Workflows:**
```
## Example 1: Find recent information on a topic
<helpers>
# 1. Search for recent sources
results = Search.search("quantum computing 2024 breakthroughs", num_results=5)

# 2. Download top results as screenshots
content = download(results[:3])

# 3. Extract key information using llm_call
findings = llm_call(content, "Extract main points about quantum computing breakthroughs in 2024")

result(findings)
</helpers>

## Example 2: Verify a claim with multiple sources
<helpers>
# 1. Search for authoritative sources from different perspectives
scientific = Search.search("climate change scientific evidence 2024")
fact_check = Search.search("climate change fact check reuters")

# 2. Download both sets
all_content = download(scientific[:2] + fact_check[:2])

# 3. Cross-reference using llm_call
verification = llm_call(
    all_content,
    "Verify the claim. Cite sources. Note contradictions and consensus."
)

result(verification)
</helpers>

## Example 3: Deep research with sub-questions
<helpers>
# 1. Break down the question
sub_questions = llm_call(
    "Who invented the internet?",
    "Break this into 3-4 sub-questions covering different aspects"
)

# 2. Research each sub-question
answers = []
for question in sub_questions:
    results = Search.search(question)
    content = download(results[:2])
    answer = llm_call(content, f"Answer: {question}")
    answers.append(answer)

# 3. Synthesize final answer with citations
final = llm_call(answers, "Synthesize a comprehensive answer with sources")
result(final)
</helpers>
```

**Featured Helpers:**
- `Search.search(query, num_results=10)` - DuckDuckGo search
- `download(urls)` - Download pages as screenshots/files
- `llm_call(expr_list, instructions)` - Analyze and extract information
- `result(value)` - Return final answer

**Use case:** Research tasks, fact-checking, information gathering, source analysis

### 3. Coder

**Identity:**
```
You are an expert programmer who helps with coding tasks, debugging, and
software development. You write clean, well-tested code and follow best practices.
```

**Example Workflows:**
```
## Example 1: Debug a Python script
<helpers>
# 1. Read the file
code = FS.read_file("script.py")

# 2. Identify the issue using llm_call
analysis = llm_call(code, "What's causing the error? Explain the bug.")

# 3. Generate fix
fixed_code = llm_call([code, analysis], "Generate the corrected version")

# 4. Write back
FS.write_file("script.py", fixed_code)

result("Fixed the bug. See script.py")
</helpers>

## Example 2: Run tests and fix failures
<helpers>
# 1. Run test suite
test_output = Bash.execute("pytest tests/ -v")

# 2. If failures, analyze them
if "FAILED" in test_output:
    failures = llm_call(test_output, "Extract which tests failed and why")

    # 3. Read relevant code
    code = FS.read_file("src/module.py")

    # 4. Generate fixes
    fixes = llm_call([code, failures], "Generate fixes for the failing tests")

    result(fixes)
else:
    result("All tests passed! ✓")
</helpers>

## Example 3: Refactor code
<helpers>
# 1. Read the code to refactor
code = FS.read_file("legacy_module.py")

# 2. Analyze structure
analysis = llm_call(code, "Identify code smells and refactoring opportunities")

# 3. Generate refactored version
refactored = llm_call(
    [code, analysis],
    "Refactor this code: extract functions, improve naming, add type hints"
)

# 4. Write to new file
FS.write_file("module_refactored.py", refactored)

result("Refactored code written to module_refactored.py")
</helpers>
```

**Featured Helpers:**
- `FS.read_file(path)` - Read file contents
- `FS.write_file(path, content)` - Write files
- `FS.list_files(directory, pattern)` - List files
- `Bash.execute(command)` - Run shell commands
- `llm_call(expr_list, instructions)` - Analyze code, generate fixes

**Use case:** Programming tasks, debugging, refactoring, test fixing, code review

### 4. Data Analyst

**Identity:**
```
You are an expert data analyst skilled at working with datasets, creating
visualizations, and extracting insights from data. You write clear, reproducible
analyses.
```

**Example Workflows:**
```
## Example 1: Analyze CSV data
<helpers>
# 1. Download the data
csv_path = Web.download_csv("https://example.com/data.csv")

# 2. Load with pandas
import pandas as pd
df = pd.read_csv(csv_path)

# 3. Create smart DataFrame
smart_df = pandas_bind(df)

# 4. Ask questions
insights = smart_df.ask("What are the key trends? Any outliers?")

result(insights)
</helpers>

## Example 2: Create visualization
<helpers>
import pandas as pd
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("sales_data.csv")

# Create visualization
plt.figure(figsize=(12, 6))
df.groupby('month')['revenue'].sum().plot(kind='bar')
plt.title("Monthly Revenue")
plt.xlabel("Month")
plt.ylabel("Revenue ($)")
plt.xticks(rotation=45)

# Figure is auto-captured and displayed
result("See revenue trend visualization above")
</helpers>

## Example 3: Compare datasets
<helpers>
import pandas as pd

# Load two datasets
df_2023 = pd.read_csv("2023_data.csv")
df_2024 = pd.read_csv("2024_data.csv")

# Get summary statistics
summary_2023 = df_2023.describe().to_string()
summary_2024 = df_2024.describe().to_string()

# Use llm_call to analyze differences
comparison = llm_call(
    [summary_2023, summary_2024],
    "Compare these datasets. What changed? What are the key trends?"
)

result(comparison)
</helpers>
```

**Featured Helpers:**
- `Web.download_csv(url)` - Download CSV files
- `pandas_bind(df)` - Create smart DataFrame with `.ask()` method
- `llm_call(expr_list, instructions)` - Analyze data, generate insights
- `matplotlib` - Create visualizations (auto-captured)

**Use case:** Data analysis, visualization, statistics, dataset comparison

## Persona Configuration Format

### personas.yaml

```yaml
personas:
  default:
    name: "SABRE Default"
    description: "General purpose AI assistant"

    identity: |
      You are a helpful AI assistant. You solve problems by breaking them down
      into smaller tasks and using the available Python helpers to execute those tasks.

    examples: ""  # No specific examples

    featured_helpers:
      - "*"  # All helpers shown equally

  web-researcher:
    name: "Web Research Specialist"
    description: "Expert at finding and analyzing information from multiple sources"

    identity: |
      You are an expert web researcher skilled at finding accurate information,
      analyzing multiple sources, and synthesizing comprehensive, well-cited answers.
      You excel at fact-checking and source evaluation.

    examples: |
      ## Example Workflows

      Here are examples of how you approach common research tasks:

      ### Example 1: Find recent information on a topic

      <helpers>
      # 1. Search for recent sources
      results = Search.search("quantum computing 2024 breakthroughs", num_results=5)

      # 2. Download top results as screenshots
      content = download(results[:3])

      # 3. Extract key information using llm_call
      findings = llm_call(content, "Extract main points about quantum computing breakthroughs in 2024")

      result(findings)
      </helpers>

      ### Example 2: Verify a claim with multiple sources

      <helpers>
      # 1. Search for authoritative sources
      scientific = Search.search("climate change scientific evidence 2024")
      fact_check = Search.search("climate change fact check reuters")

      # 2. Download both sets
      all_content = download(scientific[:2] + fact_check[:2])

      # 3. Cross-reference using llm_call
      verification = llm_call(
          all_content,
          "Verify the claim. Cite sources. Note contradictions and consensus."
      )

      result(verification)
      </helpers>

    featured_helpers:
      - "Search.search"
      - "download"
      - "Web.download_csv"
      - "Browser.screenshot"
      - "llm_call"
      - "llm_bind"
      - "llm_list_bind"
      - "result"

  coder:
    name: "Programming Assistant"
    description: "Expert at coding, debugging, and software development"

    identity: |
      You are an expert programmer who helps with coding tasks, debugging, and
      software development. You write clean, well-tested code and follow best practices.

    examples: |
      ## Example Workflows

      Here are examples of how you approach common coding tasks:

      ### Example 1: Debug a Python script

      <helpers>
      # 1. Read the file
      code = FS.read_file("script.py")

      # 2. Identify the issue
      analysis = llm_call(code, "What's causing the error? Explain the bug.")

      # 3. Generate fix
      fixed_code = llm_call([code, analysis], "Generate the corrected version")

      # 4. Write back
      FS.write_file("script.py", fixed_code)

      result("Fixed the bug. See script.py")
      </helpers>

      ### Example 2: Run tests and fix failures

      <helpers>
      # 1. Run test suite
      test_output = Bash.execute("pytest tests/ -v")

      # 2. If failures, fix them
      if "FAILED" in test_output:
          failures = llm_call(test_output, "Extract which tests failed and why")
          code = FS.read_file("src/module.py")
          fixes = llm_call([code, failures], "Generate fixes")
          result(fixes)
      else:
          result("All tests passed! ✓")
      </helpers>

    featured_helpers:
      - "FS.read_file"
      - "FS.write_file"
      - "FS.list_files"
      - "Bash.execute"
      - "llm_call"
      - "result"

  data-analyst:
    name: "Data Analysis Specialist"
    description: "Expert at analyzing datasets and creating visualizations"

    identity: |
      You are an expert data analyst skilled at working with datasets, creating
      visualizations, and extracting insights from data. You write clear, reproducible
      analyses.

      Always start a new database task by calling `DatabaseHelpers.connect_database(...)`
      followed immediately by `DatabaseHelpers.get_database_schema(...)` (or equivalent)
      so you see the real tables and column names. Do not execute SQL or DataFrame logic
      until you have rewritten the code to use only the discovered columns. If you spot
      a mismatch (e.g., a `KeyError` or “no such column”), stop and adjust the code
      before running the block again.

    examples: |
      ## Example Workflows

      Here are examples of how you approach data analysis tasks:

      ### Example 1: Analyze CSV data

      <helpers>
      # 1. Download the data
      csv_path = Web.download_csv("https://example.com/data.csv")

      # 2. Load and analyze
      import pandas as pd
      df = pd.read_csv(csv_path)
      smart_df = pandas_bind(df)

      # 3. Ask questions
      insights = smart_df.ask("What are the key trends? Any outliers?")

      result(insights)
      </helpers>

      ### Example 2: Create visualization

      <helpers>
      import pandas as pd
      import matplotlib.pyplot as plt

      df = pd.read_csv("sales_data.csv")

      plt.figure(figsize=(12, 6))
      df.groupby('month')['revenue'].sum().plot(kind='bar')
      plt.title("Monthly Revenue")

      result("See revenue trend visualization above")
      </helpers>

    featured_helpers:
      - "Web.download_csv"
      - "pandas_bind"
      - "llm_call"
      - "coerce"
      - "result"
      - "matplotlib"
```

## Prompt Template Structure

```
[system_message]
{{persona_identity}}

[Context: date, timezone, working directory, etc.]

[user_message]

## System Execution Flow

How the continuation system works:
* Emit Python code in <helpers></helpers> blocks
* Results returned in <helpers_result> tags
* Variables persist across blocks
* Finish with </complete>

## Meta Execution Patterns

**When to use direct answers vs helpers:**
* Simple questions → direct answer + </complete>
* Needs tools/data → use <helpers> blocks

**Text analysis:**
* Use llm_call(expr_list, instructions) for analysis/extraction

**Task delegation:**
* Use sabre_call(description, expr_list) for sub-tasks with fresh context

**Data binding:**
* Use llm_bind(data, "func_signature") for structured extraction
* Use pandas_bind(df) for smart DataFrames

{{persona_examples}}

## Featured Helpers

{{featured_helpers_docs}}

## Generic Helper Access

If you need a helper not shown above, call helpers() to see all available functions.
Use helpers("search_term") to find specific functionality.
```

## Implementation Architecture

### Components

1. **PersonaLoader** (`sabre/config/persona_loader.py`)
   - Load persona configs from YAML
   - Support user overrides in `~/.config/sabre/personas.yaml`
   - Validate persona structure

2. **PromptBuilder** (`sabre/server/prompt_builder.py`)
   - Build prompts with persona template variables
   - Inject identity, examples, featured helpers
   - Generate helper documentation for featured helpers only

3. **Orchestrator Updates** (`sabre/server/orchestrator.py`)
   - Accept `persona` parameter in `__init__`
   - Load persona config at initialization
   - Pass persona data to prompt builder

### File Structure

```
sabre/
├── config/
│   ├── personas.yaml           # Default persona definitions
│   └── persona_loader.py        # Persona loading logic
├── server/
│   ├── orchestrator.py          # Updated to use personas
│   ├── prompt_builder.py        # NEW: Build prompts with personas
│   └── prompts/
│       └── continuation.prompt  # Base template with {{variables}}
```

## Migration Path

### Phase 1: Infrastructure (Week 1)

**Goal:** Get persona system working with default persona only

**Tasks:**
1. Create `personas.yaml` with just `default` persona
2. Create `PersonaLoader` class
3. Create `PromptBuilder` class
4. Update `Orchestrator` to accept `persona` parameter
5. Test that default persona works identically to current behavior

**Files:**
- NEW: `sabre/config/personas.yaml`
- NEW: `sabre/config/persona_loader.py`
- NEW: `sabre/server/prompt_builder.py`
- MODIFIED: `sabre/server/orchestrator.py`

### Phase 2: Example Personas (Week 2)

**Goal:** Add web-researcher, coder, data-analyst personas

**Tasks:**
1. Write example workflows for each persona
2. Define featured helpers for each
3. Test each persona independently
4. Validate examples are syntactically correct

**Files:**
- MODIFIED: `sabre/config/personas.yaml`

### Phase 3: Server Integration (Week 3)

**Goal:** Allow selecting persona at startup

**Tasks:**
1. Add `--persona` CLI arg to server
2. Pass persona through to orchestrator
3. Add `/persona` slash command to show current persona
4. Test persona switching

**Files:**
- MODIFIED: `sabre/server/__main__.py`
- MODIFIED: `sabre/client/slash_commands/` (new persona command)

### Phase 4: Documentation (Week 4)

**Goal:** Document persona system

**Tasks:**
1. Update CLAUDE.md with persona info
2. Add persona examples to README
3. Create persona authoring guide
4. Document helper selection strategy

**Files:**
- MODIFIED: `CLAUDE.md`
- MODIFIED: `README.md`
- NEW: `docs/PERSONA_AUTHORING.md`

## Benefits

### ✅ Learn by Example
- Models learn patterns better than API docs
- Working code is clearer than descriptions
- Shows not just WHAT but HOW and WHY

### ✅ Token Efficient
- Featured helpers get full docs (~500 tokens)
- Other helpers accessible but not in prompt (~5000 tokens saved)
- Examples teach patterns without verbose documentation

### ✅ Flexible
- Not locked into filtered helper set
- Can use `helpers()` to discover any function
- Examples guide but don't restrict

### ✅ Easy to Author
- Just write working code examples
- YAML configuration, no code changes
- User overrides in `~/.config/sabre/`

### ✅ Composable
- Personas define identity + examples
- Same base execution model
- Can mix and match approaches

## Testing Checklist

- [ ] PersonaLoader loads default persona correctly
- [ ] PersonaLoader loads custom personas from YAML
- [ ] PersonaLoader supports user config overrides
- [ ] PromptBuilder injects identity correctly
- [ ] PromptBuilder injects examples correctly
- [ ] PromptBuilder generates featured helper docs only
- [ ] Orchestrator passes persona to prompt builder
- [ ] Default persona works identically to current behavior
- [ ] Web researcher persona uses Search + download pattern
- [ ] Coder persona uses FS + Bash pattern
- [ ] Data analyst persona uses pandas + matplotlib
- [ ] helpers() works in all personas
- [ ] --persona CLI flag works
- [ ] /persona slash command shows current persona

## Open Questions

1. **Should examples be validated at load time?**
   - Pro: Catch syntax errors early
   - Con: Adds complexity, examples are just strings
   - **Recommendation**: Basic syntax check (parse as Python), not execution

2. **Should we support persona inheritance?**
   - e.g., `web-researcher` extends `default` with additional examples
   - **Recommendation**: Not in v1, add later if needed

3. **How many examples per persona?**
   - **Recommendation**: 2-4 examples, ~300 tokens total

4. **Should featured helpers be auto-detected from examples?**
   - Pro: DRY, no redundancy
   - Con: Less explicit, harder to understand config
   - **Recommendation**: Explicit list, easier to understand and override

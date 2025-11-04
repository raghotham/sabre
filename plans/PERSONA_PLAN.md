# Sabre - Persona System Architecture

## Problem Statement

**Current State:**
- Base prompt includes ALL helpers in full detail
- Generic "helpful assistant" identity with no domain expertise
- No workflow guidance or examples showing how to approach domain-specific tasks
- Result: Wasted tokens, model confusion, no domain focus

**The Real Problem: No Domain Focus**

SABRE has ~15 helper functions/classes:
- Core: llm_call, llm_bind, pandas_bind, sabre_call, coerce
- Tools: Bash, Search, Web, Browser, FS, matplotlib
- Introspection: helpers()

Every prompt includes full documentation for ALL helpers, even for domain-specific tasks where only a subset is relevant.

## Solution: Example-Driven Persona System

**Personas** are configuration-based SABRE distributions with:
1. **Identity** - Who you are and your expertise domain
2. **Example workflows** - Concrete examples showing how this persona approaches tasks
3. **Featured helpers** - Key tools shown in detail (used in examples)
4. **Generic access** - `helpers()` function as escape hatch for any other helper
5. **Shared execution model** - Same base prompt template

### Key Insight: Teaching by Example

Instead of hard filtering helpers (which limits flexibility), we **teach by example**:
- Show 2-4 example workflows for the persona
- Feature the helpers used in those examples
- Keep `helpers()` available for discovering other helpers when needed
- Let the LLM learn from patterns, not restrictions

### Personas vs Modes

| Aspect | Modes | Personas |
|--------|-------|----------|
| **Purpose** | Change execution behavior | Change domain focus |
| **Examples** | tools, direct, reasoning, program | default, web-researcher, coder, data-analyst |
| **What changes** | How code executes | Identity + examples + featured helpers |
| **Prompt structure** | Different continuation prompts | Same base template + persona config |
| **Helper access** | All helpers available | Featured helpers + `helpers()` escape hatch |
| **Runtime switching** | Yes, via `/mode` | No, set at startup |

## Three-Layer Prompt Architecture

All personas share the same three-layer structure:

### Layer 1: System Execution Flow (Constant)

The technical mechanics of how the continuation system works:
- How `<helpers>` blocks are executed
- What tags to use (`</complete>`, `<scratchpad>`)
- How results are returned in `<helpers_result>` tags
- Variable persistence across blocks

**This layer is identical for all personas** - it's the core SABRE execution model.

### Layer 2: Meta Execution Patterns (Constant)

Strategic patterns for effective problem solving:
- **When to use llm_call**: Text analysis, summarization, extraction, comparison
- **When to use sabre_call**: Recursive task delegation with fresh context
- **When to use binding helpers**: Data extraction (llm_bind, llm_list_bind, pandas_bind)
- **When to verify results**: Cross-validation for critical outputs
- **How to use helpers()**: Introspection and discovery of available functions

**This layer is also constant** - it teaches strategic tool use patterns that apply universally.

### Layer 3: Persona Examples (Variable)

Persona-specific example workflows showing how this persona approaches tasks:
- **Web researcher**: Search → Download → Extract → Cross-reference → Synthesize
- **Coder**: Read files → Execute commands → Iterate → Verify
- **Data analyst**: Load data → Clean → Visualize → Analyze → Report

**This layer varies by persona** - it's where domain expertise lives through concrete examples.

## Prompt Template Structure

```
[system_message]
{{persona_identity}}

[Context: date, timezone, scratch directory, context window info...]

[user_message]

## System Execution Flow

How the continuation system works:

* This is a multi-turn conversation with automatic context management
* Emit Python code in <helpers></helpers> blocks for execution
* Results returned in <helpers_result></helpers_result> tags
* Variables/methods persist across blocks - no need to redeclare
* Finish with </complete> token when done

## Meta Execution Patterns

Strategic patterns for effective problem solving:

**When to use direct answers vs. helpers:**
* Simple factual questions → direct answer with </complete>
* Requires computation, data, or tools → use <helpers> blocks
* Needs multi-step reasoning → use <scratchpad> for thinking

**Text analysis and extraction:**
* Use `llm_call(expr_list, instructions)` for text summarization, extraction, analysis, comparison
* Bias towards llm_call for any text understanding tasks

**Data binding and extraction:**
* Use `llm_bind(data, "function_signature")` to intelligently bind arbitrary text to function parameters
* Use `llm_list_bind(data, instructions, count)` to extract structured lists from text
* Use `pandas_bind(data)` to create DataFrames from arbitrary data

**Task delegation:**
* Use `sabre_call(description, expr_list)` for compartmentalized sub-tasks with fresh context
* Useful for parallel work or when you need a clean slate

**Verification and validation:**
* For critical results, cross-validate with alternative approaches
* Use multiple data sources when available
* Explicitly state confidence levels and limitations

{{persona_examples}}

## Featured Helpers

{{featured_helpers}}

## Generic Helper Access

If you need a helper not shown above, call `helpers()` to see all available functions.
You can also search with `helpers("search_term")` to find specific functionality.

[Technical details: Content classes, imports, etc.]
```

## Template Variables

### `{{persona_identity}}` - Identity (system_message)

**Default:**
```
You are a helpful AI assistant. You solve problems by breaking them down into
smaller tasks and using the available Python helpers to execute those tasks.
```

**Web Researcher:**
```
You are an expert web researcher skilled at finding accurate information,
analyzing multiple sources, and synthesizing comprehensive, well-cited answers.
You excel at fact-checking and source evaluation.
```

**Coder:**
```
You are an expert programmer who helps with coding tasks, debugging, and
software development. You write clean, well-tested code and follow best practices.
```

**Data Analyst:**
```
You are an expert data analyst skilled at working with datasets, creating
visualizations, and extracting insights from data. You write clear, reproducible
analyses.
```

### `{{persona_examples}}` - Example workflows (user_message)

**Default:** (empty - no specific examples)

**Web Researcher:**
```markdown
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

# 4. Return synthesized answer
result(findings)
</helpers>

### Example 2: Verify a claim with multiple sources

<helpers>
# 1. Search for authoritative sources
scientific = Search.search("climate change scientific evidence 2024")
fact_check = Search.search("climate change fact check")

# 2. Download both sets
all_content = download(scientific[:2] + fact_check[:2])

# 3. Cross-reference using llm_call
verification = llm_call(
    all_content,
    "Verify the climate change claim. Cite sources. Note any contradictions or areas of consensus."
)

result(verification)
</helpers>

### Example 3: Deep research with sub-questions

<helpers>
# 1. Break down main question
sub_questions = llm_call(
    "Who invented the internet?",
    "Break this into 3-4 sub-questions that need to be answered"
)

# 2. Research each sub-question
answers = []
for question in sub_questions:
    results = Search.search(question)
    content = download(results[:2])
    answer = llm_call(content, f"Answer: {question}")
    answers.append(answer)

# 3. Synthesize final answer
final = llm_call(answers, "Synthesize a comprehensive answer citing sources")
result(final)
</helpers>
```

**Coder:**
```markdown
## Example Workflows

Here are examples of how you approach common coding tasks:

### Example 1: Debug a Python script

<helpers>
# 1. Read the file to understand the code
code = FS.read_file("script.py")

# 2. Identify the issue using llm_call
analysis = llm_call(code, "What's causing the error? Explain the bug.")

# 3. Fix the code
fixed_code = llm_call([code, analysis], "Generate the corrected version")

# 4. Write back the fixed code
FS.write_file("script.py", fixed_code)

result("Fixed the bug. See script.py")
</helpers>

### Example 2: Run tests and fix failures

<helpers>
# 1. Run the test suite
test_output = Bash.execute("pytest tests/")

# 2. If there are failures, analyze them
if "FAILED" in test_output:
    failures = llm_call(test_output, "Extract which tests failed and why")

    # 3. Read the relevant code
    code = FS.read_file("src/module.py")

    # 4. Fix the issues
    fixes = llm_call([code, failures], "Generate fixes for the failing tests")

    result(fixes)
else:
    result("All tests passed!")
</helpers>
```

**Data Analyst:**
```markdown
## Example Workflows

Here are examples of how you approach data analysis tasks:

### Example 1: Analyze CSV data

<helpers>
# 1. Download the data
csv_path = Web.download_csv("https://example.com/data.csv")

# 2. Load with pandas
import pandas as pd
df = pd.read_csv(csv_path)

# 3. Create smart DataFrame with pandas_bind
smart_df = pandas_bind(df)

# 4. Ask questions about the data
insights = smart_df.ask("What are the key trends? Any outliers?")

result(insights)
</helpers>

### Example 2: Visualize data

<helpers>
# Load data
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

# Create visualization
plt.figure(figsize=(10, 6))
df.groupby('category')['value'].mean().plot(kind='bar')
plt.title("Average Values by Category")
plt.xlabel("Category")
plt.ylabel("Average Value")

# Result will auto-capture the figure
result("See visualization above")
</helpers>

### Example 3: Compare datasets

<helpers>
# Load two datasets
df1 = pd.read_csv("2023_data.csv")
df2 = pd.read_csv("2024_data.csv")

# Use llm_call to analyze differences
comparison = llm_call(
    [df1.describe().to_string(), df2.describe().to_string()],
    "Compare these two datasets. What changed? What are the trends?"
)

result(comparison)
</helpers>
```

### `{{featured_helpers}}` - Key tools for this persona

**Default:** (all helpers shown)
```
## Core Functions

1. llm_call(expr_list, instructions) - Delegate text analysis to LLM
2. llm_bind(data, "function_signature") - Extract structured data
3. llm_list_bind(data, instructions, count) - Extract lists
4. pandas_bind(df) - Create smart DataFrames with .ask() method
5. coerce(data, type) - Type conversion
6. sabre_call(description, expr_list) - Recursive task delegation
7. result(value) - Return final result

## Tools

- Bash.execute(command) - Run bash commands
- Search.search(query, num_results=10) - Web search
- Web.download(urls) - Download web content as screenshots/files
- Web.download_csv(url) - Download CSV files
- Browser.screenshot(url) - Take webpage screenshot
- FS.read_file(path) - Read files
- FS.write_file(path, content) - Write files
- helpers() - See all available functions
- helpers("search_term") - Search for specific helpers
```

**Web Researcher:** (featured: Search, Web, llm_call)
```
## Search & Web Tools

### Search
- Search.search(query, num_results=10) - Web search via DuckDuckGo
  Returns list of SearchResult objects with .title, .url, .snippet

### Web
- Web.download(urls) - Download web pages as screenshots
  Accepts URL string, list of URLs, or SearchResult list
  Returns ImageContent (screenshots) or TextContent (PDFs/CSVs)

- Web.download_csv(url) - Download CSV file to temp path
  Returns file path for use with pd.read_csv()

### Browser
- Browser.screenshot(url, timeout=30000) - Screenshot webpage
  Uses Playwright for JavaScript-heavy sites
  Returns bytes of full-page PNG

## Core Analysis Functions

- llm_call(expr_list, instructions) - Analyze/summarize/extract from text or images
- llm_bind(data, "function(param: type) -> type") - Extract structured data
- llm_list_bind(data, instructions, count) - Extract list of items
- result(value) - Return final answer
```

**Coder:** (featured: Bash, FS, llm_call)
```
## File System Tools

### FS
- FS.read_file(path) - Read file contents
  Returns string of file contents

- FS.write_file(path, content) - Write content to file
  Creates parent directories if needed

- FS.list_files(directory=".", pattern="*") - List files matching pattern
  Returns list of file paths

### Bash
- Bash.execute(command) - Execute bash command
  Returns stdout/stderr output
  Use for running tests, git commands, package managers, etc.

## Code Analysis Functions

- llm_call(expr_list, instructions) - Analyze code, identify bugs, generate fixes
- result(value) - Return final answer
```

**Data Analyst:** (featured: pandas, matplotlib, Web.download_csv)
```
## Data Tools

### Web
- Web.download_csv(url) - Download CSV file
  Returns temp file path for pd.read_csv()

### Pandas Integration
- pandas_bind(df) - Create smart DataFrame
  Returns DataFrame with .ask(question) method
  Use .ask() to query data in natural language

### Matplotlib
- Import and use normally: `import matplotlib.pyplot as plt`
- Figures are automatically captured and displayed
- No need to call plt.show() or save manually

## Analysis Functions

- llm_call(expr_list, instructions) - Analyze data, generate insights
- coerce(data, type) - Type conversion for data cleaning
- result(value) - Return final analysis
```

## Persona Configuration

**File:** `llmvm2/config/personas.yaml`

```yaml
personas:
  default:
    name: "LLMVM Default"
    description: "General purpose LLM assistant with all capabilities"

    persona: |
      You are a helpful LLM Assistant. You are given a problem description or
      a question, and using the techniques described in the Toolformer paper,
      you deconstruct the problem/query/question into natural language and
      optional tool helper calls via the Python language.

    approach: |
      You take natural language problems, questions, and queries and solve
      them by breaking them down into smaller, discrete tasks and optionally
      working with me and my Python runtime to program and execute those tasks.

    workflow: ""  # No specific workflow

    helpers:
      - "*"  # All helpers available

  data-scientist:
    name: "Data Scientist Expert"
    description: "Expert database analyst and business intelligence specialist"

    persona: |
      You are an expert data scientist with deep knowledge of SQL databases,
      business intelligence, and data analysis. You have the ability to connect
      to and learn from databases, building persistent knowledge about their
      structure and business context over time.

    approach: |
      You approach database analysis methodically: first learning the structure,
      understanding the business domain, then building a comprehensive mental
      model that persists across sessions. Every analysis is verified through
      multiple query approaches before presenting results.

    workflow: |
      ### Database Analysis Workflow

      **Discovery Phase:**
      1. Check existing semantic analyses: `SemanticDatabaseHelpers.list_available_semantic_analyses()`
      2. Connect to database: `DatabaseHelpers.connect_database(db_path)`
      3. Review schema: `DatabaseHelpers.get_database_schema(db_path)`
      4. Review context: `DatabaseHelpers.get_business_context(db_path)`
      5. Build semantic understanding: `SemanticDatabaseHelpers.create_semantic_understanding(db_path)`

      **Analysis Phase:**
      1. Get domain suggestions: `DatabaseHelpers.suggest_analysis_queries(db_path)`
      2. Use `llm_call()` to understand user question
      3. Find relevant schema: `SemanticDatabaseHelpers.get_semantic_context(db_path, query)`
      4. Execute queries: `DatabaseHelpers.query_database(db_path, sql)`

      **Verification Phase (MANDATORY):**
      1. Generate alternatives: `SemanticDatabaseHelpers.generate_verification_queries(db_path, query)`
      2. Run 2-3 alternative query approaches
      3. Use `llm_call()` to compare results for consistency
      4. Only present results after verification confirms accuracy

      **Knowledge Building:**
      - Store insights: `DatabaseHelpers.add_insight(db_path, insight)`
      - Your learning persists across sessions via automatic caching
      - Build on previous context rather than restarting each time

    helpers:
      # Core BCL
      - "llm_call"
      - "llm_bind"
      - "llm_list_bind"
      - "pandas_bind"
      - "coerce"
      - "download"
      - "result"
      - "helpers"
      - "locals"
      - "read_memory"
      - "write_memory"
      - "read_memory_keys"
      - "read_file"
      - "write_file"
      - "delegate_task"
      - "llmvm_call"
      - "count_tokens"
      # Data-specific tools
      - "DatabaseHelpers.*"
      - "SemanticDatabaseHelpers.*"
      - "WebHelpers.download"
      - "WebHelpers.fetch_url"

  web-researcher:
    name: "Web Research Specialist"
    description: "Expert at finding, analyzing, and synthesizing web information"

    persona: |
      You are an expert web researcher skilled at finding accurate information,
      analyzing multiple sources, and synthesizing comprehensive, well-cited
      answers. You excel at fact-checking and source evaluation.

    approach: |
      You approach research systematically: understanding the question,
      identifying authoritative sources, cross-referencing findings, and
      synthesizing comprehensive answers with proper citations.

    workflow: |
      ### Web Research Workflow

      **Understanding Phase:**
      1. Break down the research question into sub-questions
      2. Identify what type of sources are needed (academic, news, technical, etc.)
      3. Use `llm_call()` to plan the research strategy

      **Gathering Phase:**
      1. Search for sources: `Search.search(query)` or `SearchTool.search(query)`
      2. Download content: `download([url1, url2, ...])`
      3. Extract relevant information from each source using `llm_call()`

      **Verification Phase:**
      1. Cross-reference findings across multiple sources
      2. Check publication dates and author credentials
      3. Identify contradictions or uncertainties
      4. Use `llm_call()` to assess source reliability

      **Synthesis Phase:**
      1. Combine findings into comprehensive answer
      2. Include citations for all claims
      3. Note any limitations or areas of uncertainty

    helpers:
      # Core BCL
      - "llm_call"
      - "llm_bind"
      - "llm_list_bind"
      - "pandas_bind"
      - "coerce"
      - "download"
      - "result"
      - "write_memory"
      - "read_memory"
      - "read_memory_keys"
      - "delegate_task"
      # Web research tools
      - "Search.*"
      - "SearchTool.*"
      - "WebHelpers.*"
      - "Browser.*"

  finance-analyst:
    name: "Finance & Markets Analyst"
    description: "Expert at financial analysis, market research, and SEC filings"

    persona: |
      You are an expert financial analyst with deep knowledge of markets,
      SEC filings, financial statements, and investment analysis. You provide
      data-driven insights on companies, sectors, and market trends.

    approach: |
      You approach financial analysis rigorously: gathering data from multiple
      sources, performing quantitative analysis, reviewing SEC filings, and
      generating actionable insights backed by data.

    workflow: |
      ### Financial Analysis Workflow

      **Data Gathering:**
      1. Identify ticker/company: `Market.search_ticker(company_name)` if needed
      2. Get fundamentals: `Market.get_company_info(ticker)`
      3. Get price data: `Market.get_stock_price(ticker, period='1y')`
      4. Review SEC filings: `Edgar.search_filings(ticker, form_type='10-K')`

      **Quantitative Analysis:**
      1. Calculate key metrics (P/E, revenue growth, margins, etc.)
      2. Compare to sector peers: `Market.get_sector_companies(sector)`
      3. Identify trends and anomalies
      4. Use `pandas_bind()` to create DataFrames for analysis

      **Qualitative Analysis:**
      1. Review management commentary from filings
      2. Assess competitive positioning
      3. Evaluate risks and opportunities

      **Synthesis:**
      1. Generate actionable insights
      2. Back all claims with specific data points
      3. Note risks and limitations
      4. Provide clear recommendations when appropriate

    helpers:
      # Core BCL
      - "llm_call"
      - "llm_bind"
      - "pandas_bind"
      - "coerce"
      - "result"
      - "write_memory"
      - "read_memory"
      - "count_tokens"
      # Finance tools
      - "Market.*"
      - "Edgar.*"
      - "WebHelpers.*"
      - "DatabaseHelpers.*"
```

## Architecture Components

### 1. Helper Registry

**File:** `llmvm2/server/helper_registry.py`

```python
from typing import List, Dict
import fnmatch
import logging

logger = logging.getLogger(__name__)


class HelperRegistry:
    """
    Manages helper registration and filtering based on persona configuration.
    """

    def __init__(self):
        # All available helper classes
        self._all_helpers: Dict[str, object] = {}

    def register(self, name: str, helper_class: object):
        """Register a helper class or function."""
        self._all_helpers[name] = helper_class
        logger.debug(f"Registered helper: {name}")

    def register_class(self, class_name: str, helper_class: type):
        """
        Register all methods of a helper class.

        Example: register_class("DatabaseHelpers", DatabaseHelpers)
        Creates: DatabaseHelpers.connect_database, DatabaseHelpers.query_database, etc.
        """
        import inspect

        for method_name, method in inspect.getmembers(helper_class, inspect.isfunction):
            if not method_name.startswith("_"):
                full_name = f"{class_name}.{method_name}"
                self.register(full_name, method)

    def get_filtered_helpers(self, patterns: List[str]) -> Dict[str, object]:
        """
        Get helpers matching the persona's allowed patterns.

        Args:
            patterns: List of patterns like ["DatabaseHelpers.*", "llm_call"]

        Returns:
            Dict of filtered helpers {name: helper_function}
        """
        if "*" in patterns:
            # All helpers allowed (default persona)
            return self._all_helpers.copy()

        filtered = {}

        for pattern in patterns:
            for name, helper in self._all_helpers.items():
                if fnmatch.fnmatch(name, pattern):
                    filtered[name] = helper

        logger.info(
            f"Filtered {len(filtered)}/{len(self._all_helpers)} helpers "
            f"for patterns: {patterns}"
        )
        return filtered

    def get_helper_descriptions(self, patterns: List[str]) -> str:
        """
        Get formatted helper descriptions for prompt injection.

        Args:
            patterns: List of allowed helper patterns

        Returns:
            Formatted string describing available helpers
        """
        filtered = self.get_filtered_helpers(patterns)

        # Group by class
        by_class: Dict[str, List[tuple[str, object]]] = {}

        for name, helper in filtered.items():
            if "." in name:
                class_name, method = name.split(".", 1)
                if class_name not in by_class:
                    by_class[class_name] = []
                by_class[class_name].append((method, helper))
            else:
                # Standalone function
                if "Core BCL" not in by_class:
                    by_class["Core BCL"] = []
                by_class["Core BCL"].append((name, helper))

        # Format for prompt
        lines = []
        for class_name, methods in sorted(by_class.items()):
            lines.append(f"\n## {class_name}\n")
            for method_name, helper in sorted(methods, key=lambda x: x[0]):
                full_name = (
                    f"{class_name}.{method_name}"
                    if class_name != "Core BCL"
                    else method_name
                )

                # Get docstring
                doc = getattr(helper, "__doc__", "") or ""
                first_line = doc.split("\n")[0].strip() if doc else ""

                lines.append(f"- {full_name}: {first_line}")

        return "\n".join(lines)
```

### 2. Persona Loader

**File:** `llmvm2/config/persona_loader.py`

```python
import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PersonaLoader:
    """Load persona configurations from YAML."""

    @staticmethod
    def load(persona_name: str = "default") -> Dict[str, Any]:
        """
        Load persona configuration.

        Args:
            persona_name: Name of persona to load

        Returns:
            Dict with persona, approach, workflow, helpers
        """
        # Try user config first
        user_config = Path.home() / ".config" / "llmvm2" / "personas.yaml"
        default_config = Path(__file__).parent / "personas.yaml"

        config_path = user_config if user_config.exists() else default_config

        logger.info(f"Loading personas from: {config_path}")

        with open(config_path) as f:
            personas = yaml.safe_load(f)

        if persona_name not in personas["personas"]:
            logger.warning(f"Persona '{persona_name}' not found, using 'default'")
            persona_name = "default"

        persona_config = personas["personas"][persona_name]
        logger.info(f"Loaded persona: {persona_config['name']}")

        return persona_config
```

### 3. Orchestrator Integration

**File:** `llmvm2/server/orchestrator.py`

```python
class Orchestrator:
    def __init__(
        self,
        executor: ResponseExecutor,
        runtime: PythonRuntime,
        persona: str = "default",  # NEW
        mode: str = "tools",
        model: str = None,
        event_callback: Callable[[Event], Awaitable[None]] = None,
    ):
        self.persona_name = persona
        self.mode = mode
        self.model = model

        # Load persona configuration
        from llmvm2.config.persona_loader import PersonaLoader

        self.persona_config = PersonaLoader.load(persona)

        # Initialize helper registry with filtered helpers
        from llmvm2.server.helper_registry import HelperRegistry

        self.helper_registry = HelperRegistry()
        self._register_all_helpers()

        # Get filtered helpers for this persona
        allowed_patterns = self.persona_config["helpers"]
        self.active_helpers = self.helper_registry.get_filtered_helpers(
            allowed_patterns
        )

        # Initialize runtime with filtered helpers
        self.runtime = PythonRuntime(active_helpers=self.active_helpers)
        self.executor = executor
        self.event_callback = event_callback

        logger.info(
            f"Initialized persona '{persona}': "
            f"{len(self.active_helpers)} helpers available"
        )

    def _register_all_helpers(self):
        """Register all available helper classes."""
        # Core BCL functions
        from llmvm2.server.bcl import BCL

        bcl = BCL()
        self.helper_registry.register("llm_call", bcl.llm_call)
        self.helper_registry.register("llm_bind", bcl.llm_bind)
        # ... register all BCL functions

        # Tool classes
        from llmvm2.server.tools import (
            DatabaseHelpers,
            SemanticDatabaseHelpers,
            WebHelpers,
            Search,
            SearchTool,
            Market,
            Edgar,
            Browser,
        )

        self.helper_registry.register_class("DatabaseHelpers", DatabaseHelpers)
        self.helper_registry.register_class(
            "SemanticDatabaseHelpers", SemanticDatabaseHelpers
        )
        self.helper_registry.register_class("WebHelpers", WebHelpers)
        self.helper_registry.register_class("Search", Search)
        self.helper_registry.register_class("SearchTool", SearchTool)
        self.helper_registry.register_class("Market", Market)
        self.helper_registry.register_class("Edgar", Edgar)
        self.helper_registry.register_class("Browser", Browser)

    def load_default_instructions(self) -> str:
        """Load instructions with persona-specific customization."""
        # Get effective mode (may auto-select reasoning)
        effective_mode = self._get_effective_mode()

        # Load base template
        prompt_name = "continuation_execution.prompt"

        # Get filtered helper descriptions
        helper_descriptions = self.helper_registry.get_helper_descriptions(
            self.persona_config["helpers"]
        )

        # Load with persona-specific template variables
        template = {
            "persona": self.persona_config["persona"],
            "persona_approach": self.persona_config["approach"],
            "persona_workflow": self.persona_config.get("workflow", ""),
            "filtered_helpers": helper_descriptions,
            "context_window_tokens": "128000",
            "context_window_words": "96000",
            "context_window_bytes": "512000",
            "scratchpad_token": "scratchpad",
        }

        prompt_parts = PromptLoader.load(
            prompt_name, mode=effective_mode, template=template
        )

        return f"{prompt_parts['system_message']}\n\n{prompt_parts['user_message']}"
```

### 4. Python Runtime Updates

**File:** `llmvm2/server/python_runtime.py`

```python
class PythonRuntime:
    def __init__(self, active_helpers: Dict[str, object]):
        """
        Initialize runtime with filtered helpers.

        Args:
            active_helpers: Dict of allowed helpers for this persona
        """
        self.active_helpers = active_helpers
        self.globals = {}
        self._init_runtime_with_helpers()

    def _init_runtime_with_helpers(self):
        """Inject only active helpers into runtime namespace."""
        # Inject Python builtins
        import builtins

        self.globals["__builtins__"] = builtins

        # Inject allowed libraries
        import numpy as np, pandas as pd, scipy, asyncio

        self.globals.update(
            {
                "np": np,
                "pd": pd,
                "scipy": scipy,
                "asyncio": asyncio,
            }
        )

        # Inject filtered helpers
        for name, helper in self.active_helpers.items():
            if "." in name:
                class_name, method = name.split(".", 1)
                # Create class instance if needed
                if class_name not in self.globals:
                    self.globals[class_name] = type(class_name, (), {})()
                setattr(self.globals[class_name], method, helper)
            else:
                # Standalone function
                self.globals[name] = helper

        logger.info(f"Runtime initialized with {len(self.active_helpers)} helpers")

    def get_available_functions(self) -> str:
        """Get string describing available functions for prompt."""
        # This is no longer used - filtered_helpers is injected directly
        return ""
```

## Benefits

### ✅ Context Efficiency
- Default persona: ~100 helpers
- Data scientist persona: ~15 helpers
- Saves thousands of tokens per request
- Faster responses, lower costs

### ✅ Focused Experience
- Model only sees relevant tools for the domain
- Clearer guidance on what to use
- Better tool selection accuracy
- Reduced confusion

### ✅ Three-Layer Architecture
- System flow: constant, shared execution mechanics
- Meta patterns: constant, strategic tool use
- Domain workflow: variable, persona-specific expertise

### ✅ Easy to Extend
- Add new persona by editing YAML config
- No code changes needed
- Share base prompt template
- Override in user config

### ✅ Composable
- Personas define WHAT (identity, helpers, workflow)
- Modes define HOW (execution behavior)
- Orthogonal concerns, multiply together

### ✅ Backward Compatible
- Default persona = all helpers (current behavior)
- Existing prompts work unchanged
- Gradual migration path

## Migration Path

### Phase 1: Infrastructure (Week 1)

**Tasks:**
1. Create `HelperRegistry` class
2. Create `personas.yaml` configuration
3. Create `PersonaLoader` class
4. Update `Orchestrator` to accept persona parameter

**Files created:**
- `llmvm2/server/helper_registry.py`
- `llmvm2/config/personas.yaml`
- `llmvm2/config/persona_loader.py`

**Files modified:**
- `llmvm2/server/orchestrator.py`
- `llmvm2/server/python_runtime.py`

### Phase 2: Template Restructuring (Week 2)

**Tasks:**
1. Extract persona sections from current prompt
2. Create base template with placeholders:
   - `{{persona}}`
   - `{{persona_approach}}`
   - `{{persona_workflow}}`
   - `{{filtered_helpers}}`
3. Update `PromptLoader` to handle new template vars
4. Test with default persona (should be identical to current)

**Files modified:**
- `llmvm2/server/prompts/python_continuation_execution_responses.prompt`
- `llmvm2/common/utils/prompt_loader.py`

### Phase 3: Persona Definitions (Week 3)

**Tasks:**
1. Define `default` persona (all helpers, generic)
2. Define `data-scientist` persona (database helpers only)
3. Define `web-researcher` persona (search/web helpers only)
4. Define `finance-analyst` persona (market/edgar helpers only)
5. Test each persona independently

**Files modified:**
- `llmvm2/config/personas.yaml`

### Phase 4: Server Integration (Week 4)

**Tasks:**
1. Add `--persona` CLI argument to server
2. Update WebSocket server to support persona selection
3. Add `/persona` slash command to client
4. Show active persona in client UI
5. Document persona system

**Files modified:**
- `llmvm2/server/api/server.py`
- `llmvm2/client/client.py`
- `README.md`, `CLAUDE.md`

## Example Usage

```bash
# Start with default persona (all helpers)
uv run python -m llmvm2.server

# Start with data scientist persona
uv run python -m llmvm2.server --persona data-scientist

# Start with web researcher persona
uv run python -m llmvm2.server --persona web-researcher

# Client can check current persona
/persona

# Client can list available personas
/persona list

# Future: runtime switching (optional)
/persona data-scientist
```

## Testing Checklist

- [ ] HelperRegistry correctly filters helpers by pattern
- [ ] PersonaLoader loads persona configs from YAML
- [ ] Default persona has all helpers (backward compatible)
- [ ] Data scientist persona has only database helpers
- [ ] Web researcher persona has only search/web helpers
- [ ] Finance analyst persona has only market/edgar helpers
- [ ] Template correctly injects persona sections
- [ ] Runtime only exposes filtered helpers
- [ ] Attempting to call unavailable helper throws error
- [ ] User config overrides default config
- [ ] `/persona` command works
- [ ] Persona shown in client UI

## Open Questions

1. **Should personas be runtime-switchable?**
   - Pro: More flexible, can change mid-conversation
   - Con: More complex, need to handle helper registry changes
   - **Recommendation**: Start with startup-only, add runtime switching later

2. **Should persona affect model selection?**
   - e.g., data-scientist defaults to o1-mini for better reasoning
   - **Recommendation**: Keep separate for now, add as optional persona config later

3. **How to handle custom user-defined personas?**
   - Load from `~/.config/llmvm2/personas.yaml`
   - Override default personas or add new ones
   - **Recommendation**: Load user config first, fall back to default

4. **Should we have persona-specific result formatting?**
   - e.g., data-scientist always formats results as tables
   - **Recommendation**: Not in initial implementation, consider later

## Future Enhancements

- [ ] Runtime persona switching via `/persona` command
- [ ] Persona-specific model selection (e.g., data-scientist → o1-mini)
- [ ] Persona-specific temperature/parameters
- [ ] Auto-detect best persona based on user query
- [ ] Persona inheritance (e.g., data-scientist extends default)
- [ ] Persona-specific result formatters
- [ ] Custom user-defined personas via config
- [ ] Persona indicator in client prompt (e.g., `[data-scientist]>`)
- [ ] Persona-specific example queries/tutorials

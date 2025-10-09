# LLMVM2 Persona System Architecture

## Problem Statement

Looking at the data scientist prompt vs. the base continuation prompt reveals a key architectural issue:

**Current State:**
- Base prompt: 692 lines of execution instructions
- Data scientist prompt: 97 lines with domain-specific persona and workflow
- Both inject ALL helpers via `{{functions}}` template
- Result: Wasted tokens on irrelevant tools, model confusion, no way to create focused distributions

**The Real Problem: Helper Bloat**

LLMVM has 13+ tool helper classes:
- Browser automation: Browser, Chrome, MacOSChromeBrowser
- Search: Search, SearchTool, SearchHN, InternalSearch
- Data: Database, SemanticDatabase, Sheets
- Finance: Market, Edgar
- Web: WebHelpers

For a data scientist persona, you only need:
- Database, SemanticDatabase, WebHelpers (downloads)
- Core BCL (llm_call, pandas_bind, etc.)

But currently ALL helpers are injected into every prompt = thousands of wasted tokens.

## Solution: Persona System

**Personas** are configuration-based LLMVM distributions with:
1. **Identity** - Who you are and your expertise domain
2. **Filtered helper sets** - Only relevant tools for that domain
3. **Domain workflow** - Specific best practices and methodologies
4. **Shared execution model** - Same base prompt template

### Personas vs Modes

| Aspect | Modes | Personas |
|--------|-------|----------|
| **Purpose** | Change execution behavior | Change domain focus |
| **Examples** | tools, direct, reasoning, program | default, data-scientist, web-researcher, finance-analyst |
| **What changes** | How code executes | Which helpers + domain expertise |
| **Prompt structure** | Different continuation prompts | Same base template + persona config |
| **Runtime switching** | Yes, via `/mode` | No, set at startup |

## Three-Layer Prompt Architecture

All personas share the same three-layer structure:

### Layer 1: System Execution Flow (Constant)

The technical mechanics of how the continuation system works:
- How `<helpers>` blocks are executed
- What tags to use (`</complete>`, `<scratchpad>`)
- How results are returned in `<helpers_result>` tags
- Variable persistence across blocks

**This layer is identical for all personas** - it's the core LLMVM execution model.

### Layer 2: Meta Execution Patterns (Constant)

Strategic patterns for effective problem solving:
- **When to use llm_call**: Text analysis, summarization, extraction, comparison
- **When to use delegate_task**: Parallelization and compartmentalization
- **When to use memory**: Context window management (write_memory/read_memory)
- **When to use binding helpers**: Data extraction (llm_bind, llm_list_bind, pandas_bind)
- **When to verify results**: Cross-validation for critical outputs

**This layer is also constant** - it teaches strategic tool use patterns that apply universally.

### Layer 3: Domain Workflow (Variable)

Persona-specific workflows and best practices:
- **Data scientist**: Discovery → Analysis → Verification → Insights
- **Web researcher**: Understanding → Gathering → Cross-referencing → Synthesis
- **Finance analyst**: Data Gathering → Quantitative Analysis → SEC Review → Insights

**This layer varies by persona** - it's where domain expertise lives.

## Prompt Template Structure

```
[system_message]
{{persona}}

[Context: date, timezone, scratch directory, context window info...]

[user_message]
{{persona_approach}}

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
* Keep expr_list under {{context_window_words}} words

**Data binding and extraction:**
* Use `llm_bind(data, "function_signature")` to intelligently bind arbitrary text to function parameters
* Use `llm_list_bind(data, instructions, count)` to extract structured lists from text
* Use `pandas_bind(data)` to create DataFrames from arbitrary data

**Parallel execution:**
* Use `delegate_task(description, context)` for compartmentalized sub-tasks
* Use `asyncio.gather()` to run multiple delegate_task calls in parallel
* Pass scratchpad thinking to delegated tasks for context

**Context window management:**
* Use `write_memory(key, summary, content)` to store completed work
* Use `read_memory_keys()` to see what's stored
* Use `read_memory(key)` to retrieve stored content
* Free up context by moving completed sub-tasks to memory

**Verification and validation:**
* For critical results, cross-validate with alternative approaches
* Use multiple data sources when available
* Explicitly state confidence levels and limitations

{{persona_workflow}}

## Available Helpers

{{filtered_helpers}}

[Technical details: 25 special functions, Content classes, imports, etc.]
```

## Template Variables

### `{{persona}}` - Identity (system_message)

**Default:**
```
You are a helpful LLM Assistant. You are given a problem description or a question,
and using the techniques described in the Toolformer paper, you deconstruct the
problem/query/question into natural language and optional tool helper calls via
the Python language.
```

**Data Scientist:**
```
You are an expert data scientist with deep knowledge of SQL databases, business
intelligence, and data analysis. You have the ability to connect to and learn
from databases, building persistent knowledge about their structure and business
context over time.
```

**Web Researcher:**
```
You are an expert web researcher skilled at finding accurate information,
analyzing multiple sources, and synthesizing comprehensive, well-cited answers.
You excel at fact-checking and source evaluation.
```

**Finance Analyst:**
```
You are an expert financial analyst with deep knowledge of markets, SEC filings,
financial statements, and investment analysis. You provide data-driven insights
on companies, sectors, and market trends.
```

### `{{persona_approach}}` - How you work (user_message intro)

**Default:**
```
You take natural language problems, questions, and queries and solve them by
breaking them down into smaller, discrete tasks and optionally working with
me and my Python runtime to program and execute those tasks.
```

**Data Scientist:**
```
You approach database analysis methodically: first learning the structure,
understanding the business domain, then building a comprehensive mental model
that persists across sessions. Every analysis is verified through multiple
query approaches before presenting results.
```

**Web Researcher:**
```
You approach research systematically: understanding the question, identifying
authoritative sources, cross-referencing findings, and synthesizing comprehensive
answers with proper citations.
```

**Finance Analyst:**
```
You approach financial analysis rigorously: gathering data from multiple sources,
performing quantitative analysis, reviewing SEC filings, and generating actionable
insights backed by data.
```

### `{{persona_workflow}}` - Domain workflow (optional section)

**Default:** (empty or minimal)

**Data Scientist:**
```markdown
### Database Analysis Workflow

Your domain-specific approach:

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
```

**Web Researcher:**
```markdown
### Web Research Workflow

Your domain-specific approach:

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
```

**Finance Analyst:**
```markdown
### Financial Analysis Workflow

Your domain-specific approach:

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
```

### `{{filtered_helpers}}` - Only relevant tools

**Default:** (all helpers)
```
Functions:

{{functions}}  # All tool helpers injected

[25 special functions listed in detail...]
```

**Data Scientist:** (filtered to ~10 relevant helpers)
```
## Database Tools

### DatabaseHelpers
- DatabaseHelpers.connect_database(db_path) - Connect and learn database structure
- DatabaseHelpers.query_database(db_path, sql) - Execute SQL queries
- DatabaseHelpers.get_database_schema(db_path) - Get cached schema information
- DatabaseHelpers.get_business_context(db_path) - Review learned business insights
- DatabaseHelpers.suggest_analysis_queries(db_path) - Get domain-specific query suggestions
- DatabaseHelpers.add_insight(db_path, insight) - Store business insights for future sessions

### SemanticDatabaseHelpers
- SemanticDatabaseHelpers.list_available_semantic_analyses() - List all knowledge bases
- SemanticDatabaseHelpers.create_semantic_understanding(db_path) - Build vector store index
- SemanticDatabaseHelpers.get_semantic_context(db_path, query) - Find relevant schema context
- SemanticDatabaseHelpers.generate_verification_queries(db_path, original_query) - Generate cross-validation queries
- SemanticDatabaseHelpers.get_semantic_suggestions(db_path, context) - Get AI-powered analysis suggestions

### WebHelpers (Download Only)
- WebHelpers.download(url) - Download data files
- WebHelpers.fetch_url(url) - Fetch web content

## Core BCL Functions

1. llm_call(expression_list, instructions) - Perform text analysis and computation
2. llm_bind(expression, function_str) - Intelligently bind data to function parameters
3. llm_list_bind(expression, instructions, count) - Extract structured lists from text
4. pandas_bind(expression) - Create intelligent DataFrames with ask() method
5. coerce(expression, type_var) - Type coercion
6. download(url_list) - Download files and data
7. result(expression) - Return final results
8. write_memory(key, summary, content) - Store content to memory
9. read_memory(key) - Retrieve content from memory
10. read_memory_keys() - List all memory keys
11. read_file(path) - Read local files
12. write_file(filename, content) - Write to scratch directory
```

**Web Researcher:** (filtered to search & web tools)
```
## Search Tools

### Search
- Search.search(query, num_results=10) - General web search
- Search.search_news(query, days_back=7) - News search
- Search.search_academic(query) - Academic paper search

### SearchTool
- SearchTool.search(query, engine='google') - Multi-engine search
- SearchTool.get_related_queries(query) - Find related search queries

### WebHelpers
- WebHelpers.download(url) - Download web content
- WebHelpers.fetch_url(url) - Fetch HTML content
- WebHelpers.extract_links(content) - Extract all links from content
- WebHelpers.extract_text(content) - Clean text extraction from HTML

### Browser (Optional)
- Browser.navigate(url) - Navigate to webpage
- Browser.get_content() - Get rendered page content
- Browser.screenshot() - Take page screenshot

## Core BCL Functions

1. llm_call(expression_list, instructions) - Text analysis and summarization
2. llm_bind(expression, function_str) - Bind data to function parameters
3. llm_list_bind(expression, instructions, count) - Extract structured lists
4. download(url_list) - Download multiple URLs
5. result(expression) - Return final results
6. write_memory(key, summary, content) - Store research findings
7. read_memory(key) - Retrieve stored findings
```

**Finance Analyst:** (filtered to market & finance tools)
```
## Financial Data Tools

### Market
- Market.search_ticker(company_name) - Find ticker symbol
- Market.get_company_info(ticker) - Get company fundamentals
- Market.get_stock_price(ticker, period='1y') - Historical price data
- Market.get_financials(ticker) - Income statement, balance sheet, cash flow
- Market.get_sector_companies(sector) - List companies in sector
- Market.calculate_metrics(ticker) - Calculate financial ratios

### Edgar (SEC Filings)
- Edgar.search_filings(ticker, form_type='10-K') - Search SEC filings
- Edgar.get_filing(accession_number) - Download specific filing
- Edgar.extract_section(filing, section_name) - Extract filing sections

### WebHelpers
- WebHelpers.download(url) - Download financial reports
- WebHelpers.fetch_url(url) - Fetch investor relations pages

### DatabaseHelpers (Optional)
- DatabaseHelpers.query_database(db_path, sql) - Query financial databases

## Core BCL Functions

1. llm_call(expression_list, instructions) - Analyze financial texts
2. pandas_bind(expression) - Create DataFrames from financial data
3. coerce(expression, type_var) - Type coercion for calculations
4. result(expression) - Return analysis results
5. write_memory(key, summary, content) - Store analysis findings
6. read_memory(key) - Retrieve stored findings
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
            if not method_name.startswith('_'):
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
                full_name = f"{class_name}.{method_name}" if class_name != "Core BCL" else method_name

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
    def load(persona_name: str = 'default') -> Dict[str, Any]:
        """
        Load persona configuration.

        Args:
            persona_name: Name of persona to load

        Returns:
            Dict with persona, approach, workflow, helpers
        """
        # Try user config first
        user_config = Path.home() / '.config' / 'llmvm2' / 'personas.yaml'
        default_config = Path(__file__).parent / 'personas.yaml'

        config_path = user_config if user_config.exists() else default_config

        logger.info(f"Loading personas from: {config_path}")

        with open(config_path) as f:
            personas = yaml.safe_load(f)

        if persona_name not in personas['personas']:
            logger.warning(f"Persona '{persona_name}' not found, using 'default'")
            persona_name = 'default'

        persona_config = personas['personas'][persona_name]
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
        persona: str = 'default',  # NEW
        mode: str = 'tools',
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
        allowed_patterns = self.persona_config['helpers']
        self.active_helpers = self.helper_registry.get_filtered_helpers(allowed_patterns)

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
        self.helper_registry.register_class("SemanticDatabaseHelpers", SemanticDatabaseHelpers)
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
        prompt_name = 'continuation_execution.prompt'

        # Get filtered helper descriptions
        helper_descriptions = self.helper_registry.get_helper_descriptions(
            self.persona_config['helpers']
        )

        # Load with persona-specific template variables
        template = {
            'persona': self.persona_config['persona'],
            'persona_approach': self.persona_config['approach'],
            'persona_workflow': self.persona_config.get('workflow', ''),
            'filtered_helpers': helper_descriptions,
            'context_window_tokens': '128000',
            'context_window_words': '96000',
            'context_window_bytes': '512000',
            'scratchpad_token': 'scratchpad',
        }

        prompt_parts = PromptLoader.load(
            prompt_name,
            mode=effective_mode,
            template=template
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
        self.globals['__builtins__'] = builtins

        # Inject allowed libraries
        import numpy as np, pandas as pd, scipy, asyncio
        self.globals.update({
            'np': np,
            'pd': pd,
            'scipy': scipy,
            'asyncio': asyncio,
        })

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

# LLMVM2 Variants Architecture

## Problem Statement

Looking at the data scientist prompt vs. the base continuation prompt reveals a key architectural question:

**Should domain-specific configurations be separate "modes" with different prompts, OR should they be "variants" with filtered helper sets?**

### Current Situation

**Base Prompt** (`python_continuation_execution_responses.prompt`):
- 692 lines of detailed execution instructions
- Generic persona: "You are a helpful LLM Assistant"
- Lists all 25 special functions + ALL tool helpers via `{{functions}}`
- No domain-specific guidance

**Data Scientist Prompt** (`data_scientist_expert.prompt`):
- 97 lines, much more concise
- Specific persona: "You are an expert data scientist..."
- Detailed workflow for database analysis (Discovery → Analysis → Verification → Insights)
- Highlights specific helpers (DatabaseHelpers, SemanticDatabaseHelpers)
- But still includes `{{functions}}` - so ALL helpers are available!

### The Real Problem: Helper Bloat

LLMVM has 13+ tool helper classes:
- Browser automation: Browser, Chrome, MacOSChromeBrowser
- Search: Search, SearchTool, SearchHN, InternalSearch
- Data: Database, SemanticDatabase, Sheets
- Finance: Market, Edgar
- Web: WebHelpers

**For a data scientist variant, you only need:**
- Database
- SemanticDatabase
- WebHelpers (for data downloads)
- Core BCL (llm_call, pandas_bind, etc.)

**But currently, ALL helpers are injected into every prompt via `{{functions}}`.**

This means:
- 🔴 Wasted context tokens on irrelevant helpers
- 🔴 Model confusion from too many options
- 🔴 Harder for model to find the right tool
- 🔴 No way to create focused, specialized distributions

## Proposed Solution: Variants System

**Variants** are configuration-based LLMVM distributions with:
1. **Filtered helper sets** - only relevant tools for that domain
2. **Domain persona** - specialized system message
3. **Workflow guidance** - optional domain-specific instructions
4. **Shared execution model** - same base continuation prompt template

### Variants vs Modes

| Aspect | Modes | Variants |
|--------|-------|----------|
| **Purpose** | Change execution behavior | Change domain focus |
| **Examples** | tools, direct, reasoning, program | default, data-scientist, web-researcher, finance-analyst |
| **What changes** | How code executes | Which helpers available + persona |
| **Prompt structure** | Different continuation prompts | Same base template + variant config |
| **Runtime switching** | Yes, via `/mode` | No, set at startup via config |

### Key Insight

The data scientist prompt shows that:
- **80% is execution mechanics** (same as base prompt)
- **15% is persona and workflow** (variant-specific)
- **5% is helper filtering** (currently missing!)

Instead of duplicating the entire prompt, we should:
1. Share the base execution template
2. Inject variant-specific persona and workflow
3. **Filter helpers to only those needed for the variant**

## Architecture Design

### 1. Variant Configuration

**File:** `llmvm2/config/variants.yaml`

```yaml
variants:
  default:
    name: "LLMVM Default"
    description: "General purpose LLM assistant with all capabilities"
    persona: "You are a helpful LLM Assistant."
    helpers:
      - "*"  # All helpers
    workflow: null  # No specific workflow

  data-scientist:
    name: "Data Scientist Expert"
    description: "Expert database analyst and business intelligence specialist"
    persona: |
      You are an expert data scientist with deep knowledge of SQL databases,
      business intelligence, and data analysis. You have the ability to connect
      to and learn from databases, building persistent knowledge about their
      structure and business context over time.
    helpers:
      # Core BCL - always available
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
      - "read_file"
      - "write_file"
      # Data-specific tools
      - "DatabaseHelpers.*"
      - "SemanticDatabaseHelpers.*"
      - "WebHelpers.download"
      - "WebHelpers.fetch_url"
    workflow: |
      Your workflow for database analysis follows these principles:

      **Discovery and Learning Phase:**
      * Check available semantic analyses using SemanticDatabaseHelpers.list_available_semantic_analyses()
      * Connect using DatabaseHelpers.connect_database()
      * Examine schema using DatabaseHelpers.get_database_schema()
      * Review existing context with DatabaseHelpers.get_business_context()
      * Create semantic understanding using SemanticDatabaseHelpers.create_semantic_understanding()

      **Analysis and Insight Generation:**
      * Use DatabaseHelpers.suggest_analysis_queries() for domain-specific suggestions
      * Execute queries using DatabaseHelpers.query_database()
      * ALWAYS verify results using SemanticDatabaseHelpers.generate_verification_queries()
      * Focus on key business metrics and actionable insights

      **Persistent Knowledge Building:**
      * Your learning persists across sessions via automatic caching
      * Build on previous insights rather than starting from scratch
      * Maintain growing understanding of business context

  web-researcher:
    name: "Web Research Specialist"
    description: "Expert at finding, analyzing, and synthesizing web information"
    persona: |
      You are an expert web researcher skilled at finding accurate information,
      analyzing sources, and synthesizing comprehensive answers from multiple sources.
    helpers:
      # Core BCL
      - "llm_call"
      - "llm_bind"
      - "llm_list_bind"
      - "pandas_bind"
      - "coerce"
      - "download"
      - "result"
      # Web research tools
      - "Search.*"
      - "SearchTool.*"
      - "WebHelpers.*"
      - "Browser.*"
    workflow: |
      Your research workflow:
      1. Understand the research question and break it into sub-questions
      2. Use Search helpers to find authoritative sources
      3. Download and analyze content from multiple sources
      4. Cross-reference findings for accuracy
      5. Synthesize a comprehensive answer with citations

  finance-analyst:
    name: "Finance & Markets Analyst"
    description: "Expert at financial analysis, market research, and SEC filings"
    persona: |
      You are an expert financial analyst with deep knowledge of markets,
      SEC filings, financial statements, and investment analysis.
    helpers:
      # Core BCL
      - "llm_call"
      - "llm_bind"
      - "pandas_bind"
      - "coerce"
      - "result"
      # Finance tools
      - "Market.*"
      - "Edgar.*"
      - "WebHelpers.*"
      - "DatabaseHelpers.*"
    workflow: |
      Your analysis workflow:
      1. Identify the company/asset and key metrics needed
      2. Gather data from Market helpers (prices, fundamentals)
      3. Review SEC filings using Edgar helpers
      4. Perform comparative analysis and trend identification
      5. Generate actionable insights with data backing
```

### 2. Helper Filtering System

**File:** `llmvm2/server/helper_registry.py`

```python
from typing import List, Dict, Set
import fnmatch
import logging

logger = logging.getLogger(__name__)

class HelperRegistry:
    """
    Manages helper registration and filtering based on variant configuration.
    """

    def __init__(self):
        # All available helper classes
        self._all_helpers: Dict[str, object] = {}

    def register(self, name: str, helper_class: object):
        """Register a helper class."""
        self._all_helpers[name] = helper_class
        logger.debug(f"Registered helper: {name}")

    def get_filtered_helpers(self, patterns: List[str]) -> Dict[str, object]:
        """
        Get helpers matching the variant's allowed patterns.

        Args:
            patterns: List of patterns like ["DatabaseHelpers.*", "llm_call"]

        Returns:
            Dict of filtered helpers {name: helper_class}
        """
        if "*" in patterns:
            # All helpers allowed
            return self._all_helpers.copy()

        filtered = {}

        for pattern in patterns:
            # Handle both class patterns (DatabaseHelpers.*) and function names (llm_call)
            for name, helper in self._all_helpers.items():
                if fnmatch.fnmatch(name, pattern):
                    filtered[name] = helper

        logger.info(f"Filtered {len(filtered)}/{len(self._all_helpers)} helpers for patterns: {patterns}")
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
        by_class: Dict[str, List[str]] = {}

        for name in filtered.keys():
            if "." in name:
                class_name, method = name.split(".", 1)
                if class_name not in by_class:
                    by_class[class_name] = []
                by_class[class_name].append(method)
            else:
                # Standalone function
                if "Core" not in by_class:
                    by_class["Core"] = []
                by_class["Core"].append(name)

        # Format for prompt
        lines = []
        for class_name, methods in sorted(by_class.items()):
            lines.append(f"\n## {class_name}")
            for method in sorted(methods):
                full_name = f"{class_name}.{method}" if class_name != "Core" else method
                helper = filtered[full_name]
                # Get docstring or signature
                doc = getattr(helper, "__doc__", "") or ""
                first_line = doc.split("\n")[0] if doc else ""
                lines.append(f"- {full_name}: {first_line}")

        return "\n".join(lines)
```

### 3. Updated Prompt Template

**File:** `modes/tools/continuation_execution.prompt`

```
[system_message]
{{variant_persona}}

[user_message]
You take natural language problems, questions, and queries and solve them by
breaking them down into smaller, discrete tasks and optionally working with
me and my Python runtime to program and execute those tasks.

{{variant_workflow}}

## Execution Workflow

* This is a multi-turn conversation with automatic context management
* Decide if sub-tasks are required to solve the query/question/problem
* If no sub-tasks needed, emit answer and finish with </complete> token
* For complex tasks, think about sub-tasks in <scratchpad></scratchpad>
* Emit Python code in <helpers></helpers> blocks to execute tasks
* Variables and methods persist across <helpers> blocks
* Results returned in <helpers_result></helpers_result> tags

## Available Helpers

{{filtered_helpers}}

[Rest of base template with execution details...]
```

### 4. Orchestrator Integration

**File:** `llmvm2/server/orchestrator.py`

```python
class Orchestrator:
    def __init__(
        self,
        executor: ResponseExecutor,
        runtime: PythonRuntime,
        variant: str = 'default',  # NEW
        mode: str = 'tools',
        model: str = None,
        event_callback: Callable[[Event], Awaitable[None]] = None,
    ):
        self.variant = variant
        self.mode = mode
        self.model = model

        # Load variant configuration
        from llmvm2.config.variant_loader import VariantLoader
        self.variant_config = VariantLoader.load(variant)

        # Initialize helper registry with filtered helpers
        from llmvm2.server.helper_registry import HelperRegistry
        self.helper_registry = HelperRegistry()
        self._register_all_helpers()

        # Get filtered helpers for this variant
        allowed_patterns = self.variant_config['helpers']
        self.active_helpers = self.helper_registry.get_filtered_helpers(allowed_patterns)

        logger.info(f"Initialized variant '{variant}': {len(self.active_helpers)} helpers available")

    def load_default_instructions(self) -> str:
        """Load instructions with variant-specific customization."""
        # Get effective mode
        effective_mode = self._get_effective_mode()

        # Load base template
        prompt_name = 'continuation_execution.prompt'

        # Get filtered helper descriptions
        helper_descriptions = self.helper_registry.get_helper_descriptions(
            self.variant_config['helpers']
        )

        # Load with variant-specific template variables
        template = {
            'variant_persona': self.variant_config['persona'],
            'variant_workflow': self.variant_config.get('workflow', ''),
            'filtered_helpers': helper_descriptions,
            # ... other template vars
        }

        prompt_parts = PromptLoader.load(
            prompt_name,
            mode=effective_mode,
            template=template
        )

        return f"{prompt_parts['system_message']}\n\n{prompt_parts['user_message']}"
```

### 5. Runtime Registration

**File:** `llmvm2/server/runtime.py`

```python
class PythonRuntime:
    def __init__(self, active_helpers: Dict[str, object]):
        """
        Initialize runtime with filtered helpers.

        Args:
            active_helpers: Dict of allowed helpers for this variant
        """
        self.active_helpers = active_helpers
        self._init_runtime_with_helpers()

    def _init_runtime_with_helpers(self):
        """Inject only active helpers into runtime namespace."""
        for name, helper in self.active_helpers.items():
            # Add to Python globals so code can call them
            if "." in name:
                class_name, method = name.split(".", 1)
                # Create class instance if needed
                if class_name not in self.globals:
                    self.globals[class_name] = type(class_name, (), {})()
                setattr(self.globals[class_name], method, helper)
            else:
                self.globals[name] = helper
```

## Benefits of Variants Architecture

### ✅ Context Efficiency
- Data scientist prompt: ~100 helpers → ~10 helpers
- Saves thousands of tokens per request
- Faster responses, lower costs

### ✅ Focused Experience
- Model only sees relevant tools
- Clearer guidance on what to use
- Better tool selection

### ✅ Easy to Extend
- Add new variant by editing YAML config
- No code changes needed
- Share base prompt template

### ✅ Composable
- Variants define WHAT helpers
- Modes define HOW execution works
- Orthogonal concerns

### ✅ Backward Compatible
- Default variant = all helpers (current behavior)
- Existing prompts work unchanged

## Migration Path

### Phase 1: Infrastructure
1. Create `HelperRegistry` class
2. Create `variants.yaml` configuration
3. Add variant loading to `Orchestrator`
4. Update `PythonRuntime` to accept filtered helpers

### Phase 2: Template Updates
1. Extract variant-specific sections from prompts
2. Create base template with `{{variant_*}}` placeholders
3. Move persona/workflow to variant configs

### Phase 3: Variant Definitions
1. Define `default` variant (all helpers)
2. Define `data-scientist` variant (filtered)
3. Define `web-researcher` variant (filtered)
4. Define `finance-analyst` variant (filtered)

### Phase 4: Server Integration
1. Add `--variant` CLI argument to server
2. Update WebSocket to support variant selection
3. Add `/variant` slash command to client
4. Show active variant in client UI

## Example Usage

```bash
# Start data scientist variant
uv run python -m llmvm2.server --variant data-scientist

# Start web researcher variant
uv run python -m llmvm2.server --variant web-researcher

# Client can switch variants
/variant data-scientist
```

## Implementation Priority

**High Priority:**
1. ✅ Helper filtering infrastructure (HelperRegistry)
2. ✅ Variant configuration system (variants.yaml)
3. ✅ Orchestrator integration

**Medium Priority:**
1. Template extraction and placeholders
2. Define core variants (default, data-scientist, web-researcher)

**Low Priority:**
1. Runtime variant switching via `/variant` command
2. Variant-specific configuration (temperature, model selection)

## Open Questions

1. **Should variants be runtime-switchable or startup-only?**
   - Runtime switching = more flexible
   - Startup only = simpler, avoids helper registry changes mid-conversation

2. **Should mode and variant be combined or separate?**
   - Current proposal: separate (orthogonal)
   - Alternative: variant = mode + helper filter

3. **How to handle custom user-defined variants?**
   - Load from `~/.config/llmvm2/custom_variants.yaml`?
   - Override default variants?

4. **Should variants affect model selection?**
   - e.g., data-scientist defaults to o1-mini for reasoning?
   - Or keep model selection independent?

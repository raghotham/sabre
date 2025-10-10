# Mode System Implementation Plan

## Overview

Implement a runtime mode switching system for llmvm2 that allows users to change execution modes via `/mode` slash command instead of environment variables. Modes control which system prompts are used and how code execution is handled.

## Proposed Directory Structure

```
llmvm2/llmvm2/server/prompts/
├── modes/
│   ├── tools/                      # Default mode - full helper execution
│   │   ├── continuation_execution.prompt
│   │   └── error_correction.prompt
│   ├── direct/                     # Direct mode - no code execution
│   │   └── continuation_execution.prompt
│   ├── data_scientist/             # Specialized for data science
│   │   ├── continuation_execution.prompt
│   │   └── error_correction.prompt
│   └── reasoning/                  # Extended thinking mode
│       ├── continuation_execution.prompt
│       └── error_correction.prompt
├── shared/                         # Shared across all modes
│   ├── result_formatting/
│   │   ├── str_result.prompt
│   │   ├── list_result.prompt
│   │   ├── llm_call_result.prompt
│   │   └── function_call_result.prompt
│   ├── helpers/
│   │   ├── llm_call.prompt
│   │   ├── llm_bind_global.prompt
│   │   ├── llm_list_bind.prompt
│   │   ├── pandas_bind.prompt
│   │   └── coerce.prompt
│   ├── compilation/
│   │   └── thread_to_program.prompt
│   └── utility/
│       ├── search_ranker.prompt
│       ├── search_expander.prompt
│       ├── query_understanding.prompt
│       └── download_and_validate.prompt
```

## Mode Definitions

| Mode | Description | Main Prompt | Code Execution |
|------|-------------|-------------|----------------|
| `tools` | Default mode with full Python helpers | `modes/tools/continuation_execution.prompt` | ✅ Full |
| `direct` | Direct LLM conversation, no code | `modes/direct/continuation_execution.prompt` | ❌ None |
| `data_scientist` | Specialized for data analysis | `modes/data_scientist/continuation_execution.prompt` | ✅ Full |
| `reasoning` | Extended thinking with reasoning tokens | `modes/reasoning/continuation_execution.prompt` | ✅ Full |
| `program` | For compiled/saved programs (special) | N/A (uses compiled code) | ✅ Compiled only |

### Mode Characteristics

**tools** (default)
- Full access to Python runtime and all helpers
- Uses `<helpers>` blocks for code execution
- Error correction enabled
- Best for: General purpose tasks, automation, data processing

**direct**
- Pure conversational mode, no code execution
- Fastest response times
- No helper parsing or execution overhead
- Best for: Quick questions, brainstorming, explanations

**data_scientist**
- Optimized prompts for data analysis workflows
- Enhanced pandas, numpy, matplotlib support
- Specialized result formatting
- Best for: Data exploration, visualization, statistical analysis

**reasoning**
- Extended thinking mode with reasoning token support
- Uses `python_continuation_execution_reasoning_responses.prompt`
- More verbose explanations of reasoning process
- Automatically selected for models that don't respect stop tokens well:
  - OpenAI: o1, o3, o4, o5, gpt-5
  - Other: Llama, grok models
- Also used when reasoning tokens are explicitly requested
- Best for: Complex problem solving, debugging, algorithm design

**program**
- Special mode for compiled programs
- No LLM calls - uses pre-compiled Python functions
- Fastest execution for repeated tasks
- Best for: Production workflows, repeated operations

## Auto-Mode Selection

In addition to manual mode selection via `/mode`, the system can automatically select the best mode based on the model being used:

**Automatic Reasoning Mode Selection**:
When the executor detects certain models, it should automatically use reasoning mode prompts:
- OpenAI models: o1, o3, o4, o5, gpt-5 (models that don't respect stop tokens well)
- Other models: Llama, grok (models known to ignore stop tokens)

This logic should be implemented in the Orchestrator:

```python
def _should_use_reasoning_mode(self, model: str) -> bool:
    """
    Check if model should use reasoning mode prompts.

    Args:
        model: Model name

    Returns:
        True if reasoning mode should be used
    """
    reasoning_models = ["o1", "o3", "o4", "o5", "gpt-5", "llama", "grok"]
    model_lower = model.lower()
    return any(rm in model_lower for rm in reasoning_models)


def load_default_instructions(self, model: str = None) -> str:
    """Load default instructions, auto-selecting mode if needed."""
    # Auto-select reasoning mode for certain models
    if model and self._should_use_reasoning_mode(model) and self.mode == "tools":
        logger.info(f"Auto-selecting reasoning mode for model: {model}")
        effective_mode = "reasoning"
    else:
        effective_mode = self.mode

    prompt_name = "continuation_execution.prompt"
    prompt_parts = PromptLoader.load(prompt_name, mode=effective_mode, template={...})
    ...
```

## Implementation Components

### 1. Prompt Loader Updates

**File:** `llmvm2/common/utils/prompt_loader.py`

```python
class PromptLoader:
    @staticmethod
    def load(prompt_name: str, mode: str = "tools", template: dict = None) -> dict:
        """
        Load a prompt, checking mode-specific directory first, then shared.

        Resolution order:
        1. prompts/modes/{mode}/{prompt_name}
        2. prompts/shared/**/{prompt_name}
        3. prompts/{prompt_name} (backward compatibility)

        Args:
            prompt_name: Name of the prompt file (e.g., 'continuation_execution.prompt')
            mode: Execution mode ('tools', 'direct', 'data_scientist', 'reasoning')
            template: Optional template variables to substitute

        Returns:
            Dictionary with 'system_message', 'user_message', and 'templates' keys
        """
        # Try mode-specific first
        mode_path = f"modes/{mode}/{prompt_name}"
        if exists(mode_path):
            return load_prompt_file(mode_path, template)

        # Try shared directory (search recursively)
        shared_path = find_in_shared(prompt_name)
        if shared_path:
            return load_prompt_file(shared_path, template)

        # Fallback to root (backward compatibility)
        return load_prompt_file(prompt_name, template)

    @staticmethod
    def find_in_shared(prompt_name: str) -> Optional[str]:
        """Recursively search for prompt in shared/ directory."""
        shared_base = Path(__file__).parent.parent / "server" / "prompts" / "shared"
        for path in shared_base.rglob(prompt_name):
            return str(path.relative_to(shared_base.parent))
        # Return None if not found
        return None
```

### 2. Orchestrator Mode Support

**File:** `llmvm2/server/orchestrator.py`

```python
class Orchestrator:
    def __init__(
        self,
        executor: ResponseExecutor,
        runtime: PythonRuntime,
        mode: str = "tools",  # NEW
        model: str = None,  # NEW - for auto-mode selection
        event_callback: Callable[[Event], Awaitable[None]] = None,
    ):
        self.mode = mode
        self.model = model
        self.executor = executor
        self.runtime = runtime
        self.event_callback = event_callback

    def _should_use_reasoning_mode(self, model: str) -> bool:
        """Check if model should use reasoning mode prompts."""
        reasoning_models = ["o1", "o3", "o4", "o5", "gpt-5", "llama", "grok"]
        model_lower = model.lower()
        return any(rm in model_lower for rm in reasoning_models)

    def _get_effective_mode(self) -> str:
        """Get effective mode, considering auto-selection."""
        if (
            self.model
            and self._should_use_reasoning_mode(self.model)
            and self.mode == "tools"
        ):
            logger.info(f"Auto-selecting reasoning mode for model: {self.model}")
            return "reasoning"
        return self.mode

    async def orchestrate(
        self, user_message: str, instructions: str = None, **kwargs
    ) -> OrchestrationResult:
        # Get effective mode (may auto-select reasoning)
        effective_mode = self._get_effective_mode()

        # Load mode-specific continuation prompt
        if effective_mode == "direct":
            # Skip helper execution entirely for direct mode
            return await self._direct_response(user_message, instructions)
        else:
            # Use continuation execution for this mode
            prompt_name = "continuation_execution.prompt"
            system_prompt = PromptLoader.load(prompt_name, mode=effective_mode)

            return await self._continuation_loop(
                user_message, instructions or system_prompt["system_message"]
            )

    async def _direct_response(
        self, user_message: str, instructions: str = None
    ) -> OrchestrationResult:
        """Direct LLM response without helper execution."""
        prompt = PromptLoader.load("continuation_execution.prompt", mode="direct")

        # Single LLM call, no helper parsing
        response = await self.executor.execute(
            user_message, instructions=instructions or prompt["system_message"]
        )

        # Stream directly to client
        full_text = ""
        async for token in response:
            full_text += token
            if self.event_callback:
                await self.event_callback(ResponseTokenEvent(token=token))

        return OrchestrationResult(
            success=True,
            final_response=full_text,
            tree=None,  # No execution tree in direct mode
        )
```

### 3. Server API - Mode Management

**File:** `llmvm2/server/api/server.py`

```python
# Store mode per WebSocket connection
connection_modes: dict[str, str] = {}  # connection_id -> mode


@app.websocket("/message")
async def message_endpoint(websocket: WebSocket):
    await websocket.accept()
    connection_id = str(uuid.uuid4())
    current_mode = "tools"  # default
    connection_modes[connection_id] = current_mode

    logger.info(f"Client {connection_id} connected")

    try:
        async for data in websocket.iter_json():
            message_type = data.get("type")

            if message_type == "mode_change":
                # Handle mode change request
                new_mode = data.get("mode")
                if new_mode in ["tools", "direct", "data_scientist", "reasoning"]:
                    connection_modes[connection_id] = new_mode
                    logger.info(f"Client {connection_id} switched to mode: {new_mode}")

                    await websocket.send_json(
                        {
                            "type": "mode_changed",
                            "mode": new_mode,
                            "message": f"Switched to '{new_mode}' mode",
                        }
                    )
                else:
                    await websocket.send_json(
                        {"type": "error", "error": f"Unknown mode: {new_mode}"}
                    )

            elif message_type == "message":
                content = data.get("content", "")
                mode = connection_modes.get(connection_id, "tools")

                logger.info(f"Processing message in mode: {mode}")

                # Create orchestrator with current mode
                orchestrator = Orchestrator(
                    executor=executor,
                    runtime=runtime,
                    mode=mode,  # Pass mode to orchestrator
                    event_callback=create_event_sender(websocket),
                )

                result = await orchestrator.orchestrate(content)

                # Send completion
                await websocket.send_json(
                    {"type": "complete", "response": result.final_response}
                )

    finally:
        logger.info(f"Client {connection_id} disconnected")
        if connection_id in connection_modes:
            del connection_modes[connection_id]
```

### 4. Client - /mode Slash Command

**File:** `llmvm2/client/client.py`

```python
class Client:
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.ws: Optional[WebSocket] = None
        self.current_mode = "tools"
        self.available_modes = {
            "tools": "Default mode with full Python helper execution",
            "direct": "Direct LLM conversation without code execution",
            "data_scientist": "Specialized mode for data science tasks",
            "reasoning": "Extended thinking mode with reasoning tokens",
        }

    async def handle_slash_command(self, command: str) -> bool:
        """
        Handle slash commands. Returns True if handled, False otherwise.

        Args:
            command: The command string (e.g., '/mode tools')

        Returns:
            True if command was handled, False if not recognized
        """
        parts = command.strip().split()
        cmd = parts[0].lower()

        if cmd == "/mode":
            if len(parts) == 1:
                # Show current mode and available modes
                self.print(f"<ansigreen>Current mode: {self.current_mode}</ansigreen>")
                self.print("\n<ansicyan>Available modes:</ansicyan>")

                for mode, desc in self.available_modes.items():
                    marker = "→" if mode == self.current_mode else " "
                    self.print(f"  {marker} <ansiyellow>{mode}</ansiyellow>: {desc}")

                self.print("\n<ansigray>Usage: /mode [mode_name]</ansigray>")
                return True

            new_mode = parts[1].lower()
            if new_mode not in self.available_modes:
                self.print(f"<ansired>Error: Unknown mode '{new_mode}'</ansired>")
                self.print(f"Available: {', '.join(self.available_modes.keys())}")
                return True

            if new_mode == self.current_mode:
                self.print(f"<ansiyellow>Already in '{new_mode}' mode</ansiyellow>")
                return True

            # Send mode change to server
            await self.ws.send_json({"type": "mode_change", "mode": new_mode})

            # Wait for confirmation
            response = await self.ws.receive_json()
            if response.get("type") == "mode_changed":
                self.current_mode = new_mode
                self.print(f"<ansigreen>✓ Switched to '{new_mode}' mode</ansigreen>")
            elif response.get("type") == "error":
                self.print(f"<ansired>Error: {response.get('error')}</ansired>")

            return True

        return False

    async def process_input(self, user_input: str):
        """Process user input, checking for slash commands first."""
        if user_input.startswith("/"):
            handled = await self.handle_slash_command(user_input)
            if handled:
                return

        # Regular message - send to server
        await self.ws.send_json({"type": "message", "content": user_input})
```

### 5. Prompt File Migration

**Current → New mapping:**

```bash
# Mode-specific prompts
python_continuation_execution_responses.prompt
  → modes/tools/continuation_execution.prompt

python_continuation_execution_reasoning_responses.prompt
  → modes/reasoning/continuation_execution.prompt

data_scientist_expert.prompt
  → modes/data_scientist/continuation_execution.prompt

python_error_correction.prompt
  → modes/tools/error_correction.prompt
  → modes/data_scientist/error_correction.prompt (copy)
  → modes/reasoning/error_correction.prompt (copy)

# Create new direct mode prompt
(NEW) → modes/direct/continuation_execution.prompt

# Result formatting - move to shared
str_result.prompt → shared/result_formatting/str_result.prompt
list_result.prompt → shared/result_formatting/list_result.prompt
llm_call_result.prompt → shared/result_formatting/llm_call_result.prompt
function_call_result.prompt → shared/result_formatting/function_call_result.prompt
assistant_result.prompt → shared/result_formatting/assistant_result.prompt
foreach_result.prompt → shared/result_formatting/foreach_result.prompt
functionmeta_result.prompt → shared/result_formatting/functionmeta_result.prompt

# Helpers - move to shared
llm_call.prompt → shared/helpers/llm_call.prompt
llm_bind_global.prompt → shared/helpers/llm_bind_global.prompt
llm_list_bind.prompt → shared/helpers/llm_list_bind.prompt
pandas_bind.prompt → shared/helpers/pandas_bind.prompt
coerce.prompt → shared/helpers/coerce.prompt

# Compilation - move to shared
thread_to_program.prompt → shared/compilation/thread_to_program.prompt

# Utility - move to shared
search_ranker.prompt → shared/utility/search_ranker.prompt
search_expander.prompt → shared/utility/search_expander.prompt
search_classifier.prompt → shared/utility/search_classifier.prompt
search_location.prompt → shared/utility/search_location.prompt
query_understanding.prompt → shared/utility/query_understanding.prompt
download_and_validate.prompt → shared/utility/download_and_validate.prompt
document_chunk.prompt → shared/utility/document_chunk.prompt
python_code_insights.prompt → shared/utility/python_code_insights.prompt

# Other prompts that need categorization
answer.prompt → shared/utility/answer.prompt
answer_nocontext.prompt → shared/utility/answer_nocontext.prompt
answer_primitive.prompt → shared/utility/answer_primitive.prompt
answer_error_correction.prompt → shared/utility/answer_error_correction.prompt
answer_regen_code_or_rewrite.prompt → shared/utility/answer_regen_code_or_rewrite.prompt
tool_call.prompt → shared/utility/tool_call.prompt
tool_execution.prompt → shared/utility/tool_execution.prompt
python_tool_execution.prompt → shared/utility/python_tool_execution.prompt
map_reduce_map.prompt → shared/utility/map_reduce_map.prompt
map_reduce_reduce.prompt → shared/utility/map_reduce_reduce.prompt
```

### 6. Direct Mode Prompt

**File:** `modes/direct/continuation_execution.prompt`

```
[system_message]
You are Claude, a helpful AI assistant powered by Anthropic.

[user_message]
Respond naturally and helpfully to the user's request.

You do NOT have access to code execution or helper tools in this mode - use your knowledge
and reasoning capabilities to provide helpful, accurate responses.

Guidelines:
- Be concise yet thorough
- Provide examples when helpful
- Explain your reasoning when appropriate
- If you're uncertain, acknowledge it
- Use markdown formatting for clarity
- Be conversational and friendly

Respond directly to the user's query.
```

## Migration Steps

### Phase 1: Directory Structure Setup

1. Create new directory structure:
```bash
cd llmvm2/llmvm2/server/prompts
mkdir -p modes/{tools,direct,data_scientist,reasoning}
mkdir -p shared/{result_formatting,helpers,compilation,utility}
```

2. Copy prompts to new locations (keep originals for now):
```bash
# Mode-specific
cp python_continuation_execution_responses.prompt modes/tools/continuation_execution.prompt
cp python_continuation_execution_reasoning_responses.prompt modes/reasoning/continuation_execution.prompt
cp data_scientist_expert.prompt modes/data_scientist/continuation_execution.prompt
cp python_error_correction.prompt modes/tools/error_correction.prompt
cp python_error_correction.prompt modes/data_scientist/error_correction.prompt
cp python_error_correction.prompt modes/reasoning/error_correction.prompt

# Result formatting
cp *_result.prompt shared/result_formatting/

# Helpers
cp llm_call.prompt llm_bind_global.prompt llm_list_bind.prompt pandas_bind.prompt coerce.prompt shared/helpers/

# Compilation
cp thread_to_program.prompt shared/compilation/

# Utility
cp search_*.prompt query_understanding.prompt download_and_validate.prompt document_chunk.prompt python_code_insights.prompt shared/utility/
cp answer*.prompt tool_*.prompt map_reduce_*.prompt shared/utility/
```

3. Create direct mode prompt:
```bash
# Create modes/direct/continuation_execution.prompt with content above
```

### Phase 2: Code Updates

4. Update `PromptLoader` in `llmvm2/common/utils/prompt_loader.py`
   - Add mode parameter to `load()` method
   - Implement resolution order (mode → shared → root)
   - Add `find_in_shared()` helper method

5. Update `Orchestrator` in `llmvm2/server/orchestrator.py`
   - Add mode parameter to `__init__()`
   - Implement `_direct_response()` method
   - Update `orchestrate()` to use mode-specific prompts

6. Update server API in `llmvm2/server/api/server.py`
   - Add `connection_modes` dictionary
   - Handle `mode_change` message type
   - Pass mode to Orchestrator

7. Update client in `llmvm2/client/client.py`
   - Add `current_mode` and `available_modes` attributes
   - Implement `handle_slash_command()` method
   - Update `process_input()` to check for slash commands

### Phase 3: Testing

8. Test each mode independently:
   - `tools` mode: Verify helper execution works
   - `direct` mode: Verify no code execution, fast responses
   - `data_scientist` mode: Verify specialized prompts load
   - `reasoning` mode: Verify reasoning tokens work
   - Mode switching: Test `/mode` command

9. Test prompt fallback:
   - Mode-specific prompt exists → uses it
   - Mode-specific missing, shared exists → uses shared
   - Neither exists → uses root (backward compatibility)

10. Test edge cases:
    - Invalid mode name
    - Switching modes mid-conversation
    - Multiple concurrent connections with different modes

### Phase 4: Cleanup

11. Update all code that calls `PromptLoader.load()` to pass mode parameter

12. Remove old prompts from root directory after confirming migration works

13. Update documentation

## Benefits

✅ **Clear naming** - Mode is part of the path, making it obvious which prompts are used where
✅ **Easy to extend** - Add new modes by creating a new directory under `modes/`
✅ **Shared prompts** - No duplication for common utilities like result formatting
✅ **Backward compatible** - Fallback to root if prompt not found in new structure
✅ **Runtime switching** - Change modes mid-conversation without restarting
✅ **Self-documenting** - Directory structure explains organization
✅ **Mode isolation** - Each mode's prompts are clearly separated
✅ **Client control** - Users can switch modes via `/mode` command

## Testing Checklist

- [ ] Directory structure created
- [ ] Prompts copied to new locations
- [ ] Direct mode prompt created
- [ ] PromptLoader updated with mode support
- [ ] Orchestrator updated with mode parameter
- [ ] Server API handles mode_change messages
- [ ] Client implements /mode slash command
- [ ] Tools mode works (default behavior)
- [ ] Direct mode works (no code execution)
- [ ] Data scientist mode works (specialized prompts)
- [ ] Reasoning mode works (extended thinking)
- [ ] Auto-mode selection works for o1/o3/gpt-5 models
- [ ] Mode switching works mid-conversation
- [ ] Shared prompts resolve correctly
- [ ] Backward compatibility maintained
- [ ] Multiple concurrent connections work
- [ ] Documentation updated

## Future Enhancements

- [ ] Add mode indicator in client prompt (e.g., `[tools]>`)
- [ ] Mode-specific helper availability (e.g., direct mode has no helpers)
- [ ] Save mode preference per conversation
- [ ] Mode-specific configuration (e.g., temperature, max tokens)
- [ ] Auto-detect best mode based on user query (e.g., data science queries → data_scientist mode)
- [ ] Custom user-defined modes via config file
- [ ] Show mode indicator when auto-selected (e.g., "Using reasoning mode for o3-mini")

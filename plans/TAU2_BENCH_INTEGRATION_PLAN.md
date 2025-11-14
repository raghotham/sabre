# SABRE - tau2-bench Integration via MCP

**Depends on:** [MCP_INTEGRATION_PLAN.md](MCP_INTEGRATION_PLAN.md)

## Table of Contents

- [Problem Statement](#problem-statement)
- [Solution: MCP-Based tau2 Integration](#solution-mcp-based-tau2-integration)
  - [Key Insight](#key-insight)
- [Architecture Overview](#architecture-overview)
  - [Component Diagram](#component-diagram)
  - [Comparison: Rejected vs MCP Approach](#comparison-rejected-vs-mcp-approach)
- [How It Works](#how-it-works)
  - [Tool Call Flow](#tool-call-flow)
  - [Sequence Diagram](#sequence-diagram)
- [Implementation Components](#implementation-components)
  - [1. tau2-bench-mcp Server Adapter](#1-tau2-bench-mcp-server-adapter)
  - [2. SabreAgent (LocalAgent Implementation)](#2-sabreagent-localagent-implementation)
  - [3. Evaluation Harness Integration](#3-evaluation-harness-integration)
  - [4. Configuration](#4-configuration)
- [Implementation Roadmap](#implementation-roadmap)
  - [Prerequisites](#prerequisites)
  - [Phase 1: tau2-bench-mcp Server Modifications](#phase-1-tau2-bench-mcp-server-modifications)
  - [Phase 2: SabreAgent Implementation](#phase-2-sabreagent-implementation)
  - [Phase 3: Evaluation Harness Integration](#phase-3-evaluation-harness-integration)
  - [Phase 4: Testing and Validation](#phase-4-testing-and-validation)
- [Benefits](#benefits)
- [Testing Strategy](#testing-strategy)
- [Example Usage](#example-usage)
- [Troubleshooting](#troubleshooting)
- [Future Enhancements](#future-enhancements)

## Problem Statement

**Goal**: Evaluate SABRE agent against tau2-bench benchmark tasks.

**tau2-bench Requirements**:
- Agent implements `LocalAgent[AgentState]` interface
- Agent receives task description and available tool schemas
- Agent returns tool call **requests** (not executions)
- tau2-bench controls tool execution in simulation environment
- tau2-bench measures success, efficiency, policy compliance

**Rejected Approach** (from `sabre_tau2_benckmark` branch):
- Added `ToolRegistry` with "internal" and "external" tools
- Modified `Orchestrator` to pause execution on external tool calls
- Saved execution state for resumption after tool results
- Required significant orchestrator changes
- **Status**: Not approved due to orchestrator complexity

**Challenge**: How to integrate with tau2-bench without modifying SABRE's orchestrator?

## Solution: MCP-Based tau2 Integration

Use the MCP integration (from `MCP_INTEGRATION_PLAN.md`) to route tau2 tool calls through an MCP server. The orchestrator runs normally - it just sees MCP tools like any other external service.

### Key Insight

**tau2 tools ARE external tools** - just like Postgres or GitHub tools. Instead of special-casing them with pause/resume logic, expose them via MCP:

1. **tau2-bench-mcp server** exposes domain tools (search_flights, book_flight, etc.)
2. **SABRE agent** connects to this MCP server at initialization
3. **MCP tools** are injected into Python runtime namespace
4. **LLM calls tools** normally in `<helpers>` blocks
5. **MCP client routes** calls to tau2-bench-mcp server
6. **tau2 simulation** executes tools and returns results
7. **Orchestrator continues** with results - no pause/resume needed!

## Architecture Overview

### Component Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                  tau2-bench Evaluation Harness                   │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  1. Start tau2-bench-mcp server for domain                 │  │
│  │  2. Create SabreAgent with MCP connection                  │  │
│  │  3. Run task turns, measure success                        │  │
│  └────────────────────────────────────────────────────────────┘  │
└────────────────┬───────────────────────────────────┬─────────────┘
                 │                                   │
                 │ initialize                        │ spawn
                 ▼                                   ▼
┌─────────────────────────────────┐   ┌──────────────────────────────┐
│      SabreAgent                 │   │  tau2-bench-mcp Server       │
│  (implements LocalAgent)        │   │                              │
│                                 │   │  • Wraps tau2 domain         │
│  • Connects to tau2-mcp at init │   │  • Exposes tools via MCP     │
│  • No pause/resume logic        │◄──┤  • Executes in simulation    │
│  • Just calls orchestrator      │   │  • Returns results           │
└────────────────┬────────────────┘   └──────────────────────────────┘
                 │                                   ▲
                 │ uses                              │
                 ▼                                   │
┌─────────────────────────────────┐                  │
│    SABRE Orchestrator           │                  │
│    (UNCHANGED!)                 │                  │
│                                 │                  │
│  • Runs continuation loop       │                  │
│  • Executes <helpers> blocks    │                  │
│  • Calls tools in namespace     │                  │
└────────────────┬────────────────┘                  │
                 │                                   │
                 │ tool call detected                │
                 ▼                                   │
┌─────────────────────────────────┐                  │
│    MCPHelperAdapter             │                  │
│    (from MCP_INTEGRATION_PLAN)  │                  │
│                                 │                  │
│  • Routes Tau2.* calls to MCP   │                  │
│  • Transforms args/results      │                  │
└────────────────┬────────────────┘                  │
                 │                                   │
                 │ JSON-RPC over stdio               │
                 └───────────────────────────────────┘
```

## How It Works

### Tool Call Flow

```
1. User: "Search for flights from SFO to JFK"

2. LLM generates response with <helpers> block containing:
   - Call to Tau2.search_flights with origin and destination
   - Print statement to display results

3. Orchestrator parses <helpers> block
   → Python Runtime executes code

4. Runtime encounters: Tau2.search_flights(...)
   → Tau2 is an object in namespace (injected by MCP adapter)
   → search_flights is a callable

5. Callable routes to MCPHelperAdapter
   → Adapter invokes tool on tau2 MCP server

6. MCPClient sends JSON-RPC request with:
   - Method: "tools/call"
   - Tool name: "search_flights"
   - Arguments: origin="SFO", destination="JFK"

7. tau2-bench-mcp server receives request
   → Executes in tau2 simulation environment
   → Returns flights data

8. MCPClient receives JSON-RPC response containing:
   - Result content with flight data as text

9. MCPHelperAdapter transforms MCP content to Python value
   → Returns to runtime

10. Runtime continues execution:
    - Assigns flights variable
    - Executes print statement
    - Captures output

11. Orchestrator captures output
    → Builds <helpers_result> tag
    → Continues to next LLM call

12. LLM sees result, generates final response to user
```

### Sequence Diagram

```
tau2-harness  SabreAgent  Orchestrator  Runtime  MCPAdapter  tau2-mcp  tau2-sim
     │            │            │           │          │          │         │
     │─init──────▶│            │           │          │          │         │
     │            │─connect───────────────────────────▶│         │         │
     │            │            │           │          │─spawn───▶│         │
     │            │◀──────────────────────────────────────ready──│         │
     │            │            │           │          │          │         │
     │─task msg──▶│            │           │          │          │         │
     │            │─run───────▶│           │          │          │         │
     │            │            │─execute──▶│          │          │         │
     │            │            │           │          │          │         │
     │            │            │     LLM generates <helpers>     │         │
     │            │            │◀─response─│          │          │         │
     │            │            │           │          │          │         │
     │            │            │─exec code─▶│         │          │         │
     │            │            │           │─Tau2.*──▶│          │         │
     │            │            │           │          │─RPC─────▶│         │
     │            │            │           │          │          │─exec───▶│
     │            │            │           │          │          │◀────────│
     │            │            │           │          │◀─result──│         │
     │            │            │           │◀─────────│          │         │
     │            │            │◀──result──│          │          │         │
     │            │            │           │          │          │         │
     │            │            │     Builds <helpers_result>     │         │
     │            │            │─continue─▶│          │          │         │
     │            │            │           │          │          │         │
     │            │            │     LLM sees result, responds   │         │
     │            │            │◀─final────│          │          │         │
     │            │◀───result──│           │          │          │         │
     │◀─response──│            │           │          │          │         │
```

## Implementation Components

### 1. tau2-bench-mcp Server Adapter

**Location**: `tau2-bench-mcp/src/tau2_mcp/evaluation_mode.py` (new file)

The existing tau2-bench-mcp server needs an "evaluation mode" for integration. This involves:

**EvaluationMCPServer Class**:
- Takes an existing tau2 domain instance managed by the harness
- Single session lifecycle (no session management needed)
- Synchronous tool execution
- Detailed logging for debugging

**Key Methods**:
- **list_tools**: Returns all domain tools converted to MCP Tool schema format
- **call_tool**: Executes tools in the tau2 domain environment and converts results to MCP content format
- **convert_tool_to_mcp**: Transforms tau2 Tool objects to MCP Tool schema
- **convert_result_to_mcp**: Converts tau2 tool results to MCP TextContent

**CLI Wrapper** (`tau2-bench-mcp/src/tau2_mcp/cli.py`):
- Add command-line flags for evaluation mode
- Accept domain name and session ID parameters
- Load the specified domain and start the evaluation server
- Run in stdio mode for MCP communication

### 2. SabreAgent (LocalAgent Implementation)

**Location**: `sabre/benchmarks/tau2/sabre_agent.py` (new file)

**SabreAgent Class**:
Implements tau2-bench's `LocalAgent` interface while using SABRE's orchestration engine through MCP.

**Key Differences from Rejected Approach**:
- No ToolRegistry needed
- No pause/resume orchestrator logic
- Uses standard MCP integration
- Orchestrator remains completely unchanged

**Initialization**:
- Takes tau2 tools, domain policy, and LLM model
- Initializes standard SABRE components (Runtime, Executor, Orchestrator)
- MCP connection established per evaluation session

**MCP Connection Method**:
- Called at start of each evaluation session
- Initializes MCPClientManager
- Configures tau2-mcp server with stdio transport
- Connects to the spawned server
- Discovers available tools via MCP
- Injects tools into runtime namespace under "Tau2" prefix

**System Prompt Building**:
- Combines domain policy with tool usage instructions
- Documents available tools
- Provides examples of calling tools in helpers blocks
- Emphasizes authentication and policy compliance rules

**Message Generation**:
- Receives user or tool messages from tau2-bench
- Converts to SABRE format
- Runs orchestrator normally (no special logic!)
- Converts orchestration result back to tau2 AssistantMessage
- Updates state and returns to tau2-bench

**Message Conversion Utilities**:
- Convert UserMessage to plain text
- Convert ToolMessage to formatted tool result
- Convert MultiToolMessage to formatted list of results
- Convert orchestration result to AssistantMessage

### 3. Evaluation Harness Integration

**Location**: `sabre/benchmarks/tau2/run_evaluation.py` (new file)

**Main Evaluation Function**:
- Takes domain name, number of tasks, model, and MCP command
- Loads the specified tau2 domain
- Initializes SABRE agent with domain tools and policy
- Iterates through tasks
- Runs each task with the agent
- Collects and saves results

**Result Collection**:
- Saves results to JSON file in specified output directory
- Logs progress and success/failure for each task
- Prints summary statistics

**CLI Entry Point**:
- Accepts command-line arguments for domain, tasks, model, MCP path, and output directory
- Validates arguments
- Calls main evaluation function
- Returns exit code based on success

### 4. Configuration

**Location**: `sabre/benchmarks/tau2/config.yaml`

**MCP Server Configuration**:
- Command to start tau2-mcp server
- Can use installed command, absolute path, or uv run for development
- Arguments for evaluation mode

**SABRE Agent Settings**:
- Default model selection
- Maximum iterations for orchestrator
- Timeout settings

**Evaluation Settings**:
- Output directory for results
- Trace saving preferences
- Verbosity level

## Implementation Roadmap

### Prerequisites

**Must be completed first:**
- ✅ MCP Integration (Phase 1-3 from `MCP_INTEGRATION_PLAN.md`)
  - MCPClient
  - MCPClientManager
  - MCPHelperAdapter
  - Runtime integration

### Phase 1: tau2-bench-mcp Server Modifications

**Goal**: Add evaluation mode to tau2-bench-mcp server

**Tasks:**
1. Create evaluation mode module in tau2-bench-mcp
2. Implement EvaluationMCPServer class with domain wrapping
3. Add CLI flags for evaluation mode with domain and session parameters
4. Test server in isolation with mock agent

**Testing Approach**:
- Start server in evaluation mode with retail domain
- Use MCP test client to verify tool discovery
- Execute sample tools and verify results

**Deliverables:**
- Evaluation mode module implementation
- Updated CLI with evaluation mode support
- Tests for evaluation mode functionality

### Phase 2: SabreAgent Implementation

**Goal**: Implement LocalAgent interface using MCP

**Tasks:**
1. Create SABRE benchmarks tau2 directory structure
2. Implement SabreAgent class with LocalAgent interface
3. Add message conversion utilities between tau2 and SABRE formats
4. Add tool namespace injection for Tau2 prefix
5. Write comprehensive unit tests

**Testing Approach**:
- Create agent with mock tools and verify initialization
- Test MCP connection establishment
- Simulate turn-by-turn message processing
- Verify tool namespace injection

**Deliverables:**
- SabreAgent implementation
- Unit test suite
- Documentation for agent usage

### Phase 3: Evaluation Harness Integration

**Goal**: Integrate with tau2-bench evaluation framework

**Tasks:**
1. Create evaluation runner script
2. Add tau2-bench CLI integration
3. Implement result collection and reporting
4. Add configuration files for MCP and agent settings

**Testing Approach**:
- Run single task from retail domain
- Run batch of multiple tasks
- Verify result collection and JSON output
- Test with different models

**Deliverables:**
- Evaluation runner implementation
- Configuration file
- Integration tests
- Usage documentation

### Phase 4: Testing and Validation

**Goal**: Comprehensive testing and benchmarking

**Tasks:**
1. End-to-end testing with all tau2 domains
2. Performance benchmarking and optimization
3. Error handling and recovery validation
4. Documentation and usage examples

**Test Scenarios:**
- Single task execution
- Multi-task batch evaluation
- Error recovery (server crash, timeout, invalid responses)
- Different models (gpt-4o, gpt-4o-mini)
- All domains (retail, airline, telecom)

**Deliverables:**
- Complete test suite covering all scenarios
- Benchmark results and performance analysis
- Troubleshooting guide
- Example notebooks demonstrating usage

## Testing Strategy

### Unit Tests

**SabreAgent Tests** will verify:
- Agent initialization with correct model and settings
- MCP server connection establishment
- Message format conversion between tau2 and SABRE
- Tool namespace injection into runtime
- State management across turns

### Integration Tests

**End-to-End Tests** will verify:
- Single task execution from start to finish
- Multi-turn conversations with tool calls
- Tool execution routing through MCP
- Result collection and formatting
- Error handling and recovery

### Performance Tests

**Benchmark Suite** will measure:
- Time to complete single task
- Throughput for batch evaluations
- Average time per task
- Resource utilization
- MCP communication overhead

## Example Usage

### Quick Test (1 Task)

To run a single task test:
1. Navigate to SABRE directory
2. Set OPENAI_API_KEY and TAU2_DATA_DIR environment variables
3. Run evaluation script with retail domain, 1 task, and gpt-4o-mini model
4. Check results in output directory

### Full Domain Evaluation

To run complete domain evaluation:
1. Run evaluation with 50 retail tasks using gpt-4o model
2. Save results to full_eval directory
3. Can also run with custom tau2-mcp path if not in system PATH
4. Results saved as JSON files

### Compare Models

To compare different models:
1. Loop through model list (gpt-4o-mini, gpt-4o, claude-3-sonnet)
2. Run evaluation for each model with same task set
3. Save results to model_comparison directory
4. Analyze results with comparison script

### Programmatic Usage

For custom evaluation scripts:
1. Import SabreAgent and tau2 domain utilities
2. Get domain and initialize agent with tools and policy
3. Iterate through tasks
4. Run each task and collect results
5. Print or save results as needed

## Troubleshooting

### MCP Server Not Starting

**Symptom**: Connection refused or tau2-mcp command not found

**Solutions**:
- Verify tau2-mcp is installed and in PATH
- Use absolute path to tau2-mcp binary
- Use uv run with directory specification for development mode

### Tool Calls Not Routing to MCP

**Symptom**: Tool not found errors or no MCP communication

**Solutions**:
- Verify MCP manager is initialized (not None)
- Check Tau2 namespace exists in runtime
- Verify tool attributes are accessible on Tau2 object
- Enable debug logging for MCP components

### Orchestrator Timeout

**Symptom**: Tasks timeout after default 2 minute limit

**Solutions**:
- Increase max_iterations on orchestrator
- Adjust timeout setting in executor
- Check if tasks are getting stuck in loops

### tau2 Tool Execution Errors

**Symptom**: MCP returns errors from tau2 simulation

**Solutions**:
- Check tau2-mcp server logs for details
- Test tool directly in tau2 domain without MCP
- Verify tool arguments match expected schema
- Check domain state is valid for operation

### Results Not Saving

**Symptom**: No output files created after evaluation

**Solutions**:
- Ensure output directory exists and has write permissions
- Check directory permissions
- Use absolute path for output directory
- Verify disk space is available

## Future Enhancements

### Multi-Turn Tool Calls

Support complex multi-turn interactions where multiple tool calls are needed across conversation turns. The LLM can chain operations: first find user, then get their orders, then process exchange request.

### Resource Support

Use MCP resources to read domain state directly without tool calls. This enables reading database state, current session info, or domain metadata via MCP resource URIs.

### Parallel Evaluations

Run multiple evaluations concurrently with worker processes. This dramatically speeds up large-scale evaluations across multiple domains or model configurations.

### Interactive Debugging

Step-by-step debugging of evaluation runs with breakpoints. Pause execution on tool calls, inspect state, and continue. Useful for understanding agent behavior and diagnosing issues.

### Persona Integration

Create tau2-specific personas that feature domain tools prominently. Personas provide example workflows for common task patterns and emphasize relevant tools for each domain.

### Benchmark Dashboard

Web UI for viewing and analyzing results. Display success rates by domain, compare models side-by-side, inspect individual task traces with tool call details, and export reports.

## Summary

This plan provides a **clean, maintainable approach** to integrating SABRE with tau2-bench evaluation:

1. **Leverages MCP integration** from `MCP_INTEGRATION_PLAN.md`
2. **No orchestrator changes** - runs standard continuation loop
3. **tau2 tools via MCP** - treated like any external service
4. **Simple implementation** - approximately 500 lines of new code
5. **Well-tested** - comprehensive test coverage
6. **Future-proof** - works with any MCP server

The key insight: **tau2 tools are external tools, and MCP is designed for exactly this use case.**

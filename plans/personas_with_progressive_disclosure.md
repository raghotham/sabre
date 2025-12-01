# Personas with Progressive Disclosure

## Executive Summary

This proposal combines:
- **Directory-based structure** from Claude Agent Skills
- **Progressive disclosure** to load context on-demand

**Result**: Base prompt of ~1,800 tokens (vs 12,000 current), expanding only when needed.

### Problem Statement

Current SABRE prompts load all helper signatures and persona context upfront, consuming ~12,000 tokens regardless of task complexity. A simple "What's a DaemonSet?" query pays the same token cost as a complex multi-step incident investigation. This is inefficient and limits the number of helpers and workflows we can support.

### Solution Overview

Progressive disclosure defers loading detailed context until the LLM explicitly requests it. The base prompt contains only an **index** of available capabilities. When the LLM needs specific workflow steps or helper signatures, it calls discovery helpers that inject the relevant content into the conversation. This approach mirrors how humans work—we don't memorize entire runbooks; we look up specific procedures when needed.

---

## Progressive Disclosure Model

The system operates on two levels of context loading:

**Level 1 (Always Loaded)**: The minimal context required for the LLM to understand what capabilities exist and how to request more detail. This includes the execution flow, a one-line persona description, category names, workflow names, and instructions for expanding context.

**Level 2 (On-Demand)**: Detailed content loaded only when explicitly requested. This includes full persona identity with principles and guidelines, specific workflow procedures, and helper function signatures with documentation.

```
LEVEL 1: Always Loaded (~1,800 tokens)
├── Base execution flow
├── Persona summary (1-line)
├── Helper categories (names only)
├── Workflow index (names only)
└── Discovery instructions

LEVEL 2: On-Demand
├── Full persona identity (~400 tokens)
├── Specific workflow (~300-500 tokens each)
└── Helper signatures (~400 tokens per category)
```

This separation ensures the LLM always knows *what* is available without paying the token cost of *how* until needed.

---

## Directory Structure

Each persona lives in its own directory under `~/.config/sabre/personas/`. This structure provides:

- **Modularity**: Each persona is self-contained and can be versioned independently
- **Extensibility**: Users can add custom personas without modifying SABRE core
- **Organization**: Workflows are grouped logically within each persona
- **Discoverability**: Standard file names make it easy to find and edit persona definitions

```
~/.config/sabre/personas/kubernetes-sre/
├── PERSONA.md              # Identity + indices
├── helpers.yaml            # Categories + filters
└── workflows/
    ├── incident-response.md
    ├── debugging.md
    ├── deployment.md
    ├── scaling.md
    ├── fix-crashloop.md
    ├── fix-oom-killed.md
    └── fix-high-cpu.md
```

**PERSONA.md**: The main persona definition file containing YAML frontmatter (metadata, workflow index, helper filters) and markdown body (identity, principles, guidelines).

**helpers.yaml**: Defines helper categories and filtering rules. Categories group related helpers for discovery, while include/exclude patterns control which helpers are available to this persona.

**workflows/**: Directory containing individual workflow files. Each workflow is a standalone markdown document describing a specific procedure or troubleshooting guide.

---

## File Formats

This section details the format of each file type in the persona directory.

### PERSONA.md

The PERSONA.md file uses a hybrid format: YAML frontmatter for structured metadata followed by markdown for human-readable content. The frontmatter is parsed at load time to build the workflow index and helper filters. The markdown body is loaded on-demand via `get_persona_context()`.

```markdown
---
name: kubernetes-sre
description: Kubernetes operations, incidents, and reliability

helper_categories:
  k8s: Kubectl and cluster operations
  monitoring: Prometheus, Grafana, alerting
  cloud: AWS/GCP infrastructure commands
  bash: Shell command execution

workflows:
  incident-response: Incident triage and mitigation
  debugging: Pod and service troubleshooting
  deployment: Release and rollout management
  scaling: Capacity planning and autoscaling
  fix-crashloop: Resolve CrashLoopBackOff issues
  fix-oom-killed: Troubleshoot out of memory errors
  fix-high-cpu: Diagnose high CPU usage

include_helpers: ["Bash.*", "K8s.*", "Prometheus.*"]
exclude_helpers: ["Database.*", "Browser.*"]
---

# Kubernetes SRE

Expert in Kubernetes operations, incident response, and
site reliability engineering.

## Core Principles
1. Mitigate first, diagnose second
2. Document actions during incidents
3. Automate repetitive operations
4. Monitor everything, alert judiciously
```

### workflows/incident-response.md

Workflow files are pure markdown documents. They should be structured with clear steps, commands, and decision points. The format is flexible, but consistent structure helps the LLM follow procedures reliably.

```markdown
# Incident Response

## 1. Assess Severity
- Check affected services and user impact
- Determine blast radius
- Classify: P1 (critical) / P2 (major) / P3 (minor)

## 2. Mitigate
- Rollback recent deployments if suspected
- Scale up resources if capacity issue
- Failover to healthy replicas/regions

## 3. Diagnose
- Check pod status: `kubectl get pods`
- Review logs: `kubectl logs `
- Check events: `kubectl get events`

## 4. Resolve & Document
- Apply fix
- Verify recovery
- Update incident timeline
```

### workflows/fix-crashloop.md

This example shows a troubleshooting workflow for a specific issue. These targeted workflows help the LLM provide precise, actionable guidance rather than generic advice.

```markdown
# Fix CrashLoopBackOff

## Symptoms
- Pod repeatedly restarting
- Status shows CrashLoopBackOff

## Diagnostic Steps
1. Check exit code: `kubectl describe pod `
2. Get previous logs: `kubectl logs  --previous`
3. Check resource limits
4. Verify config/secrets mounted correctly

## Common Causes
- Application error on startup
- Missing environment variables
- Resource limits too low
- Liveness probe misconfigured

## Resolution
- Fix application code/config
- Adjust resource limits
- Update probe settings
```

### helpers.yaml

The helpers.yaml file controls which helpers are available to the persona and how they're organized into categories. Categories serve two purposes:

1. **Discovery**: The LLM can load all helpers in a category with `get_helpers("category_name")`
2. **Documentation**: Category descriptions appear in the base prompt to guide the LLM

The `include` and `exclude` patterns use glob-style matching against helper names. This allows precise control over which helpers a persona can access—a data-scientist persona might exclude Bash for safety, while an SRE persona would include it.

```yaml
categories:
  k8s:
    description: Kubectl and cluster operations
    patterns: ["K8s.*", "Kubectl.*"]
  monitoring:
    description: Prometheus and alerting
    patterns: ["Prometheus.*", "Grafana.*"]
  cloud:
    description: Cloud infrastructure
    patterns: ["AWS.*", "GCP.*"]
  bash:
    description: Shell command execution
    patterns: ["Bash.*"]

include: ["K8s.*", "Bash.*", "Prometheus.*"]
exclude: ["Database.*", "Browser.*"]
```

---

## Base Prompt (Level 1)

The base prompt is what the LLM sees at the start of every conversation. It contains just enough context for the LLM to:

1. Understand its role (one-line persona description)
2. Know how to execute code (basic `<helpers>` syntax)
3. See what capabilities exist (category and workflow indices)
4. Know how to load more context (discovery helper instructions)

This is approximately ~1,800 tokens—a fraction of the current ~12,000 token prompts:

```
# Persona: Kubernetes SRE
Kubernetes operations, incidents, and reliability.

## Execution
Code in <helpers> blocks.
...
...

## Helper Categories
- k8s: Kubectl and cluster operations
- monitoring: Prometheus and alerting
- cloud: Cloud infrastructure
- bash: Shell command execution

## Workflows
- incident-response: Incident triage and mitigation
- debugging: Pod and service troubleshooting
- deployment: Release and rollout management
- scaling: Capacity planning and autoscaling
- fix-crashloop: Resolve CrashLoopBackOff issues
- fix-oom-killed: Troubleshoot out of memory errors
- fix-high-cpu: Diagnose high CPU usage

## Expanding Context
  get_persona_context()        → Load full identity
  get_workflow("name")         → Load specific workflow
  get_helpers("category")      → Load helper signatures

## Always Available
  result(value), answer(value)
```

---

## Discovery Helpers

Discovery helpers are special functions that inject additional context into the conversation. Unlike regular helpers that perform actions, these modify the prompt context for subsequent LLM reasoning.

**How it works**: When the LLM calls a discovery helper, the orchestrator intercepts the result and appends it to the conversation context. The LLM then continues with the expanded context available.

| Helper | Loads | Tokens |
| --- | --- | --- |
| `get_persona_context()` | Full identity, principles, guidelines | +400 |
| `get_workflow("name")` | Specific workflow procedures | +300-500 |
| `get_helpers("category")` | Helper signatures with documentation | +400 |

**Design principle**: The LLM should only load context it actually needs. For simple factual questions, no expansion is needed. For complex tasks, the LLM loads relevant workflows and helpers progressively.

---

## Usage Examples

These examples demonstrate how token consumption scales with task complexity. Simple tasks remain cheap, while complex tasks pay only for the context they actually use.

### Simple Query (No Expansion)

**Tokens**: 1,800

```
User: What's a DaemonSet?
Assistant: A DaemonSet ensures a pod runs on every node...
```

No workflows or helpers needed → base cost only. The LLM answers from its training knowledge without loading any additional context. This is the ideal case for factual questions where the persona's specialized workflows aren't needed.

---

### Incident Response

**Tokens**: 1,800 + 400 + 400 = 2,600

This example shows the two-phase pattern: first load context, then act. The LLM recognizes this as an incident from the urgency and symptoms, loads the appropriate workflow and helpers, then follows the structured procedure.

```
User: Production is down! Pods failing in payments service.

<helpers>
# Phase 1: Load context for incident handling
workflow = get_workflow("incident-response")
helpers = get_helpers("k8s")
</helpers>

<helpers_result>
[Incident workflow + K8s.* signatures injected into context]
</helpers_result>

<helpers>
# Phase 2: Execute workflow steps
# Following workflow: Assess → Mitigate → Diagnose
status = K8s.get_pods(namespace="payments")
events = K8s.get_events(namespace="payments")
answer(f"Pod status:\n{status}\n\nRecent events:\n{events}")
</helpers>
```

The workflow guides the LLM through a structured response: assess severity first, then mitigate, then diagnose. Without this workflow loaded, the LLM might skip straight to diagnosis, potentially leaving a production issue unmitigated.

---

### Specific Issue

**Tokens**: 1,800 + 400 + 400 = 2,600

When the user mentions a specific error or symptom, the LLM can load a targeted troubleshooting workflow. These issue-specific workflows contain expert knowledge about common causes and resolution steps.

```
User: Pod is in CrashLoopBackOff, how do I fix it?

<helpers>
# Load the specific troubleshooting guide for this error
workflow = get_workflow("fix-crashloop")
helpers = get_helpers("k8s")
</helpers>

<helpers_result>
[CrashLoopBackOff workflow + K8s.* signatures injected]
</helpers_result>

Following the workflow:
1. First, let's check the exit code...
```

The targeted workflow ensures the LLM doesn't miss common causes (like liveness probe misconfiguration) that it might overlook without the structured checklist.

---

### Complex Investigation

**Tokens**: 1,800 + 400 + 400 + 400 + 400 = 3,400

For complex investigations that span multiple domains, the LLM can load multiple workflows and the full persona context. This is the most expensive case, but still ~75% cheaper than loading everything upfront.

```
User: Why did API latency spike during last deployment?

<helpers>
# Complex investigation requires multiple context sources
ctx = get_persona_context()      # Full SRE principles
wf1 = get_workflow("debugging")   # General troubleshooting
wf2 = get_workflow("deployment")  # Deployment-specific checks
helpers = get_helpers("monitoring") # Prometheus/Grafana helpers
</helpers>
```

The LLM recognizes this as a cross-cutting issue (deployment + performance) and loads relevant workflows from both domains. The full persona context provides guiding principles for how to approach the investigation.

---

## Summary

### Benefits

Progressive disclosure provides:

- **Minimal base**: ~1,800 tokens with workflow index—enough to know what's available, not enough to bloat simple queries
- **On-demand loading**: Specific workflows load when needed—pay only for what you use
- **Scalability**: Many workflows without base prompt bloat—add 50 workflows without affecting base cost
- **Better focus**: Loaded context is relevant to the current task, reducing noise and improving response quality

### Token Savings

| Query Type | Current | Progressive | Savings |
| --- | --- | --- | --- |
| Simple factual | 12,000 | 1,800 | 85% |
| Single workflow | 12,000 | 2,600 | 78% |
| Complex multi-workflow | 12,000 | 3,400 | 72% |

### Trade-offs

- **Extra round-trip**: Loading context requires an additional `<helpers>` execution before acting. For most tasks this is negligible, but it does add latency.
- **LLM judgment**: The LLM must correctly identify which workflows to load. If it loads the wrong workflow or skips loading when needed, quality suffers. Good workflow naming and descriptions mitigate this.
- **Implementation complexity**: The orchestrator must handle discovery helpers specially, injecting content into context rather than returning results normally.

### Conclusion

Progressive disclosure transforms SABRE's token economics from O(all_content) to O(used_content). For a persona with many specialized workflows, this enables rich expert knowledge without prohibitive costs. The approach is particularly valuable as personas grow more sophisticated—each new workflow adds zero cost to queries that don't need it.

# k8s-ai-bench Benchmark Setup for SABRE

This directory contains configurations and scripts for running k8s-ai-bench benchmarks to compare direct LLM performance vs SABRE's agentic approach with MCP tools.

## Prerequisites

1. **kubectl-ai binary** - The k8s-ai-bench harness and kubectl-ai CLI
2. **Kubernetes cluster** - A working k8s cluster (kind, minikube, or remote)
3. **SABRE server** - Running with kubectl-ai MCP server configured

## Architecture

```
sabre/
├── benchmarks/k8s-ai-bench/
│   ├── README.md                    # This file
│   ├── setup_benchmark.py           # Setup script (prereqs + MCP config)
│   ├── run_benchmark.sh             # Main benchmark runner script
│   ├── analyze_results.sh           # Results analyzer
│   ├── configs/
│   │   └── mcp.yaml                 # MCP server configuration template
│   └── results/                     # Benchmark results storage
│       └── <run-name>/              # Individual run results
```

## Setup

### 1. Install kubectl-ai

```bash
# Clone kubectl-ai repository
git clone https://github.com/kubernetes-ai/kubectl-ai.git
cd kubectl-ai

# Build kubectl-ai binary
go build -o kubectl-ai ./cmd

# Install to PATH
sudo mv kubectl-ai /usr/local/bin/
```

### 2. Build k8s-ai-bench and Set Environment Variable

```bash
cd /path/to/kubectl-ai/k8s-ai-bench
go build -o k8s-ai-bench .

# REQUIRED: Set the K8S_AI_BENCH_DIR environment variable
export K8S_AI_BENCH_DIR=/path/to/kubectl-ai/k8s-ai-bench

# Add to your shell profile (~/.bashrc, ~/.zshrc) for persistence:
echo 'export K8S_AI_BENCH_DIR=/path/to/kubectl-ai/k8s-ai-bench' >> ~/.zshrc
```

### 3. Configure MCP Server for SABRE

Create `~/.config/sabre/mcp.yaml`:

```yaml
# SABRE MCP Server Configuration for kubectl-ai
mcp_servers:
  kubectl-ai:
    type: stdio
    command: kubectl-ai
    args: ["--mcp-server"]
    enabled: true
    timeout: 60
```

Or use our setup script:
```bash
uv run python setup_benchmark.py
```

### 4. Verify Setup

```bash
# Check prerequisites only
uv run python setup_benchmark.py --check

# Setup MCP config only
uv run python setup_benchmark.py --setup-mcp

# Force overwrite existing MCP config
uv run python setup_benchmark.py --force

# Verify MCP servers are configured
uv run sabre list

# Start SABRE server
uv run sabre-server

# In SABRE client, verify tools are available:
# - kubectl_ai.bash - Execute bash commands
# - kubectl_ai.kubectl - Execute kubectl commands
```

## Running Benchmarks

### Quick Start

```bash
# Run all tasks with SABRE
./run_benchmark.sh

# Run specific task pattern
./run_benchmark.sh --task-pattern "create-pod"

# Run with specific model
./run_benchmark.sh --model gpt-4o
```

### Running Manually

#### 1. Start SABRE Server

```bash
cd /path/to/sabre
uv run sabre-server
```

#### 2. Run k8s-ai-bench

The benchmark expects an OpenAI-compatible API. SABRE provides this at `http://localhost:8011/v1`.

```bash
cd /path/to/kubectl-ai/k8s-ai-bench

# Run with SABRE as backend
./k8s-ai-bench run \
    --agent-bin kubectl-ai \
    --llm-provider openai \
    --models gpt-4o-mini \
    --output-dir .build/sabre-test \
    --concurrency 1

# Run specific tasks
./k8s-ai-bench run \
    --agent-bin kubectl-ai \
    --llm-provider openai \
    --models gpt-4o-mini \
    --task-pattern "create-pod" \
    --output-dir .build/sabre-test \
    --concurrency 1
```

**Important Environment Variables:**
```bash
# Point to SABRE's OpenAI-compatible endpoint
export OPENAI_BASE_URL="http://localhost:8011/v1"
export OPENAI_API_KEY="sk-..."  # Your actual OpenAI key (used by SABRE)
export OPENAI_MODEL=gpt-4o-mini
```

### Baseline Comparison (Direct OpenAI)

To compare against direct OpenAI (without SABRE):

```bash
cd /path/to/kubectl-ai/k8s-ai-bench

# Run directly against OpenAI
export OPENAI_BASE_URL="https://api.openai.com/v1"
./k8s-ai-bench run \
    --agent-bin kubectl-ai \
    --llm-provider openai \
    --models gpt-4o-mini \
    --output-dir .build/baseline-test \
    --concurrency 1
```

## Analyzing Results

### Using k8s-ai-bench Analyzer

```bash
cd /path/to/kubectl-ai/k8s-ai-bench

# Analyze results in markdown format
./k8s-ai-bench analyze --input-dir .build/sabre-test

# Analyze with failure details
./k8s-ai-bench analyze --input-dir .build/sabre-test --show-failures

# Save to file
./k8s-ai-bench analyze --input-dir .build/sabre-test --results-filepath results.md
```

### Using Our Analyzer

```bash
./analyze_results.sh .build/sabre-test
```

## Task Categories

The benchmark includes 25 Kubernetes tasks across several categories:

### Create/Deploy Tasks
- `create-pod` - Create a basic nginx pod
- `create-network-policy` - Create network policies
- `create-pod-mount-configmaps` - Create pods with ConfigMap mounts
- `create-pod-resources-limits` - Create pods with resource limits
- `create-simple-rbac` - Create RBAC roles and bindings
- `create-canary-deployment` - Set up canary deployments

### Troubleshooting/Fix Tasks
- `fix-crashloop` - Fix CrashLoopBackOff errors
- `fix-image-pull` - Fix ImagePullBackOff errors
- `fix-oomkilled` - Fix OOMKilled pod issues
- `fix-pending-pod` - Fix pending pod issues
- `fix-probes` - Fix liveness/readiness probes
- `fix-rbac-wrong-resource` - Fix RBAC permission issues
- `fix-service-routing` - Fix service routing issues
- `fix-service-with-no-endpoints` - Fix services with no endpoints

### Scaling Tasks
- `scale-deployment` - Scale up deployments
- `scale-down-deployment` - Scale down deployments
- `horizontal-pod-autoscaler` - Configure HPA

### Advanced Tasks
- `rolling-update-deployment` - Perform rolling updates
- `deployment-traffic-switch` - Switch traffic between deployments
- `multi-container-pod-communication` - Multi-container pod setup
- `statefulset-lifecycle` - StatefulSet management
- `resize-pvc` - Resize persistent volume claims
- `debug-app-logs` - Debug applications using logs
- `list-images-for-pods` - List container images
- `setup-dev-cluster` - Set up development cluster

## Results Directory Structure

```
results/
└── sabre-gpt4o-mini-run1/
    ├── create-pod/
    │   ├── results.yaml      # Pass/fail status
    │   ├── log.txt           # Execution logs
    │   └── trace.yaml        # API trace
    ├── fix-crashloop/
    │   └── ...
    └── summary.md            # Overall summary
```

## Troubleshooting

### Empty Response from LLM

If you see "Empty response from LLM" in results:
- Ensure SABRE server is running: `uv run sabre-server`
- Check server logs: `tail -f ~/.local/state/sabre/logs/server.log`
- Verify MCP tools are loaded: `uv run sabre list`

### MCP Tool Not Found

If kubectl_ai tools aren't available:
- Check MCP config: `cat ~/.config/sabre/mcp.yaml`
- Verify kubectl-ai is in PATH: `which kubectl-ai`
- Restart SABRE server after config changes

### Kubernetes Connection Issues

If tasks fail with connection errors:
- Verify cluster access: `kubectl cluster-info`
- Check kubeconfig: `echo $KUBECONFIG`
- Ensure cluster is running for kind: `kind get clusters`

### Timeouts

If tasks timeout:
- Increase MCP timeout in `~/.config/sabre/mcp.yaml`
- Check for slow cluster operations
- Consider running with `--concurrency 1`

## Comparing SABRE vs Baseline

To compare SABRE's performance against direct OpenAI:

1. Run baseline (direct OpenAI):
```bash
export OPENAI_BASE_URL="https://api.openai.com/v1"
./k8s-ai-bench run --output-dir .build/baseline ...
```

2. Run with SABRE:
```bash
export OPENAI_BASE_URL="http://localhost:8011/v1"
./k8s-ai-bench run --output-dir .build/sabre ...
```

3. Compare results:
```bash
./k8s-ai-bench analyze --input-dir .build/baseline > baseline.md
./k8s-ai-bench analyze --input-dir .build/sabre > sabre.md
diff baseline.md sabre.md
```

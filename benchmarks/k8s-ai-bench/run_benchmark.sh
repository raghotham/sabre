#!/bin/bash
#
# Run k8s-ai-bench benchmark with SABRE
#
# Usage:
#   ./run_benchmark.sh [options]
#
# Options:
#   --model MODEL           Model to use (default: gpt-4o-mini)
#   --task-pattern PATTERN  Task pattern to filter (default: all)
#   --output-dir DIR        Output directory (default: auto-generated)
#   --baseline              Run against direct OpenAI (not SABRE)
#   --concurrency N         Number of concurrent tasks (default: 1)
#   --help                  Show this help message
#

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SABRE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# K8S_AI_BENCH_DIR must be set by user
if [[ -z "$K8S_AI_BENCH_DIR" ]]; then
    echo -e "${RED}[ERROR]${NC} K8S_AI_BENCH_DIR environment variable not set"
    echo "Please set it to the path of your k8s-ai-bench directory:"
    echo "  export K8S_AI_BENCH_DIR=/path/to/kubectl-ai/k8s-ai-bench"
    exit 1
fi

# Defaults
MODEL="gpt-4o-mini"
TASK_PATTERN=""
OUTPUT_DIR=""
BASELINE=false
CONCURRENCY=1
SABRE_PORT=8011

print_usage() {
    head -20 "$0" | tail -15
}

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --task-pattern)
            TASK_PATTERN="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --baseline)
            BASELINE=true
            shift
            ;;
        --concurrency)
            CONCURRENCY="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Generate output directory name if not provided
if [[ -z "$OUTPUT_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    if [[ "$BASELINE" == "true" ]]; then
        OUTPUT_DIR="$RESULTS_DIR/baseline-${MODEL}-${TIMESTAMP}"
    else
        OUTPUT_DIR="$RESULTS_DIR/sabre-${MODEL}-${TIMESTAMP}"
    fi
fi

# Check prerequisites
log_info "Checking prerequisites..."

# Check k8s-ai-bench directory
if [[ ! -d "$K8S_AI_BENCH_DIR" ]]; then
    log_error "k8s-ai-bench directory not found: $K8S_AI_BENCH_DIR"
    log_info "Set K8S_AI_BENCH_DIR environment variable to point to k8s-ai-bench directory"
    exit 1
fi

# Check k8s-ai-bench binary
if [[ ! -x "$K8S_AI_BENCH_DIR/k8s-ai-bench" ]]; then
    log_error "k8s-ai-bench binary not found or not executable"
    log_info "Build it with: cd $K8S_AI_BENCH_DIR && go build"
    exit 1
fi

# Check kubectl-ai binary
if ! command -v kubectl-ai &> /dev/null; then
    log_error "kubectl-ai binary not found in PATH"
    exit 1
fi

# Check OPENAI_API_KEY
if [[ -z "$OPENAI_API_KEY" ]]; then
    log_error "OPENAI_API_KEY environment variable not set"
    exit 1
fi

# Check kubernetes cluster
if ! kubectl cluster-info &> /dev/null; then
    log_warn "Cannot connect to Kubernetes cluster"
    log_info "Make sure you have a running cluster (kind, minikube, etc.)"
fi

# Set up environment for SABRE or baseline
if [[ "$BASELINE" == "true" ]]; then
    log_info "Running BASELINE benchmark (direct OpenAI)"
    export OPENAI_BASE_URL="https://api.openai.com/v1"
else
    log_info "Running SABRE benchmark"
    export OPENAI_BASE_URL="http://localhost:$SABRE_PORT/v1"

    # Check if SABRE server is running
    if ! curl -s "http://localhost:$SABRE_PORT/health" &> /dev/null; then
        log_warn "SABRE server doesn't appear to be running on port $SABRE_PORT"
        log_info "Start it with: cd $SABRE_ROOT && uv run sabre-server"

        read -p "Would you like to start SABRE server now? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Starting SABRE server in background..."
            cd "$SABRE_ROOT"
            uv run sabre-server &
            SABRE_PID=$!
            log_info "SABRE server started (PID: $SABRE_PID)"
            sleep 5  # Wait for server to start
        else
            log_error "SABRE server required for non-baseline runs"
            exit 1
        fi
    else
        log_info "SABRE server is running on port $SABRE_PORT"
    fi
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

log_info "Configuration:"
log_info "  Model: $MODEL"
log_info "  Task Pattern: ${TASK_PATTERN:-all}"
log_info "  Output Dir: $OUTPUT_DIR"
log_info "  Concurrency: $CONCURRENCY"
log_info "  API Base URL: $OPENAI_BASE_URL"

# Build k8s-ai-bench command
CMD="$K8S_AI_BENCH_DIR/k8s-ai-bench run"
CMD="$CMD --agent-bin kubectl-ai"
CMD="$CMD --llm-provider openai"
CMD="$CMD --models $MODEL"
CMD="$CMD --output-dir $OUTPUT_DIR"
CMD="$CMD --concurrency $CONCURRENCY"

if [[ -n "$TASK_PATTERN" ]]; then
    CMD="$CMD --task-pattern $TASK_PATTERN"
fi

# Run benchmark
log_info "Starting benchmark..."
log_info "Command: $CMD"
echo

cd "$K8S_AI_BENCH_DIR"
eval $CMD
RESULT=$?

echo
if [[ $RESULT -eq 0 ]]; then
    log_info "Benchmark completed successfully!"
else
    log_warn "Benchmark completed with errors (exit code: $RESULT)"
fi

# Run analysis
log_info "Analyzing results..."
"$K8S_AI_BENCH_DIR/k8s-ai-bench" analyze \
    --input-dir "$OUTPUT_DIR" \
    --results-filepath "$OUTPUT_DIR/summary.md" \
    --show-failures

log_info "Results saved to: $OUTPUT_DIR"
log_info "Summary: $OUTPUT_DIR/summary.md"

# Copy summary to results directory
if [[ -f "$OUTPUT_DIR/summary.md" ]]; then
    cat "$OUTPUT_DIR/summary.md"
fi

exit $RESULT

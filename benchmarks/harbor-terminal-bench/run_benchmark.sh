#!/bin/bash
#
# Run Terminal-Bench 2.0 benchmark with SABRE via Harbor
#
# Usage:
#   ./run_benchmark.sh [options]
#
# Options:
#   --model MODEL           Model to use (default: gpt-4o)
#   --dataset DATASET       Dataset to run (default: terminal-bench@2.0)
#   --task-name TASK        Run specific task by name (within the dataset)
#   --n-concurrent N        Number of concurrent tasks (default: 4)
#   --jobs-dir DIR          Output directory for results (default: ./results)
#   --env ENV               Environment type: docker or local (default: docker)
#   --version VERSION       SABRE version/branch to install (default: main)
#   --timeout-multiplier N  Multiplier for timeouts (default: 1)
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
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Defaults
MODEL="gpt-4o"
DATASET="terminal-bench@2.0"
TASK_NAME=""
N_CONCURRENT=4
JOBS_DIR=""
ENV_TYPE="docker"
SABRE_VERSION="main"
TIMEOUT_MULTIPLIER="1"

print_usage() {
    head -18 "$0" | tail -15
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

log_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --task-name)
            TASK_NAME="$2"
            shift 2
            ;;
        --n-concurrent)
            N_CONCURRENT="$2"
            shift 2
            ;;
        --jobs-dir)
            JOBS_DIR="$2"
            shift 2
            ;;
        --env)
            ENV_TYPE="$2"
            shift 2
            ;;
        --version)
            SABRE_VERSION="$2"
            shift 2
            ;;
        --timeout-multiplier)
            TIMEOUT_MULTIPLIER="$2"
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

# Generate output directory if not provided
if [[ -z "$JOBS_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d-%H%M%S)
    JOBS_DIR="$RESULTS_DIR/sabre-${MODEL//\//-}-${TIMESTAMP}"
fi

# Check prerequisites
log_step "Checking prerequisites..."

# Check harbor CLI
if ! command -v harbor &> /dev/null; then
    log_error "harbor CLI not found in PATH"
    log_info "Install harbor: pip install harbor-bench"
    log_info "Or: uv pip install harbor-bench"
    exit 1
fi
log_info "harbor CLI found: $(which harbor)"

# Check Docker (if using docker environment)
if [[ "$ENV_TYPE" == "docker" ]]; then
    if ! command -v docker &> /dev/null; then
        log_error "docker not found in PATH (required for --env docker)"
        exit 1
    fi
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running"
        exit 1
    fi
    log_info "Docker is available"
fi

# Check OPENAI_API_KEY
if [[ -z "$OPENAI_API_KEY" ]]; then
    log_error "OPENAI_API_KEY environment variable not set"
    exit 1
fi
log_info "OPENAI_API_KEY is set"

# Add sabre_harbor to Python path
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Display configuration
echo ""
log_step "Configuration:"
log_info "  Model: $MODEL"
log_info "  Dataset: $DATASET"
log_info "  Task: ${TASK_NAME:-all}"
log_info "  Concurrent tasks: $N_CONCURRENT"
log_info "  Jobs directory: $JOBS_DIR"
log_info "  Environment: $ENV_TYPE"
log_info "  SABRE version: $SABRE_VERSION"
log_info "  Timeout multiplier: $TIMEOUT_MULTIPLIER"
echo ""

# Create results directory
mkdir -p "$JOBS_DIR"

# Build harbor command
CMD="harbor run"
CMD="$CMD -d $DATASET"
CMD="$CMD --agent-import-path sabre_harbor:SabreAgent"
CMD="$CMD --jobs-dir $JOBS_DIR"
CMD="$CMD --n-concurrent $N_CONCURRENT"
CMD="$CMD --env $ENV_TYPE"
CMD="$CMD --timeout-multiplier $TIMEOUT_MULTIPLIER"

# Add task name if specified
if [[ -n "$TASK_NAME" ]]; then
    CMD="$CMD --task-name $TASK_NAME"
fi

# Set environment variables for the agent
export OPENAI_MODEL="$MODEL"

# Run benchmark
log_step "Starting Terminal-Bench 2.0 benchmark..."
log_info "Command: $CMD"
echo ""

eval $CMD
RESULT=$?

echo ""
if [[ $RESULT -eq 0 ]]; then
    log_info "Benchmark completed successfully!"
else
    log_warn "Benchmark completed with errors (exit code: $RESULT)"
fi

# Display results location
log_info "Results saved to: $JOBS_DIR"

# List result files
if [[ -d "$JOBS_DIR" ]]; then
    log_info "Result files:"
    ls -la "$JOBS_DIR"
fi

exit $RESULT

#!/bin/bash
#
# Analyze k8s-ai-bench results
#
# Usage:
#   ./analyze_results.sh <results-dir>
#   ./analyze_results.sh --compare <dir1> <dir2>
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
K8S_AI_BENCH_DIR="${K8S_AI_BENCH_DIR:-$HOME/Documents/workspace/kubectl-ai/k8s-ai-bench}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

if [[ "$1" == "--compare" ]]; then
    if [[ -z "$2" || -z "$3" ]]; then
        echo "Usage: $0 --compare <dir1> <dir2>"
        exit 1
    fi

    DIR1="$2"
    DIR2="$3"

    echo -e "${BLUE}=== Comparing Results ===${NC}"
    echo

    # Analyze both directories
    echo -e "${GREEN}Results from: $DIR1${NC}"
    "$K8S_AI_BENCH_DIR/k8s-ai-bench" analyze --input-dir "$DIR1" 2>/dev/null || true
    echo

    echo -e "${GREEN}Results from: $DIR2${NC}"
    "$K8S_AI_BENCH_DIR/k8s-ai-bench" analyze --input-dir "$DIR2" 2>/dev/null || true
    echo

    # Count pass/fail
    count_results() {
        local dir="$1"
        local pass=0
        local fail=0
        for f in "$dir"/*/results.yaml; do
            if [[ -f "$f" ]]; then
                result=$(grep "^result:" "$f" | cut -d' ' -f2)
                if [[ "$result" == "success" ]]; then
                    ((pass++)) || true
                else
                    ((fail++)) || true
                fi
            fi
        done
        echo "$pass $fail"
    }

    read pass1 fail1 <<< $(count_results "$DIR1")
    read pass2 fail2 <<< $(count_results "$DIR2")

    total1=$((pass1 + fail1))
    total2=$((pass2 + fail2))

    echo -e "${BLUE}=== Summary ===${NC}"
    echo -e "$(basename $DIR1): ${GREEN}$pass1 passed${NC}, ${RED}$fail1 failed${NC} (total: $total1)"
    echo -e "$(basename $DIR2): ${GREEN}$pass2 passed${NC}, ${RED}$fail2 failed${NC} (total: $total2)"

    if [[ $total1 -gt 0 && $total2 -gt 0 ]]; then
        pct1=$(echo "scale=1; $pass1 * 100 / $total1" | bc)
        pct2=$(echo "scale=1; $pass2 * 100 / $total2" | bc)
        echo -e "Pass rate: $pct1% vs $pct2%"
    fi

else
    # Single directory analysis
    if [[ -z "$1" ]]; then
        echo "Usage: $0 <results-dir>"
        echo "       $0 --compare <dir1> <dir2>"
        exit 1
    fi

    RESULTS_DIR="$1"

    if [[ ! -d "$RESULTS_DIR" ]]; then
        echo -e "${RED}Error: Directory not found: $RESULTS_DIR${NC}"
        exit 1
    fi

    echo -e "${BLUE}=== k8s-ai-bench Results Analysis ===${NC}"
    echo -e "Directory: $RESULTS_DIR"
    echo

    # Run k8s-ai-bench analyzer
    "$K8S_AI_BENCH_DIR/k8s-ai-bench" analyze \
        --input-dir "$RESULTS_DIR" \
        --show-failures

    echo
    echo -e "${BLUE}=== Task Details ===${NC}"

    # Show individual task results
    for task_dir in "$RESULTS_DIR"/*/; do
        if [[ -d "$task_dir" ]]; then
            task_name=$(basename "$task_dir")
            results_file="$task_dir/results.yaml"

            if [[ -f "$results_file" ]]; then
                result=$(grep "^result:" "$results_file" | cut -d' ' -f2)
                if [[ "$result" == "success" ]]; then
                    echo -e "${GREEN}✅ $task_name${NC}"
                else
                    echo -e "${RED}❌ $task_name${NC}"
                    # Show failure reason if available
                    if grep -q "failures:" "$results_file"; then
                        grep -A2 "message:" "$results_file" | head -3 | sed 's/^/   /'
                    fi
                fi
            fi
        fi
    done
fi

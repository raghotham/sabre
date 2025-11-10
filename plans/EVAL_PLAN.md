# SABRE Persona Evaluation Plan

## Overview

Evaluate memory system impact using **established data science benchmarks** rather than custom evals. This ensures:
- ✅ Credible, peer-reviewed evaluation methodology
- ✅ Comparable results with other systems
- ✅ Comprehensive coverage of real-world tasks
- ✅ Persona-specific evaluation (data-analyst first)

## Benchmark Selection

### Primary: DataSciBench (222 prompts, 6 task categories)

**Why DataSciBench:**
- Comprehensive coverage of data science workflows
- Multi-step tasks (aligns with SABRE's continuation model)
- Both coarse and fine-grained metrics
- Open source, reproducible

**Six Task Categories:**
1. Data cleaning and preprocessing
2. Data exploration and statistics
3. Data visualization
4. Predictive modeling
5. Data mining and pattern recognition
6. Interpretability and report generation

**Metrics:**
- **Success Rate (SR)**: Complete success rate over 10 runs
- **Completion Rate (CR)**: Step-by-step completion quality (0-2 per step)
- **Fine-grained**: Data Quality (F1), Plot Validity (F2), Data Accuracy (F3), Viz Completeness (F4), Model Accuracy (F5)

**Dataset:**
- 222 prompts (easy: 167, medium: 30, hard: 25)
- 519 test cases with ground truth
- 25 evaluation functions

### Secondary: CIBench (Code Interpreter Evaluation)

**Why CIBench:**
- Evaluates process AND output quality
- Measures tool usage effectiveness
- Validates code execution correctness

**Metrics:**
- **Process-oriented**:
  - Tool Call Rate: Correct helper invocation
  - Executable Rate: Error-free code execution

- **Output-oriented**:
  - Numeric Accuracy: Precision of numerical results
  - Text Score: Structural text quality (ROUGE)
  - Visualization Score: Visual output similarity

### Stretch Goal: DABstep (Financial Analytics)

**Why DABstep:**
- Real-world complexity (450+ tasks from production system)
- Multi-step reasoning requirements
- High difficulty (best agents: 14.55% accuracy)

**Use case:** Advanced evaluation after passing DataSciBench

---

## Evaluation Strategy

### Phase 1: Baseline (No Memory)

**Setup:**
- SABRE with data-analyst persona
- No memory system enabled
- Fresh namespace for each task
- Standard helpers available

**Run:**
```bash
# Run DataSciBench baseline
uv run datascibench \
  --persona=data-analyst \
  --memory=false \
  --output=results/baseline_databench.json

# Run CIBench baseline
uv run python -m sabre.eval.cibench \
  --persona=data-analyst \
  --memory=false \
  --output=results/baseline_cibench.json
```

**Expected Results:**
- DataSciBench SR: ~40-60% (estimate based on GPT-4 performance)
- CIBench Tool Call Rate: ~70-80%
- CIBench Executable Rate: ~60-70%

**Collect:**
- Success rates per task category
- Execution time per task
- Token usage per task
- Helper call frequency
- Error types and frequencies

---

### Phase 2: With Memory (After Implementation)

**Setup:**
- SABRE with data-analyst persona
- Memory system enabled (all 5 types)
- Semantic memory pre-populated with common schemas
- Episodic memory empty (learns during eval)
- Procedural memory has standard workflows

**Run:**
```bash
# Run DataSciBench with memory
uv run datascibench \
  --persona=data-analyst \
  --memory=true \
  --output=results/memory_databench.json

# Run CIBench with memory
uv run python -m sabre.eval.cibench \
  --persona=data-analyst \
  --memory=true \
  --output=results/memory_cibench.json
```

**Expected Improvements:**
- DataSciBench SR: +10-20% (faster data exploration)
- CIBench Tool Call Rate: +5-10% (learned patterns)
- CIBench Executable Rate: +10-15% (cached schemas, fewer errors)
- Execution Time: -30-50% (cached operations)
- Token Usage: -20-40% (less redundant context)

---

### Phase 3: Longitudinal Evaluation (Expertise Accumulation)

**Hypothesis:** Memory enables learning over time, improving performance on later tasks

**Setup:**
- Run DataSciBench tasks in sequential batches
- Track memory accumulation between batches
- Measure performance improvement over time

**Batches:**
1. **Batch 1 (Cold start)**: First 50 tasks, empty memory
2. **Batch 2 (Learning)**: Next 50 tasks, memory from Batch 1
3. **Batch 3 (Warm)**: Next 50 tasks, memory from Batch 1+2
4. **Batch 4 (Expert)**: Final tasks, full accumulated memory

**Track:**
- Success rate per batch (should increase)
- Cache hit rate per batch (should increase)
- Average execution time per batch (should decrease)
- Memory size growth
- Episodic memory usage frequency

**Expected Trend:**
```
Batch 1 (cold):    SR=45%, Time=30s, Cache=0%
Batch 2 (warm):    SR=52%, Time=25s, Cache=25%
Batch 3 (warmer):  SR=58%, Time=20s, Cache=45%
Batch 4 (expert):  SR=65%, Time=15s, Cache=60%
```

---

## Evaluation Metrics (Comprehensive)

### Performance Metrics

| Metric | Baseline | With Memory | Target Improvement |
|--------|----------|-------------|-------------------|
| **DataSciBench Success Rate** | 50% | 60-70% | +20-40% |
| **DataSciBench Completion Rate** | 1.2/2.0 | 1.5/2.0 | +25% |
| **CIBench Tool Call Rate** | 75% | 80-85% | +7-13% |
| **CIBench Executable Rate** | 65% | 75-80% | +15-23% |
| **CIBench Numeric Accuracy** | 70% | 75-80% | +7-14% |

### Efficiency Metrics

| Metric | Baseline | With Memory | Target Improvement |
|--------|----------|-------------|-------------------|
| **Avg Execution Time** | 25s | 15s | -40% |
| **Token Usage (Input)** | 2000 | 1200 | -40% |
| **Token Usage (Output)** | 800 | 600 | -25% |
| **API Calls (Database)** | 5 | 2 | -60% |
| **API Calls (Web)** | 3 | 1 | -67% |

### Memory Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Cache Hit Rate** | % of operations using cached data | >50% |
| **Semantic Facts** | Number of facts stored | 100-500 |
| **Episodic Episodes** | Number of recorded incidents | 50-200 |
| **Procedural Workflows** | Number of learned procedures | 5-20 |
| **Memory Size** | Total disk usage | <100MB |

### Quality Metrics

| Metric | Description | Target |
|--------|-------------|--------|
| **Correctness** | Matches ground truth | No degradation |
| **Completeness** | All required steps executed | No degradation |
| **Code Quality** | Executable, efficient | No degradation |
| **Explanation Quality** | Clear, accurate insights | Improvement |

---

## Implementation Plan

### Step 1: Integrate DataSciBench

```python
# sabre/eval/datascibench.py

import json
from pathlib import Path
from sabre.server.orchestrator import Orchestrator
from sabre.eval.metrics import calculate_success_rate, calculate_completion_rate


class DataSciEvaluator:
    """Run DataSciBench evaluation"""

    def __init__(self, persona: str, enable_memory: bool):
        self.persona = persona
        self.enable_memory = enable_memory
        self.results = []

    async def load_benchmark(self, path: str):
        """Load DataSciBench prompts and test cases"""
        with open(path) as f:
            self.benchmark = json.load(f)

        print(f"Loaded {len(self.benchmark['prompts'])} prompts")
        print(f"Categories: {set(p['category'] for p in self.benchmark['prompts'])}")

    async def run_task(self, task: dict) -> dict:
        """Execute single DataSciBench task"""
        orchestrator = Orchestrator(
            persona=self.persona, enable_memory=self.enable_memory
        )

        start_time = time.time()

        # Run task
        result = await orchestrator.run(
            conversation_id=f"databench_{task['id']}", input_text=task["prompt"]
        )

        execution_time = time.time() - start_time

        # Evaluate result against ground truth
        eval_result = self.evaluate_result(
            result, task["ground_truth"], task["eval_fn"]
        )

        return {
            "task_id": task["id"],
            "category": task["category"],
            "difficulty": task["difficulty"],
            "success": eval_result["success"],
            "completion_score": eval_result["completion_score"],
            "execution_time": execution_time,
            "tokens_used": result.get("tokens", {}),
            "memory_stats": result.get("memory_stats", {}),
        }

    async def run_all(self) -> dict:
        """Run full benchmark"""
        for task in self.benchmark["prompts"]:
            result = await self.run_task(task)
            self.results.append(result)

            # Log progress
            print(
                f"[{len(self.results)}/{len(self.benchmark['prompts'])}] "
                f"{task['id']}: {'✓' if result['success'] else '✗'} "
                f"({result['execution_time']:.1f}s)"
            )

        # Calculate aggregate metrics
        return self.calculate_metrics()

    def calculate_metrics(self) -> dict:
        """Calculate benchmark metrics"""
        total = len(self.results)

        # Overall metrics
        success_rate = sum(r["success"] for r in self.results) / total
        avg_completion = sum(r["completion_score"] for r in self.results) / total
        avg_time = sum(r["execution_time"] for r in self.results) / total

        # By category
        by_category = {}
        for category in set(r["category"] for r in self.results):
            category_results = [r for r in self.results if r["category"] == category]
            by_category[category] = {
                "success_rate": sum(r["success"] for r in category_results)
                / len(category_results),
                "avg_time": sum(r["execution_time"] for r in category_results)
                / len(category_results),
            }

        # By difficulty
        by_difficulty = {}
        for difficulty in ["easy", "medium", "hard"]:
            diff_results = [r for r in self.results if r["difficulty"] == difficulty]
            if diff_results:
                by_difficulty[difficulty] = {
                    "success_rate": sum(r["success"] for r in diff_results)
                    / len(diff_results),
                    "count": len(diff_results),
                }

        return {
            "overall": {
                "success_rate": success_rate,
                "completion_rate": avg_completion,
                "avg_execution_time": avg_time,
                "total_tasks": total,
            },
            "by_category": by_category,
            "by_difficulty": by_difficulty,
            "memory_enabled": self.enable_memory,
        }
```

### Step 2: Run Baseline

```bash
# Install DataSciBench
git clone https://github.com/xxxxx/DataSciBench
cd DataSciBench
pip install -e .

# Run baseline evaluation
cd /Users/raghu/dev/sabre
uv run datascibench \
  --benchmark=../DataSciBench/benchmark.json \
  --persona=data-analyst \
  --memory=false \
  --output=results/baseline_databench.json \
  --verbose
```

### Step 3: Analyze Results

```python
# sabre/eval/compare.py


def compare_results(baseline_path: str, memory_path: str):
    """Compare baseline vs memory results"""

    with open(baseline_path) as f:
        baseline = json.load(f)

    with open(memory_path) as f:
        memory = json.load(f)

    print("=" * 60)
    print("DataSciBench Comparison: Baseline vs Memory")
    print("=" * 60)

    print("\nOverall Performance:")
    print(
        f"  Success Rate:     {baseline['overall']['success_rate']:.1%} → {memory['overall']['success_rate']:.1%} "
        f"({memory['overall']['success_rate'] - baseline['overall']['success_rate']:+.1%})"
    )
    print(
        f"  Completion Rate:  {baseline['overall']['completion_rate']:.2f} → {memory['overall']['completion_rate']:.2f} "
        f"({memory['overall']['completion_rate'] - baseline['overall']['completion_rate']:+.2f})"
    )
    print(
        f"  Avg Time:         {baseline['overall']['avg_execution_time']:.1f}s → {memory['overall']['avg_execution_time']:.1f}s "
        f"({memory['overall']['avg_execution_time'] - baseline['overall']['avg_execution_time']:+.1f}s)"
    )

    print("\nBy Category:")
    for category in baseline["by_category"]:
        base_sr = baseline["by_category"][category]["success_rate"]
        mem_sr = memory["by_category"][category]["success_rate"]
        print(
            f"  {category:30s}: {base_sr:.1%} → {mem_sr:.1%} ({mem_sr - base_sr:+.1%})"
        )

    print("\nBy Difficulty:")
    for difficulty in ["easy", "medium", "hard"]:
        if difficulty in baseline["by_difficulty"]:
            base_sr = baseline["by_difficulty"][difficulty]["success_rate"]
            mem_sr = memory["by_difficulty"][difficulty]["success_rate"]
            print(
                f"  {difficulty:10s}: {base_sr:.1%} → {mem_sr:.1%} ({mem_sr - base_sr:+.1%})"
            )
```

---

## Success Criteria

### Minimum Viable Improvement (Memory worth implementing)

1. ✅ **DataSciBench Success Rate**: +10% improvement
2. ✅ **Execution Time**: -20% reduction
3. ✅ **Quality**: No degradation (<-2%)
4. ✅ **Cache Hit Rate**: >40% on repeated operations

### Strong Performance (Memory significantly valuable)

1. ✅ **DataSciBench Success Rate**: +20% improvement
2. ✅ **Execution Time**: -40% reduction
3. ✅ **Quality**: Slight improvement (+2-5%)
4. ✅ **Cache Hit Rate**: >60% on repeated operations
5. ✅ **Longitudinal**: 2x improvement from Batch 1 to Batch 4

### Exceptional Performance (Memory transformative)

1. ✅ **DataSciBench Success Rate**: +30% improvement
2. ✅ **Execution Time**: -50% reduction
3. ✅ **Quality**: Significant improvement (+5-10%)
4. ✅ **Cache Hit Rate**: >70% on repeated operations
5. ✅ **Longitudinal**: 3x improvement from Batch 1 to Batch 4

---

## Timeline

### Week 1: Setup
- [ ] Download DataSciBench dataset
- [ ] Implement DataSciEvaluator
- [ ] Implement CIBench evaluator
- [ ] Test evaluation pipeline

### Week 2: Baseline
- [ ] Run DataSciBench baseline (no memory)
- [ ] Run CIBench baseline (no memory)
- [ ] Analyze results
- [ ] Document baseline performance

### Week 3-6: Implement Memory System
- [ ] Implement semantic memory
- [ ] Implement episodic memory
- [ ] Implement procedural memory
- [ ] Implement prospective memory
- [ ] Integration with orchestrator

### Week 7: With Memory Evaluation
- [ ] Run DataSciBench with memory
- [ ] Run CIBench with memory
- [ ] Longitudinal evaluation (4 batches)
- [ ] Analyze improvements

### Week 8: Analysis & Decision
- [ ] Compare baseline vs memory
- [ ] Analyze which memory types help most
- [ ] Document findings
- [ ] Decision: Is memory worth it?

---

## Risks & Mitigations

### Risk 1: Benchmark doesn't align with SABRE's strengths
**Mitigation:** Run multiple benchmarks (DataSciBench + CIBench + DABstep subset)

### Risk 2: Memory overhead negates benefits
**Mitigation:** Track memory operations separately, optimize hot paths

### Risk 3: Tasks don't benefit from memory (one-off tasks)
**Mitigation:** Include longitudinal eval to measure accumulated expertise

### Risk 4: Ground truth evaluation is hard
**Mitigation:** Use benchmark's built-in eval functions, supplement with LLM-as-judge

---

## Next Steps

1. **Download DataSciBench**: Clone repo and set up benchmark
2. **Implement evaluator**: Build DataSciEvaluator class
3. **Run baseline**: Get baseline numbers before implementing memory
4. **Share results**: Review baseline to confirm eval is working
5. **Implement memory**: Build memory system
6. **Re-run eval**: Compare with vs without memory
7. **Make decision**: Is memory worth the complexity?

## References

- DataSciBench: https://arxiv.org/html/2502.13897
- CIBench: https://arxiv.org/html/2407.10499v1
- DABstep: https://arxiv.org/abs/2506.23719

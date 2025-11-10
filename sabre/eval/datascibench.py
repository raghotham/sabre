"""DataSciBench evaluation runner for SABRE.

This module runs DataSciBench tasks through SABRE and measures performance.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from sabre.sdk import SabreClient
from sabre.common.models.events import (
    HelpersExecutionEndEvent,
)

logger = logging.getLogger(__name__)


@dataclass
class TaskStep:
    """Single step in task execution."""

    step_id: int
    code: str
    result: str
    is_success: bool
    error: str | None = None


@dataclass
class TaskOutput:
    """Output for a single DataSciBench task."""

    task_id: str
    output_dir: str
    time_cost: float
    steps: list[TaskStep]
    error_count: int
    tokens_used: dict[str, int]
    final_result: str
    is_success: bool


@dataclass
class BenchmarkResult:
    """Aggregate results for full benchmark run."""

    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    avg_time: float
    total_tokens: dict[str, int]
    task_outputs: list[TaskOutput]

    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0


class DataSciBenchEvaluator:
    """Evaluates SABRE on DataSciBench tasks."""

    def __init__(
        self,
        benchmark_path: Path,
        output_dir: Path,
        enable_memory: bool = False,
        persona: str = "data-analyst",
        server_url: str = "http://localhost:8011",
    ):
        """Initialize evaluator.

        Args:
            benchmark_path: Path to DataSciBench data/ directory
            output_dir: Directory to save results
            enable_memory: Whether to enable SABRE's memory system (not yet implemented)
            persona: Persona to use for evaluation (not yet implemented)
            server_url: SABRE server URL
        """
        self.benchmark_path = Path(benchmark_path)
        self.output_dir = Path(output_dir)
        self.enable_memory = enable_memory
        self.persona = persona

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Create SABRE client
        self.client = SabreClient(base_url=server_url, timeout=600.0)

        # Track results
        self.task_outputs: list[TaskOutput] = []

    def load_tasks(self, task_filter: str | None = None, limit: int | None = None) -> list[dict]:
        """Load DataSciBench tasks from data/ directory.

        Args:
            task_filter: Optional filter for task IDs (e.g., "bcb" for BigCodeBench tasks)
            limit: Maximum number of tasks to load

        Returns:
            List of task dicts with id, prompt, data_source_type
        """
        tasks = []
        data_dir = self.benchmark_path / "data"

        if not data_dir.exists():
            raise FileNotFoundError(f"DataSciBench data directory not found: {data_dir}")

        # Iterate through task folders
        for task_dir in sorted(data_dir.iterdir()):
            if not task_dir.is_dir():
                continue

            task_id = task_dir.name

            # Apply filter
            if task_filter and not task_id.startswith(task_filter):
                continue

            # Load prompt
            prompt_file = task_dir / "prompt.json"
            if not prompt_file.exists():
                logger.warning(f"No prompt.json found for task {task_id}")
                continue

            with open(prompt_file) as f:
                prompt_data = json.load(f)

            tasks.append(
                {
                    "id": task_id,
                    "prompt": prompt_data["prompt"],
                    "data_source_type": prompt_data.get("data_source_type", ""),
                    "task_dir": task_dir,
                }
            )

            if limit and len(tasks) >= limit:
                break

        logger.info(f"Loaded {len(tasks)} tasks from {data_dir}")
        return tasks

    async def run_task(self, task: dict) -> TaskOutput:
        """Run a single DataSciBench task through SABRE.

        Args:
            task: Task dict with id, prompt, task_dir

        Returns:
            TaskOutput with results
        """
        task_id = task["id"]
        prompt = task["prompt"]

        # Create output directory for this task
        task_output_dir = self.output_dir / task_id
        task_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Running task {task_id}")

        # Augment prompt with DataSciBench conventions
        # Input data is at ../, output should be saved to ./
        augmented_prompt = (
            "All input source data is at the `../` folder. "
            "All output files should be saved to the current folder `./`.\n\n"
            f"{prompt}"
        )

        # Track execution
        steps: list[TaskStep] = []
        error_count = 0
        step_id = 0

        # Callback to capture helpers execution
        async def event_callback(event):
            nonlocal step_id, error_count

            if isinstance(event, HelpersExecutionEndEvent):
                # Capture code execution result
                code = event.data.get("code", "")
                result = event.data.get("result", "")
                is_success = event.data.get("success", False)

                if not is_success:
                    error_count += 1

                steps.append(
                    TaskStep(
                        step_id=step_id,
                        code=code,
                        result=result,
                        is_success=is_success,
                        error=None,
                    )
                )
                step_id += 1

        start_time = time.time()

        try:
            # Run task through SDK client
            result = await self.client.run(
                message=augmented_prompt,
                conversation_id=None,  # Create new conversation
                event_callback=event_callback,
            )

            execution_time = time.time() - start_time

            # Extract result from SabreResult
            final_result = result.response
            is_success = result.success
            tokens_used = {
                "input": result.input_tokens,
                "output": result.output_tokens,
                "reasoning": result.reasoning_tokens,
            }

            logger.info(
                f"Task {task_id} completed in {execution_time:.2f}s (success={is_success}, errors={error_count})"
            )

            return TaskOutput(
                task_id=task_id,
                output_dir=str(task_output_dir),
                time_cost=execution_time,
                steps=steps,
                error_count=error_count,
                tokens_used=tokens_used,
                final_result=final_result,
                is_success=is_success,
            )

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Task {task_id} failed with exception: {e}")

            return TaskOutput(
                task_id=task_id,
                output_dir=str(task_output_dir),
                time_cost=execution_time,
                steps=steps,
                error_count=error_count + 1,
                tokens_used={},
                final_result=str(e),
                is_success=False,
            )

    async def run_all(self, task_filter: str | None = None, limit: int | None = None) -> BenchmarkResult:
        """Run all DataSciBench tasks.

        Args:
            task_filter: Optional filter for task IDs
            limit: Maximum number of tasks to run

        Returns:
            BenchmarkResult with aggregate metrics
        """
        tasks = self.load_tasks(task_filter=task_filter, limit=limit)

        if not tasks:
            raise ValueError("No tasks loaded")

        logger.info(f"Running {len(tasks)} tasks")

        for idx, task in enumerate(tasks, 1):
            logger.info(f"[{idx}/{len(tasks)}] Starting task {task['id']}")

            task_output = await self.run_task(task)
            self.task_outputs.append(task_output)

            # Save intermediate results
            self.save_results()

        # Calculate aggregate metrics
        return self.calculate_metrics()

    def calculate_metrics(self) -> BenchmarkResult:
        """Calculate aggregate metrics from task outputs."""
        total_tasks = len(self.task_outputs)
        successful_tasks = sum(1 for t in self.task_outputs if t.is_success)
        failed_tasks = total_tasks - successful_tasks

        avg_time = sum(t.time_cost for t in self.task_outputs) / total_tasks if total_tasks > 0 else 0

        # Aggregate token usage
        total_tokens = {
            "input": sum(t.tokens_used.get("input", 0) for t in self.task_outputs),
            "output": sum(t.tokens_used.get("output", 0) for t in self.task_outputs),
        }

        return BenchmarkResult(
            total_tasks=total_tasks,
            successful_tasks=successful_tasks,
            failed_tasks=failed_tasks,
            avg_time=avg_time,
            total_tokens=total_tokens,
            task_outputs=self.task_outputs,
        )

    def save_results(self):
        """Save results to output directory."""
        results = self.calculate_metrics()

        # Save summary
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "success_rate": results.success_rate,
                    "total_tasks": results.total_tasks,
                    "successful_tasks": results.successful_tasks,
                    "failed_tasks": results.failed_tasks,
                    "avg_time": results.avg_time,
                    "total_tokens": results.total_tokens,
                },
                f,
                indent=2,
            )

        # Save detailed results (JSONL format)
        details_path = self.output_dir / "results.jsonl"
        with open(details_path, "w") as f:
            for task_output in self.task_outputs:
                f.write(json.dumps(asdict(task_output)) + "\n")

        logger.info(f"Results saved to {self.output_dir}")


async def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run DataSciBench evaluation with SABRE")
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(__file__).parent.parent.parent / "tmp" / "DataSciBench",
        help="Path to DataSciBench directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parent.parent.parent / "results" / "baseline",
        help="Output directory for results",
    )
    parser.add_argument("--memory", action="store_true", help="Enable memory system")
    parser.add_argument("--persona", default="data-analyst", help="Persona to use")
    parser.add_argument("--filter", help="Filter tasks by ID prefix (e.g., 'bcb')")
    parser.add_argument("--limit", type=int, help="Limit number of tasks")
    parser.add_argument("--server", default="http://localhost:8011", help="SABRE server URL")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    evaluator = DataSciBenchEvaluator(
        benchmark_path=args.benchmark,
        output_dir=args.output,
        enable_memory=args.memory,
        persona=args.persona,
        server_url=args.server,
    )

    # Check server health
    print("Checking SABRE server health...")
    if not await evaluator.client.health_check():
        print(f"ERROR: SABRE server not reachable at {args.server}")
        print("Please start the server with: uv run sabre")
        return

    print(f"✓ Server healthy at {args.server}\n")

    results = await evaluator.run_all(task_filter=args.filter, limit=args.limit)

    # Print summary
    print("\n" + "=" * 60)
    print("DataSciBench Evaluation Results")
    print("=" * 60)
    print(f"Success Rate: {results.success_rate:.1%}")
    print(f"Total Tasks: {results.total_tasks}")
    print(f"Successful: {results.successful_tasks}")
    print(f"Failed: {results.failed_tasks}")
    print(f"Avg Time: {results.avg_time:.2f}s")
    print(f"Total Tokens: {results.total_tokens}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""
Wrapper script for running DataSciBench benchmarks with compatibility fixes.

This script applies necessary compatibility patches and handles environment variable
expansion in config files WITHOUT modifying DataSciBench's code.

Features:
- Downloads ground truth data from HuggingFace
- Runs benchmarks with compatibility patches
- Evaluates results
- Stores results in benchmarks/DataSciBench/results/

Usage:
    python run_benchmark.py --task_id csv_excel_0 --config configs/config_gpt4o_baseline.yaml --data_type csv --max_runs 1
"""

import sys
import os
from pathlib import Path
import tempfile
import re
import yaml
import shutil
import subprocess

# Add DataSciBench to Python path
DATASCIBENCH_DIR = Path(__file__).parent.parent.parent / "tmp" / "DataSciBench"
sys.path.insert(0, str(DATASCIBENCH_DIR))


# Apply openai/httpx compatibility patch BEFORE any openai imports
def patch_openai_httpx():
    """Patch AsyncHttpxClientWrapper to accept 'proxies' and convert to 'proxy'."""
    try:
        from openai import _base_client

        OriginalAsyncHttpxClientWrapper = _base_client.AsyncHttpxClientWrapper

        class PatchedAsyncHttpxClientWrapper(OriginalAsyncHttpxClientWrapper):
            """Wrapper that accepts both 'proxy' and 'proxies' parameters."""

            def __init__(self, *, proxies=None, proxy=None, **kwargs):
                actual_proxy = proxy if proxy is not None else proxies
                super().__init__(proxy=actual_proxy, **kwargs)

        _base_client.AsyncHttpxClientWrapper = PatchedAsyncHttpxClientWrapper
        print("✓ Applied openai/httpx compatibility patch")

    except Exception as e:
        print(f"✗ Failed to patch openai: {e}")
        import traceback

        traceback.print_exc()


def expand_env_vars(text: str) -> str:
    """
    Expand environment variables in text.

    Supports syntax: ${VAR_NAME} and $VAR_NAME
    """

    def replace_var(match):
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))

    pattern = r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)"
    return re.sub(pattern, replace_var, text)


def prepare_config(config_path: str) -> str:
    """
    Load config file, expand environment variables, and return path to processed config.

    Returns:
        Path to temporary config file with env vars expanded
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Read config file
    with open(config_path, "r") as f:
        config_text = f.read()

    # Expand environment variables
    expanded_text = expand_env_vars(config_text)

    # Write to temporary file
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    tmp.write(expanded_text)
    tmp.close()

    print(f"✓ Loaded config from {config_path} with env var expansion")
    print(f"  Temporary config: {tmp.name}")

    return tmp.name


def download_ground_truth_data():
    """Download ground truth data from HuggingFace if not already present."""
    gt_data_dir = DATASCIBENCH_DIR / "gt_data"
    zip_file = DATASCIBENCH_DIR / "DataSciBench_GroundTruth_Data.zip"

    # Check if ground truth data already exists
    if gt_data_dir.exists() and any(gt_data_dir.iterdir()):
        print("✓ Ground truth data already present")
        return

    # Check if zip file exists but not extracted
    if zip_file.exists() and not gt_data_dir.exists():
        print("✓ Found ground truth zip, extracting...")
        subprocess.run(["unzip", "-q", str(zip_file)], cwd=str(DATASCIBENCH_DIR), check=True)
        print("✓ Ground truth data extracted")
        return

    print("→ Downloading ground truth data from HuggingFace...")

    # Read HF token
    hf_key_path = Path.home() / ".hf" / "key"
    if not hf_key_path.exists():
        print("✗ HuggingFace key not found at ~/.hf/key")
        print("  Please create this file with your HuggingFace token")
        return

    with open(hf_key_path, "r") as f:
        hf_token = f.read().strip()

    # Download using huggingface_hub
    try:
        from huggingface_hub import hf_hub_download, login

        login(token=hf_token)

        file_path = hf_hub_download(
            repo_id="zd21/DataSciBench",
            filename="DataSciBench_GroundTruth_Data.zip",
            repo_type="dataset",
            local_dir=str(DATASCIBENCH_DIR),
        )

        print(f"✓ Downloaded to: {file_path}")

        # Extract
        print("→ Extracting ground truth data...")
        subprocess.run(["unzip", "-q", str(zip_file)], cwd=str(DATASCIBENCH_DIR), check=True)
        print("✓ Ground truth data extracted")

    except Exception as e:
        print(f"✗ Failed to download ground truth data: {e}")
        import traceback

        traceback.print_exc()


def copy_ground_truth_for_task(task_id: str):
    """Copy ground truth files for a specific task to data directory."""
    gt_source = DATASCIBENCH_DIR / "gt_data" / task_id / "gt"
    gt_dest = DATASCIBENCH_DIR / "data" / task_id / "gt"

    if not gt_source.exists():
        print(f"⚠ No ground truth data found for {task_id}")
        return

    if gt_dest.exists():
        print(f"✓ Ground truth already copied for {task_id}")
        return

    print(f"→ Copying ground truth for {task_id}...")
    shutil.copytree(gt_source, gt_dest)
    print(f"✓ Ground truth copied for {task_id}")


def run_evaluation(task_id: str, model_id: str):
    """Run evaluation for a specific task and model."""
    print(f"\n{'=' * 60}")
    print(f"Running evaluation for {task_id} with model {model_id}")
    print(f"{'=' * 60}\n")

    # Run evaluation script
    eval_cmd = [sys.executable, "-m", "experiments.evaluate", "--task_id", task_id, "--model_id", model_id]

    result = subprocess.run(eval_cmd, cwd=str(DATASCIBENCH_DIR), capture_output=False, text=True)

    if result.returncode == 0:
        print(f"\n✓ Evaluation completed for {task_id}")
    else:
        print(f"\n✗ Evaluation failed for {task_id}")

    return result.returncode == 0


def store_results(task_id: str, model_id: str, config_path: str):
    """Store results in benchmarks/DataSciBench/results/ directory."""
    # Determine results directory based on config
    config_name = Path(config_path).stem
    benchmark_dir = Path(__file__).parent
    results_dir = benchmark_dir / "results" / config_name / task_id
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n→ Storing results in {results_dir}")

    # Copy evaluation results CSV
    eval_results = DATASCIBENCH_DIR / "evaluation_results" / f"{model_id}_results.csv"
    if eval_results.exists():
        dest_eval = results_dir / f"{model_id}_results.csv"
        shutil.copy2(eval_results, dest_eval)
        print(f"  ✓ Copied {eval_results.name}")

    # Copy task outputs and logs
    data_dir = DATASCIBENCH_DIR / "data" / task_id
    for item in data_dir.glob(f"{model_id}_*"):
        if item.is_dir():
            dest_dir = results_dir / item.name
            if dest_dir.exists():
                shutil.rmtree(dest_dir)
            shutil.copytree(item, dest_dir)
            print(f"  ✓ Copied {item.name}/")
        elif item.suffix == ".jsonl":
            dest_file = results_dir / item.name
            shutil.copy2(item, dest_file)
            print(f"  ✓ Copied {item.name}")

    print(f"\n✓ Results stored in {results_dir}")

    # Create summary file
    summary_file = results_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Task: {task_id}\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Config: {config_name}\n")
        f.write(f"\nEvaluation Results: {model_id}_results.csv\n")
        f.write(f"Outputs: {model_id}_outputs.jsonl\n")
        f.write(f"Logs: {model_id}_0/\n")

    print(f"✓ Created summary: {summary_file}")


def get_model_id_from_config(config_path: str) -> str:
    """Extract model ID from config file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model = config.get("llm", {}).get("model", "unknown")
    # Clean up model name for use as ID
    model_id = model.split("/")[-1].replace(":", "-")
    return model_id


def main():
    """Run DataSciBench with compatibility patches."""
    # Apply compatibility patch
    patch_openai_httpx()

    # Parse arguments - check if --config is provided
    args = sys.argv[1:]
    config_path = None
    config_idx = None
    original_config_path = None

    for i, arg in enumerate(args):
        if arg == "--config" and i + 1 < len(args):
            config_path = args[i + 1]
            original_config_path = config_path  # Save original for later
            config_idx = i
            break

    # Get model ID from config
    model_id = None
    if original_config_path:
        model_id = get_model_id_from_config(original_config_path)
        print(f"→ Model ID: {model_id}")

    # Prepare config with env var expansion if config provided
    temp_config_path = None
    if config_path:
        temp_config_path = prepare_config(config_path)
        # Replace config path in arguments
        args[config_idx + 1] = temp_config_path

    # Download ground truth data if not already present
    print(f"\n{'=' * 60}")
    print("Checking ground truth data...")
    print(f"{'=' * 60}\n")
    download_ground_truth_data()

    # Change to DataSciBench directory
    original_dir = os.getcwd()
    os.chdir(DATASCIBENCH_DIR)

    try:
        # Import and run DataSciBench
        # We need to override sys.argv for argparse
        sys.argv = [sys.argv[0]] + args

        # Import the main module
        from experiments import run_examples

        # Run the main block
        import asyncio
        from dataclasses import asdict
        from src.logs import create_logger, get_model_name
        from metagpt.logs import logger
        import time
        import json
        from src.schemas import SciAgentBenchOutput
        from src.utils import change_dir, change_metalog_path

        # Get parsed arguments
        parsed_args = run_examples.get_args()

        # Load data folders
        data_dir = "data/"
        folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
        num_folders = len(folders)

        if parsed_args.task_id is None:
            task_id = folders
        else:
            task_id = parsed_args.task_id if "[" not in parsed_args.task_id else eval(parsed_args.task_id)

        if isinstance(task_id, str):
            folders = [f"{parsed_args.task_id}"]
            num_folders = 1
        elif isinstance(task_id, list):
            folders = task_id

        # Filter by data_source
        data_source_type = parsed_args.data_source_type
        filtered_folders = []
        for folder in folders:
            prompt_file = os.path.join(data_dir, folder, "prompt.json")
            if not os.path.exists(prompt_file):
                continue
            with open(prompt_file, "r") as file:
                prompt_data = eval(file.read())
                if data_source_type is None or prompt_data["data_source_type"].startswith(data_source_type):
                    filtered_folders.append(folder)

        folders = filtered_folders

        # Run tasks
        for id, folder in enumerate(folders):
            prompt_file = os.path.join(data_dir, folder, "prompt.json")

            # Copy ground truth for this task
            copy_ground_truth_for_task(folder)

            for sub_idx in range(parsed_args.max_runs):
                with open(prompt_file, "r") as file:
                    prompt_data = eval(file.read())

                if folder.startswith("bcb"):
                    result_logger, time_logger, log_dir, run_dir = create_logger(
                        folder, sub_idx, config_name=parsed_args.config, split=False
                    )
                else:
                    result_logger, time_logger, log_dir, run_dir = create_logger(
                        folder, sub_idx, config_name=parsed_args.config, split=True
                    )

                log_file_path = os.path.join(log_dir, "logs.txt")
                sys_log_file_path = os.path.join(log_dir, "sys_logs.txt")

                # Check if already completed
                sys_log = ""
                if os.path.exists(sys_log_file_path):
                    with open(sys_log_file_path, "r") as f:
                        sys_log = str(f.read())
                if "JSONDecodeError" in sys_log and "chatanywhere_error" not in sys_log:
                    print(f"Skipping folder {folder}")
                    continue
                if os.path.getsize(log_file_path) != 0 and not parsed_args.continue_gen:
                    print(f"Skipping folder {folder}")
                    continue

                # Get model name
                model_name = get_model_name(parsed_args.config)
                if not folder.startswith("bcb"):
                    model_name = model_name.split("/")[-1]
                output_dict_path = os.path.join(run_dir, f"{model_name}_outputs.jsonl")
                output_dict_path = os.path.abspath(output_dict_path)

                # Build requirement
                if not prompt_data["data_source_type"].startswith("1"):
                    requirement = run_examples.SPECIFY_PATH_PROMPT + prompt_data["prompt"]
                else:
                    requirement = prompt_data["prompt"]

                if "bcb" in folder and parsed_args.skip_bcb:
                    print(f"Skipping {folder}")
                    continue
                if parsed_args.data_type not in folder:
                    print(f"Skipping {folder}")
                    continue
                if parsed_args.gt_prompt is not None:
                    requirement = parsed_args.gt_prompt + "\n" + requirement

                sys_output_path = os.path.join(log_dir, "sys_logs.txt")
                print(sys_output_path)
                print(output_dict_path)

                # Run task with logging
                with change_metalog_path(logger=logger, file_path=sys_output_path) as temp_logger:
                    with change_dir(log_dir):
                        try:
                            temp_logger.info(f"Processing {folder} ({id}/{num_folders})")
                            temp_logger.info(f"Prompt:\n{requirement}")

                            time_logger.info(f"Processing {folder} ({id}/{num_folders})")
                            start_time = time.time()
                            plan_list, cost_list, error_counter_list = asyncio.run(
                                run_examples.main(requirement, parsed_args)
                            )
                            end_time = time.time()
                            elapsed_time = end_time - start_time

                            temp_logger.info(f"Completed processing folder {folder} ({id + 1}/{num_folders})")
                            temp_logger.info(f"Plan list:\n{plan_list}")
                            temp_logger.info(f"Cost list:\n{cost_list}")
                            temp_logger.info(f"Error counter list:\n{error_counter_list}")

                            result_logger.info(f"Plan list:\n{plan_list}")
                            result_logger.info(f"Cost list:\n{cost_list}")
                            result_logger.info(f"Error counter list:\n{error_counter_list}")

                            time_logger.info(f"Elapsed time: {elapsed_time:.2f} seconds")

                            output_dict = SciAgentBenchOutput(
                                output_dir=log_dir,
                                time_cost=elapsed_time,
                                error_list=error_counter_list[-1],
                                cost=cost_list[-1],
                                plan=plan_list[-1],
                            )
                            output_dict = asdict(output_dict)

                            with open(output_dict_path, "a") as f:
                                f.write(json.dumps(output_dict) + "\n")

                        except Exception as e:
                            temp_logger.info("=" * 52)
                            temp_logger.info(f"{e}\n{'=' * 52}")

            # After all runs for this task, run evaluation and store results
            if model_id and original_config_path:
                # Run evaluation
                run_evaluation(folder, model_id)

                # Store results
                store_results(folder, model_id, original_config_path)

    finally:
        # Clean up temp config
        if temp_config_path:
            try:
                Path(temp_config_path).unlink()
                print(f"✓ Cleaned up temporary config: {temp_config_path}")
            except:
                pass

        # Restore original directory
        os.chdir(original_dir)


if __name__ == "__main__":
    main()

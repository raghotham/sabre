#!/usr/bin/env python3
"""
Setup script for DataSciBench benchmarks

This script:
- Clones DataSciBench repository
- Creates Python 3.11 virtual environment (required for faiss-cpu)
- Installs MetaGPT and dependencies
- Downloads ground truth data from HuggingFace
"""

import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd: list[str], cwd: Path = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    return subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)


def download_ground_truth_data(datascibench_dir: Path, pip_cmd: Path):
    """Download ground truth data from HuggingFace if not already present."""
    import os

    print()
    print("=" * 60)
    print("Downloading Ground Truth Data")
    print("=" * 60)
    print()

    gt_data_dir = datascibench_dir / "gt_data"
    zip_file = datascibench_dir / "DataSciBench_GroundTruth_Data.zip"

    # Check if ground truth data already exists
    if gt_data_dir.exists() and any(gt_data_dir.iterdir()):
        print("✓ Ground truth data already present")
        return

    # Check if zip file exists but not extracted
    if zip_file.exists() and not gt_data_dir.exists():
        print("✓ Found ground truth zip, extracting...")
        subprocess.run(["unzip", "-q", str(zip_file)], cwd=str(datascibench_dir), check=True)
        print("✓ Ground truth data extracted")
        return

    # Get HF token from env var or file
    hf_token = os.environ.get("HF_TOKEN")
    hf_key_path = Path.home() / ".hf" / "key"

    if not hf_token and hf_key_path.exists():
        with open(hf_key_path, "r") as f:
            hf_token = f.read().strip()

    if not hf_token:
        print("❌ Error: HuggingFace token not found")
        print("   Please either:")
        print("   1. Set HF_TOKEN environment variable, or")
        print("   2. Create ~/.hf/key with your HuggingFace token")
        print()
        print("   Get your token from: https://huggingface.co/settings/tokens")
        sys.exit(1)

    print("→ Downloading ground truth data from HuggingFace...")

    try:
        # Install huggingface_hub if not already installed
        print("→ Installing huggingface_hub...")
        run_command([str(pip_cmd), "install", "-q", "huggingface_hub"])

        # Download using huggingface_hub (need to use the venv's python to import)
        download_script = f"""
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download, login

login(token='{hf_token}')

file_path = hf_hub_download(
    repo_id='zd21/DataSciBench',
    filename='DataSciBench_GroundTruth_Data.zip',
    repo_type='dataset',
    local_dir='{datascibench_dir}',
)

print(f"Downloaded to: {{file_path}}")
"""

        # Get python path from pip path
        if sys.platform == "win32":
            python_cmd = pip_cmd.parent / "python.exe"
        else:
            python_cmd = pip_cmd.parent / "python"

        result = subprocess.run(
            [str(python_cmd), "-c", download_script],
            cwd=str(datascibench_dir),
            capture_output=True,
            text=True,
            check=True,
        )

        print(f"✓ {result.stdout.strip()}")

        # Extract
        print("→ Extracting ground truth data...")
        subprocess.run(["unzip", "-q", str(zip_file)], cwd=str(datascibench_dir), check=True)
        print("✓ Ground truth data extracted (86.2 MB)")

    except Exception as e:
        print(f"❌ Failed to download ground truth data: {e}")
        sys.exit(1)


def main():
    # Setup paths
    script_dir = Path(__file__).parent.resolve()
    repo_root = script_dir.parent.parent
    tmp_dir = repo_root / "tmp"
    datascibench_dir = tmp_dir / "DataSciBench"

    print("=" * 60)
    print("DataSciBench Setup")
    print("=" * 60)
    print()

    # Check for Python 3.11
    python_cmd = shutil.which("python3.11")
    if not python_cmd:
        print("❌ Error: Python 3.11 is required but not found")
        print("   Install Python 3.11 and try again")
        sys.exit(1)

    print("✓ Found Python 3.11")

    # Create tmp directory
    tmp_dir.mkdir(exist_ok=True)

    # Clone DataSciBench if not exists
    if datascibench_dir.exists():
        print(f"✓ DataSciBench already cloned at {datascibench_dir}")
    else:
        print("→ Cloning DataSciBench...")
        run_command(["git", "clone", "https://github.com/LeonDLotter/DataSciBench.git"], cwd=tmp_dir)
        print("✓ Cloned DataSciBench")

    # Create virtual environment
    venv_dir = datascibench_dir / ".venv"
    if venv_dir.exists():
        print("✓ Virtual environment already exists")
    else:
        print("→ Creating Python 3.11 virtual environment...")
        run_command([python_cmd, "-m", "venv", ".venv"], cwd=datascibench_dir)
        print("✓ Created virtual environment")

    # Determine pip path based on platform
    if sys.platform == "win32":
        pip_cmd = venv_dir / "Scripts" / "pip"
    else:
        pip_cmd = venv_dir / "bin" / "pip"

    print("✓ Activated virtual environment")

    # Install MetaGPT
    metagpt_dir = datascibench_dir / "MetaGPT"
    if not metagpt_dir.exists():
        print("❌ Error: MetaGPT directory not found in DataSciBench")
        sys.exit(1)

    print("→ Installing MetaGPT...")
    run_command([str(pip_cmd), "install", "-q", "."], cwd=metagpt_dir)
    print("✓ Installed MetaGPT")

    # Install DataSciBench dependencies
    requirements_file = datascibench_dir / "requirements.txt"
    if not requirements_file.exists():
        print("❌ Error: requirements.txt not found")
        sys.exit(1)

    print("→ Installing DataSciBench dependencies...")
    run_command([str(pip_cmd), "install", "-q", "-r", "requirements.txt"], cwd=datascibench_dir)
    print("✓ Installed dependencies")

    # Download ground truth data
    download_ground_truth_data(datascibench_dir, pip_cmd)

    print()
    print("=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print()
    print(f"DataSciBench is installed at: {datascibench_dir}")
    print()
    print("Next steps:")
    print("Run benchmarks from the sabre root directory:")
    print(f"  cd {repo_root}")
    print("  OPENAI_API_KEY=`cat ~/.openai/key` python benchmarks/DataSciBench/run_benchmark.py \\")
    print("    --task_id csv_excel_0 \\")
    print("    --config benchmarks/DataSciBench/configs/config_gpt4o_baseline.yaml \\")
    print("    --data_type csv \\")
    print("    --max_runs 1")
    print()


if __name__ == "__main__":
    main()

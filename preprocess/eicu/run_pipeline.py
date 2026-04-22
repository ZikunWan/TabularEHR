"""
Run eICU preprocessing pipeline in a fixed order.

Order:
1) 1_build_cohorts.py
2) 2_prepare_tasks.py
3) 3_generate_sample_info.py
4) 4_partition_patients.py
5) 5_generate_embeddings.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run eICU preprocessing pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=str(Path(__file__).parent / "config.yaml"),
        help="Path to preprocess/eicu config.yaml",
    )
    parser.add_argument("--start-step", type=int, default=1, help="Start step in [1, 5]")
    parser.add_argument("--end-step", type=int, default=5, help="End step in [1, 5]")
    args = parser.parse_args()

    if not (1 <= args.start_step <= 5 and 1 <= args.end_step <= 5 and args.start_step <= args.end_step):
        raise ValueError("Invalid step range. Use 1 <= start-step <= end-step <= 5.")

    script_dir = Path(__file__).parent
    steps = [
        (1, script_dir / "1_build_cohorts.py", ["--config", args.config]),
        (2, script_dir / "2_prepare_tasks.py", ["--config", args.config]),
        (3, script_dir / "3_generate_sample_info.py", ["--config", args.config]),
        (4, script_dir / "4_partition_patients.py", []),
        (5, script_dir / "5_generate_embeddings.py", []),
    ]

    for step_id, script_path, extra_args in steps:
        if not (args.start_step <= step_id <= args.end_step):
            continue
        cmd = [sys.executable, str(script_path)] + extra_args
        print(f"[Step {step_id}] Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

    print("Pipeline completed.")


if __name__ == "__main__":
    main()

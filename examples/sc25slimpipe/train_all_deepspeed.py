#!/usr/bin/env python3

import argparse
import itertools
import logging
import os
import subprocess
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Define constants
HOME_DIR = os.path.expanduser("~")
BASE_DIR = f"{HOME_DIR}/Megatron-LM/examples/sc25slimpipe"
MEGATRON_PATH = f"{HOME_DIR}/github/Megatron-DeepSpeed"
DEEPSPEED_PATH = f"{HOME_DIR}/github/DeepSpeed"
ENTRY_POINT = "pretrain_llama_deepspeed.sh"
LOGS_DIR = f"{BASE_DIR}/logs/system_4M"


@dataclass
class ModelConfig:
    name: str
    file_name: str
    sp_value: int
    ep_value: int  # Added EP value as model-dependent


@dataclass
class ExperimentConfig:
    name: str
    models: list[ModelConfig]


class TrainingManager:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self.train_iters = 26
        self.global_token_size = 4 * 1024 * 1024  # 4M tokens
        self.seq_length_bases = [64, 128, 256, 512]
        self.np_options = [128, 256, 512]
        self.zero = 3
        self.recompute_options = ["full"]  # ["no", "full"]
        self.fpdt = 0
        self.fpdt_offload = 0

        # Define experiment configurations with model-specific EP values
        self.experiment_configs = [
            ExperimentConfig(
                name="deepspeed/llama",
                models=[
                    ModelConfig(
                        name="LLaMA-70B-GQA", file_name="70bgqa", sp_value=8, ep_value=1
                    ),
                    ModelConfig(
                        name="LLaMA-150B-GQA",
                        file_name="150bgqa",
                        sp_value=8,
                        ep_value=1,
                    ),
                ],
            ),
            ExperimentConfig(
                name="deepspeed/mixtral",
                models=[
                    ModelConfig(
                        name="Mixtral-8x7B", file_name="8x7b", sp_value=8, ep_value=8
                    ),
                    ModelConfig(
                        name="Mixtral-8x22B", file_name="8x22b", sp_value=8, ep_value=8
                    ),
                ],
            ),
        ]

    def switch_git_branch(self, repo_path, branch_name):
        """Switch to specified branch in git repository"""
        if self.dry_run:
            logger.info(
                f"[DRY RUN] Would switch to branch {branch_name} in {repo_path}"
            )
            return

        try:
            cwd = os.getcwd()
            os.chdir(repo_path)
            subprocess.run(["git", "switch", branch_name], check=True)
            os.chdir(cwd)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to switch branch in {repo_path}: {e}")
            exit(1)

    def run_experiment(self, experiment, model, np, seq_length_base, recompute):
        """Run a single experiment with the given parameters"""
        seq_length = seq_length_base * 1024
        # Calculate global batch size based on global token size
        global_batch_size = self.global_token_size // seq_length
        sp = model.sp_value
        ep = model.ep_value  # Use model-specific EP value
        tp = 1
        dp = np // (tp * sp * ep)

        # Print configuration summary
        logger.info(
            f"---------- Processing entry for {model.name} with {np} GPUs ----------"
        )
        logger.info(
            f"  Model: {model.name}, Global Batch Size: {global_batch_size}, Sequence Length: {seq_length}"
        )
        logger.info(
            f"  GPUs: {np}, TP: {tp}, SP: {sp}, EP: {ep}, DP: {dp}, Zero: {self.zero}"
        )
        logger.info(
            f"  FPDT: {self.fpdt}, FPDT Offload: {self.fpdt_offload}, Recompute: {recompute}"
        )
        if global_batch_size < dp:
            logger.warning(
                f"Global batch size {global_batch_size} is less than DP size {dp}. Skipping this configuration."
            )
            return

        # Switch git branches
        self.switch_git_branch(MEGATRON_PATH, "slimpipe")
        self.switch_git_branch(DEEPSPEED_PATH, "slimpipe")

        # Build environment variables dictionary
        env_vars = {
            "NP": str(np),
            "TRAIN_ITERS": str(self.train_iters),
            "GLOBAL_BATCH_SIZE": str(global_batch_size),
            "SEQUENCE_LENGTH": str(seq_length),
            "TP": str(tp),
            "SP": str(sp),
            "EP": str(ep),
            "ZERO": str(self.zero),
            "RECOMPUTE": str(recompute),
            "FPDT": str(self.fpdt),
            "FPDT_OFFLOAD": str(self.fpdt_offload),
            "MEGATRON_PATH": MEGATRON_PATH,
            "DEEPSPEED_PATH": DEEPSPEED_PATH,
            "LOG_DIR": f"{LOGS_DIR}/NP={np}/{model.name}/deepspeed/{seq_length_base}K/t{tp}s{sp}e{ep}d{dp}z{self.zero}-{recompute}",
            "EXP": experiment.name,
            "MODEL": model.file_name,
        }

        # Construct command
        cmd = f"{BASE_DIR}/{ENTRY_POINT}"

        # Display command
        env_str = " ".join([f"{k}={v}" for k, v in env_vars.items()])
        logger.info(f"Command: {env_str} {cmd}")
        logger.info(
            "------------------------------------------------------------------"
        )

        # Execute command with environment variables (unless dry run)
        if not self.dry_run:
            try:
                current_env = os.environ.copy()
                current_env.update(env_vars)
                subprocess.run(cmd, env=current_env, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Command failed with exit code {e.returncode}")
        else:
            logger.info(f"[DRY RUN] Would execute: {env_str} {cmd}")

    def run_all_experiments(self):
        """Run all combinations of experiment configurations"""
        # Generate all combinations of parameters
        combinations = itertools.product(
            self.experiment_configs,
            self.np_options,
            self.seq_length_bases,
            self.recompute_options,
        )

        for experiment_config, np, seq_length_base, recompute in combinations:
            for model in experiment_config.models:
                self.run_experiment(
                    experiment_config, model, np, seq_length_base, recompute
                )


if __name__ == "__main__":
    # Add command-line argument parsing
    parser = argparse.ArgumentParser(description="Training Manager for LLM experiments")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show commands without executing them"
    )
    args = parser.parse_args()

    # Initialize trainer with dry-run option
    trainer = TrainingManager(dry_run=args.dry_run)
    trainer.run_all_experiments()

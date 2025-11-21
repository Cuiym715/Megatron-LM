#!/usr/bin/env python3
"""
LLM Benchmark Runner Script

This script runs benchmarks for various LLM configurations including LLaMA and Mixtral models.
It calculates appropriate parameters and executes the training scripts with the right settings.
Supports reading and writing configurations from CSV files.
"""

import os
import subprocess
import argparse
import csv
from typing import Dict, List, Optional, Any


class LLMBenchmarkRunner:
    """Manager for running LLM benchmark experiments with different configurations."""

    def __init__(self, config_path: Optional[str] = None):
        # Common configuration
        home_dir = os.path.expanduser("~")
        self.base_dir = f"{home_dir}/Megatron-LM/examples/sc25slimpipe"
        self.megatron_path = f"{home_dir}/Megatron-LM"
        self.entry_point = "pretrain_llama.sh"
        self.logs_dir = f"{self.base_dir}/logs/extended"
        self.experiment_name = "slimpipe/llama"
        self.train_iters = 26
        self.record_memory_history = 0

        # CSV header mapping (CSV header -> internal parameter name)
        self.csv_mapping = {
            "#GPUs": "NP",
            "s": "SEQ_LENGTH",
            "Model": "MODEL",
            "B": "B",
            "b": "b",
            "t": "t",
            "c": "c",
            "p": "p",
            "e": "e",
            "d": "d",
            "n": "n",
            "v": "v",
            "l": "l",
            "ckpt": "RECOMPUTE",
            "offload ratio": "OFFLOAD_ALPHA",
        }

        # Reverse mapping for writing CSV
        self.reverse_csv_mapping = {v: k for k, v in self.csv_mapping.items()}

        # Define configurations
        self.config_path = config_path
        self.configs = self._initialize_configs()

    def _initialize_configs(self) -> Dict[str, List[Dict[str, Any]]]:
        """Initialize model configurations from CSV if provided, otherwise use defaults.

        Returns:
            A dictionary of model configurations where keys are model types and values are
            lists of configuration dictionaries.
        """
        configs = {}

        # Load from CSV if provided
        if self.config_path and os.path.exists(self.config_path):
            self._load_configs_from_csv()
            return self.configs

        # Default configurations (will be returned if no CSV is provided or it can't be loaded)
        llama_70b_config = {
            "MODEL": "LLaMA-70B-GQA",
            "MODEL_FILE": "70bgqa",
            "NP": 128,
            "t": 4,
            "c": 2,
            "p": 16,
            "n": 32,
            "B": 4,
            "RECOMPUTE": "partial+fc1",
            "OFFLOAD_ALPHA": 1,
            "DP_OVERLAP": 1,
            "TP_OVERLAP": 1,
            "ALLOC_CONF": 0,
            "SEQ_LENGTH": 2097152,  # 2048 * 1024
            "b": 1,  # micro-batch size
            "d": 1,  # calculated from other params usually
            "v": 1,  # some parameter
            "l": 1,  # virtual layers
        }

        llama_150b_config = {
            "MODEL": "LLaMA-150B-GQA",
            "MODEL_FILE": "150bgqa",
            "NP": 128,
            "t": 4,
            "c": 2,
            "p": 16,
            "n": 32,
            "B": 4,
            "RECOMPUTE": "partial+fc1",
            "OFFLOAD_ALPHA": 1,
            "DP_OVERLAP": 1,
            "TP_OVERLAP": 1,
            "ALLOC_CONF": 0,
            "SEQ_LENGTH": 2097152,  # 2048 * 1024
            "b": 1,  # micro-batch size
            "d": 1,  # calculated from other params usually
            "v": 1,  # some parameter
            "l": 1,  # virtual layers
        }

        mixtral_8x7b_config = {
            "MODEL": "Mixtral-8x7B",
            "MODEL_FILE": "8x7b",
            "NP": 256,
            "t": 1,
            "c": 16,
            "e": 8,
            "p": 8,
            "n": 16,
            "B": 4,
            "RECOMPUTE": "partial+fc1",
            "OFFLOAD_ALPHA": 1,
            "DP_OVERLAP": 1,
            "TP_OVERLAP": 1,
            "ALLOC_CONF": 0,
            "SEQ_LENGTH": 4194304,  # 4096 * 1024
            "b": 1,  # micro-batch size
            "d": 2,  # calculated from other params usually
            "v": 1,  # some parameter
            "l": 1,  # virtual layers
        }

        mixtral_8x22b_config = {
            "MODEL": "Mixtral-8x22B",
            "MODEL_FILE": "8x22b",
            "NP": 448,
            "t": 1,
            "c": 8,
            "e": 8,
            "p": 56,
            "n": 112,
            "B": 4,
            "RECOMPUTE": "partial+fc1",
            "OFFLOAD_ALPHA": 1,
            "DP_OVERLAP": 1,
            "TP_OVERLAP": 1,
            "ALLOC_CONF": 0,
            "SEQ_LENGTH": 4194304,  # 4096 * 1024
            "b": 1,  # micro-batch size
            "d": 1,  # calculated from other params usually
            "v": 1,  # some parameter
            "l": 1,  # virtual layers
        }

        configs["LLaMA-70B-GQA"] = [llama_70b_config]
        configs["LLaMA-150B-GQA"] = [llama_150b_config]
        configs["Mixtral-8x7B"] = [mixtral_8x7b_config]
        configs["Mixtral-8x22B"] = [mixtral_8x22b_config]

        return configs

    def _load_configs_from_csv(self) -> None:
        """Load configurations from a CSV file."""
        try:
            configs = {}
            with open(self.config_path, "r", newline="") as csvfile:
                reader = csv.DictReader(csvfile)

                # Validate headers
                required_headers = [
                    "#GPUs",
                    "s",
                    "Model",
                    "B",
                    "t",
                    "c",
                    "p",
                    "ckpt",
                    "offload ratio",
                ]
                missing_headers = [
                    h for h in required_headers if h not in reader.fieldnames
                ]
                if missing_headers:
                    print(
                        f"Warning: Missing required headers in CSV: {', '.join(missing_headers)}"
                    )
                    print("Using default configurations instead")
                    return

                for row in reader:
                    # Map CSV columns to internal parameter names
                    config = {}
                    for csv_name, internal_name in self.csv_mapping.items():
                        if csv_name in row:
                            value = row[csv_name].strip()

                            # Convert numeric values
                            if internal_name in [
                                "NP",
                                "B",
                                "b",
                                "t",
                                "c",
                                "p",
                                "n",
                                "d",
                                "v",
                                "l",
                                "SEQ_LENGTH",
                            ]:
                                if value:
                                    config[internal_name] = int(value)
                            elif internal_name in ["OFFLOAD_ALPHA"]:
                                if value:
                                    config[internal_name] = float(value)
                            else:
                                if value:
                                    config[internal_name] = value

                    # Set MODEL_FILE based on MODEL
                    model_name = config["MODEL"]
                    if "LLaMA-70B" in model_name:
                        config["MODEL_FILE"] = "70bgqa"
                    elif "LLaMA-150B" in model_name:
                        config["MODEL_FILE"] = "150bgqa"
                    elif "Mixtral-8x7B" in model_name:
                        config["MODEL_FILE"] = "8x7b"
                    elif "Mixtral-8x22B" in model_name:
                        config["MODEL_FILE"] = "8x22b"

                    # Add default values for required parameters that might be missing
                    config.setdefault("DP_OVERLAP", 0)
                    if config["t"] > 1:
                        config.setdefault("TP_OVERLAP", 1)
                    else:
                        config.setdefault("TP_OVERLAP", 0)
                    config.setdefault("ALLOC_CONF", 0)

                    # Add to appropriate model category
                    if model_name not in configs:
                        configs[model_name] = []
                    configs[model_name].append(config)

            if configs:
                self.configs = configs
                print(
                    f"Loaded {sum(len(configs[model]) for model in configs)} configurations from {self.config_path}"
                )
            else:
                print(f"No valid configurations found in {self.config_path}")

        except Exception as e:
            print(f"Error loading configurations from CSV: {e}")
            print("Using default configurations instead")

    def save_configs_to_csv(self, output_path: str) -> None:
        """Save current configurations to a CSV file.

        Args:
            output_path: Path where to save the configurations
        """
        try:
            with open(output_path, "w", newline="") as csvfile:
                fieldnames = [
                    "#GPUs",
                    "s",
                    "Model",
                    "B",
                    "b",
                    "t",
                    "c",
                    "p",
                    "e",
                    "d",
                    "n",
                    "v",
                    "l",
                    "ckpt",
                    "offload ratio",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for model_name, model_configs in self.configs.items():
                    for config in model_configs:
                        row = {}
                        # Map internal parameter names to CSV column names
                        for internal_name, value in config.items():
                            if internal_name in self.reverse_csv_mapping:
                                row[self.reverse_csv_mapping[internal_name]] = value

                        writer.writerow(row)

            print(f"Configurations saved to {output_path}")
        except Exception as e:
            print(f"Error saving configurations to CSV: {e}")

    @staticmethod
    def ceil_div(n: int, divisor: int) -> int:
        """Calculate ceiling division."""
        return (n + divisor - 1) // divisor

    @staticmethod
    def round_up_to_multiple(num: int, multiple_of: int) -> int:
        """Round up to the nearest multiple."""
        remainder = num % multiple_of
        if remainder == 0:
            return num
        return num + multiple_of - remainder

    def run_config(self, model_name: str, config_index: int = 0) -> None:
        """Run a specific configuration.

        Args:
            model_name: Name of the model
            config_index: Index of the configuration to run (default: 0)
        """
        if model_name not in self.configs:
            print(f"Error: Model '{model_name}' not found")
            self.list_configs()
            return

        if config_index >= len(self.configs[model_name]):
            print(
                f"Error: Configuration index {config_index} out of range for model '{model_name}'"
            )
            return

        config = self.configs[model_name][config_index]
        print(
            f"---------- Running configuration: {model_name} (Config #{config_index}) ----------"
        )

        # Switch to the correct branch
        current_dir = os.getcwd()
        os.chdir(self.megatron_path)
        subprocess.run(["git", "switch", "wei/attn_balance"], check=True)
        os.chdir(current_dir)

        # Extract parameters from config
        seq_length = config["SEQ_LENGTH"]
        n = config["n"]

        # Calculate parameters
        standard_slice_length = self.ceil_div(seq_length, n)
        micro_seq_length = self.round_up_to_multiple(standard_slice_length, 128)
        actual_seq_length = micro_seq_length * n
        padding = actual_seq_length - seq_length

        # Calculate derived parameter d if not explicitly specified
        if "d" not in config or config["d"] == 0:
            d = config["NP"] // config["t"] // config["c"] // config["p"]
            config["d"] = d
        else:
            d = config["d"]

        # Get the value for l (virtual layers)
        l = config.get("l", 1)

        # Print configuration summary
        print(f"  Model: {config['MODEL']}, GPUs: {config['NP']}, Batch: {config['B']}")
        print(f"  TP: {config['t']}, CP: {config['c']}")

        if "e" in config:
            print(f"  EP: {config['e']}")
            dir_suffix = f"t{config['t']}c{config['c']}p{config['p']}d{d}n{config['n']}e{config['e']}-{config['RECOMPUTE']}-{config['OFFLOAD_ALPHA']}"
        else:
            dir_suffix = f"t{config['t']}c{config['c']}p{config['p']}d{d}n{config['n']}-{config['RECOMPUTE']}-{config['OFFLOAD_ALPHA']}"

        print(f"  PP: {config['p']}, DP: {d}")
        print(f"  Slices: {config['n']}, Virtual Layers: {l}")
        print(f"  Sequence Length: {seq_length}, Padding: {padding}")
        print(
            f"  Actual Sequence Length: {actual_seq_length}, Micro Sequence Length: {micro_seq_length}"
        )
        print(f"  Recompute: {config['RECOMPUTE']}, Offload: {config['OFFLOAD_ALPHA']}")

        # Add sequence length to log path for better organization
        seq_length_kb = seq_length // 1024
        log_path = f"{self.logs_dir}/{config['MODEL']}/{seq_length_kb}k/{dir_suffix}"

        # Build environment variables for the command
        env_vars = {
            "NP": str(config["NP"]),
            "RECORD_MEMORY_HISTORY": str(self.record_memory_history),
            "TRAIN_ITERS": str(self.train_iters),
            "GLOBAL_BATCH_SIZE": str(config["B"]),
            "TP": str(config["t"]),
            "CP": str(config["c"]),
            "PP": str(config["p"]),
            "PP_l": str(l),
            "RECOMPUTE": config["RECOMPUTE"],
            "OFFLOAD_ALPHA": str(config["OFFLOAD_ALPHA"]),
            "DP_OVERLAP": str(config.get("DP_OVERLAP", 1)),
            "TP_OVERLAP": str(config.get("TP_OVERLAP", 1)),
            "ALLOC_CONF": str(config.get("ALLOC_CONF", 0)),
            "SEQUENCE_LENGTH": str(actual_seq_length),
            "NUM_SLICES": str(config["n"]),
            "MICRO_SEQ_LENGTH": str(micro_seq_length),
            "MEGATRON_PATH": self.megatron_path,
            "LOG_DIR": log_path,
            "EXP": self.experiment_name,
            "MODEL": config["MODEL_FILE"],
        }

        # Add micro batch size if present
        if "b" in config:
            env_vars["MICRO_BATCH_SIZE"] = str(config["b"])

        # Add EP if present
        if "e" in config:
            env_vars["EP"] = str(config["e"])

        # Create environment with existing env plus our additions
        cmd_env = os.environ.copy()
        cmd_env.update(env_vars)

        # Display command summary
        print(
            f"  Running with sequence length {seq_length_kb}k and offload alpha {config['OFFLOAD_ALPHA']}"
        )
        print("  Command environment variables:")
        for key, value in env_vars.items():
            print(f"    {key}={value}")

        script_path = f"{self.base_dir}/{self.entry_point}"
        print(f"  Running: {script_path}")
        print("----------------------------------------------------")

        # Execute the command
        try:
            subprocess.run(script_path, env=cmd_env, check=True)
        except subprocess.CalledProcessError as e:
            print(
                f"Error running configuration for {model_name} (Config #{config_index}): {e}"
            )

    def list_configs(self) -> None:
        """List all available configurations."""
        print("Available configurations:")
        for model_name, model_configs in self.configs.items():
            print(f"  Model: {model_name}")
            for idx, config in enumerate(model_configs):
                seq_length_kb = config["SEQ_LENGTH"] // 1024
                print(
                    f"    {idx}: GPUs: {config['NP']}, Seq Length: {seq_length_kb}k, Batch: {config['B']}"
                )
                print(
                    f"       TP: {config['t']}, CP: {config['c']}, PP: {config['p']}, Slices: {config['n']}"
                )
                if "e" in config:
                    print(f"       EP: {config['e']}")
                print(
                    f"       Recompute: {config['RECOMPUTE']}, Offload: {config['OFFLOAD_ALPHA']}"
                )
            print()

    def run_all_configs(self) -> None:
        """Run all available configurations."""
        for model_name, model_configs in self.configs.items():
            for idx in range(len(model_configs)):
                self.run_config(model_name, idx)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run LLM benchmark experiments")
    parser.add_argument("--model", help="Model name to run")
    parser.add_argument(
        "--config-index",
        type=int,
        default=0,
        help="Configuration index to run (default: 0)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List available configurations"
    )
    parser.add_argument("--config-file", help="Path to CSV configuration file")
    parser.add_argument(
        "--save-config", help="Save current configurations to specified CSV file"
    )
    parser.add_argument(
        "--run-all", action="store_true", help="Run all available configurations"
    )

    args = parser.parse_args()

    runner = LLMBenchmarkRunner(config_path=args.config_file)

    if args.save_config:
        runner.save_configs_to_csv(args.save_config)

    if args.list:
        runner.list_configs()
    elif args.run_all:
        runner.run_all_configs()
    elif args.model:
        runner.run_config(args.model, args.config_index)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import csv
import os
import subprocess
import sys
import time

# Check arguments
if len(sys.argv) != 2:
    print(f"Usage: {sys.argv[0]} <csv_file>")
    sys.exit(1)

csv_file = sys.argv[1]

# Validate file exists
if not os.path.isfile(csv_file):
    print(f"Error: File {csv_file} not found")
    sys.exit(1)

# Define constants
TRAIN_ITERS = 5
HOME_DIR = os.path.expanduser("~")
BASE_DIR = f"{HOME_DIR}/Megatron-LM/examples/sc25slimpipe"
RECORD_MEMORY_HISTORY = 0

# Model configuration maps
SLIMPIPE_EXPERIMENTS = {
    "LLaMA-13B": "slimpipe/llama",
    "LLaMA-70B-GQA": "slimpipe/llama",
    "LLaMA-175B": "slimpipe/llama",
    "LLaMA-150B-GQA": "slimpipe/llama",
    "Mixtral-8x7B": "slimpipe/mixtral",
    "Mixtral-8x22B": "slimpipe/mixtral",
}

# Model configuration maps
MEGATRON_EXPERIMENTS = {
    "LLaMA-13B": "megatron/llama",
    "LLaMA-70B-GQA": "megatron/llama",
    "LLaMA-175B": "megatron/llama",
    "LLaMA-150B-GQA": "megatron/llama",
    "Mixtral-8x7B": "megatron/mixtral",
    "Mixtral-8x22B": "megatron/mixtral",
}

MODEL_FILES = {
    "LLaMA-13B": "13b",
    "LLaMA-70B-GQA": "70bgqa",
    "LLaMA-175B": "175b",
    "LLaMA-150B-GQA": "150bgqa",
    "Mixtral-8x7B": "8x7b",
    "Mixtral-8x22B": "8x22b",
}

# Process CSV file
with open(csv_file, "r") as f:
    reader = csv.reader(f)
    header = next(reader)  # Read header row

    # Create sanitized column names
    column_names = [col.replace("#", "_").replace(" ", "_") for col in header]

    # Process each data row
    for row in reader:
        # Skipped commented rows
        if row[0].startswith("#"):
            continue

        # Create a dictionary mapping column names to values
        row_data = dict(zip(column_names, row))

        # Print configuration summary
        print(
            f"---------- Processing entry for {row_data['Model']} with {row_data['_GPUs']} GPUs ----------"
        )
        print(
            f"  Model: {row_data['Model']}, GPUs: {row_data['_GPUs']}, Batch: {row_data['B']}, Microbatch: {row_data['b']} TP: {row_data['t']}, CP: {row_data['c']}, PP: {row_data['p']}"
        )
        print(
            f"  EP: {row_data['e']}, DP: {row_data['d']}, Slices: {row_data['n']}, Trunks: {row_data['v']}, Virtual Layers: {row_data['l']}"
        )
        print(
            f"  Recompute: {row_data['recompute']}, Offload: {row_data.get('offload', '0')}"
        )

        # Set up training parameters
        EXPERIMENT_NAME = (
            SLIMPIPE_EXPERIMENTS[row_data["Model"]]
            if int(row_data["n"]) > 1
            else MEGATRON_EXPERIMENTS[row_data["Model"]]
        )
        MODEL_FILE = MODEL_FILES[row_data["Model"]]
        SEQ_LENGTH = int(row_data["s"])
        MICRO_SEQ_LENGTH = SEQ_LENGTH // int(row_data["n"])
        RECOMPUTE = row_data["recompute"]
        POST_LM_PP = "1" if int(row_data["n"]) > 1 else "0"

        global_token_size = int(row_data["B"]) * SEQ_LENGTH // 1024 // 1024

        MEGATRON_PATH = f"{HOME_DIR}/Megatron-LM"
        ENTRY_POINT = "pretrain_llama.sh"
        LOGS_DIR = f"{BASE_DIR}/logs/system_{global_token_size}M/NP={row_data['_GPUs']}/{row_data['Model']}"
        if int(row_data["n"]) > 1:
            LOGS_DIR += "/slimpipe"
        else:
            LOGS_DIR += "/kwai"

        def round_down_to_nearest_power_of_2(n):
            if n <= 0:
                raise ValueError("Input must be a positive integer.")
            return 2 ** (n.bit_length() - 1)

        LOGS_DIR += f"/{round_down_to_nearest_power_of_2(SEQ_LENGTH // 1024)}K"

        # Change to Megatron directory and switch branch
        os.chdir(MEGATRON_PATH)
        subprocess.run(["git", "switch", "wei/attn_balance"])
        os.chdir(BASE_DIR)

        # Build environment variables for command
        env = os.environ.copy()
        env.update(
            {
                "NP": row_data["_GPUs"],
                "RECORD_MEMORY_HISTORY": str(RECORD_MEMORY_HISTORY),
                "PROFILE": "0",
                "TORCH_CUDA_ARCH_LIST": "9.0",
                "TRAIN_ITERS": str(TRAIN_ITERS),
                "GLOBAL_BATCH_SIZE": row_data["B"],
                "MICRO_BATCH_SIZE": row_data["b"],
                "TP": row_data["t"],
                "CP": row_data["c"],
                "PP": row_data["p"],
                "EP": row_data["e"],
                "RECOMPUTE": RECOMPUTE,
                "OFFLOAD_ALPHA": row_data.get("offload", "0"),
                "DP_OVERLAP": "0",
                "TP_OVERLAP": "1",
                "SEQUENCE_LENGTH": str(SEQ_LENGTH),
                "MEGATRON_PATH": MEGATRON_PATH,
                "LOG_DIR": f"{LOGS_DIR}/t{row_data['t']}c{row_data['c']}p{row_data['p']}e{row_data['e']}d{row_data['d']}n{row_data['n']}v{row_data['v']}-{RECOMPUTE}",
                "EXP": EXPERIMENT_NAME,
                "MODEL": MODEL_FILE,
            }
        )
        if int(row_data["n"]) > 1:
            env.update(
                {
                    "NUM_SLICES": row_data["n"],
                    "MICRO_SEQ_LENGTH": str(MICRO_SEQ_LENGTH),
                }
            )

        # Add PP_l if v > 1
        if int(row_data["v"]) > 1:
            env["PP_l"] = row_data["l"]

        # Build command
        cmd = f"{BASE_DIR}/{ENTRY_POINT}"

        # Display command
        print(f"Command: {cmd}")
        print("------------------------------------------------------------------")

        # Execute command
        subprocess.run(cmd, env=env, shell=True)

        print("Waiting for 10 seconds before next run...")
        time.sleep(10)

This repository is a fork of [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/). The original README is [here](README_orig.md).

# Efficient Activation Rematerialization & SlimPipe

*Efficient Activation Rematerialization* contains two strategies: Pipeline-Parallel-Aware Offloading, which maximizes the utilization of host memory for storing activations, and Compute-Memory Balanced Checkpointing, which seeks a practical equilibrium between activation memory and computational efficiency. Based on these two strategies, one can further optimize the hybrid parallelism configurations.

*SlimPipe* is a fine-grained pipeline parallelism that employs uniform sequence slicing coupled with 1F1B schedule. It is equipped with a sophisticated workload redistribution technique to balance the computation. As a result, SlimPipe reduces the memory overhead and pipeline bubbbles simultaneously.

These technologies are introduced by two of our papers:

[[USENIX ATC '24] Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism](https://www.usenix.org/conference/atc24/presentation/yuan)


[[SC '25] SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training](https://dl.acm.org/doi/10.1145/3712285.3759855)


## Features

This repository implements the following technologies:
- Pipeline-Parallelism-Aware Offloading
- Compute-Memory Balanced Checkpointing
- Workload Balanced Context Parallelism
- [SlimPipe] Uniform Slicing Pipeline Parallelsim 
- [SlimPipe] Context Exchange in Pipeline World 
- [SlimPipe] Vocabulary Parallelsim in Pipeline World
- Context Parallelsim Passing Query and Output


## Citation

```bibtex
@inproceedings{10.5555/3691992.3692026,
author = {Yuan, Tailing and Liu, Yuliang and Ye, Xucheng and Zhang, Shenglong and Tan, Jianchao and Chen, Bin and Song, Chengru and Zhang, Di},
title = {Accelerating the training of large language models using efficient activation rematerialization and optimal hybrid parallelism},
year = {2024},
booktitle = {Proceedings of the 2024 USENIX Conference on Usenix Annual Technical Conference},
}

@inproceedings{10.1145/3712285.3759855,
    author = {Li, Zhouyang and Liu, Yuliang and Zhang, Wei and Yuan, Tailing and Chen, Bin and Song, Chengru},
    title = {SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training},
    year = {2025},
    url = {https://doi.org/10.1145/3712285.3759855},
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
}
```

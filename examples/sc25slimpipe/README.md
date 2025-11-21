# Introduction
This directory contains scripts used to reproduce the results in *SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training*.
The [paper](https://dl.acm.org/doi/10.1145/3712285.3759855) is to appear at the International Conference for High Performance Computing, Networking, Storage, and Analysis (SC25).


# Instructions
1. Download the tokenizer. You may first apply for access on the HuggingFace website.
```bash
git clone https://huggingface.co/meta-llama/Llama-3.2-1B
```

2. Download the dataset and preprocess it.
```bash
./make_dataset.sh
```

3. (Optional) You can use your own tokenizer or dataset other than the preseted.
Directly modify the paths in `./dataset/sample_wiki_128000`.
```bash
DATA_ARGS="
    --data-path /path/to/your/data \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model /path/to/your/tokenizer \
    --split 100,0,0 \
"
```

4. Run the example experiment.
```bash
EXP=llama ./pretrain_llama.sh
```

5. To view and modify the experiment settings, just edit `./exp/llama`.


# Citation
```bibtex
@inproceedings{10.1145/3712285.3759855,
    author = {Li, Zhouyang and Liu, Yuliang and Zhang, Wei and Yuan, Tailing and Chen, Bin and Song, Chengru},
    title = {SlimPipe: Memory-Thrifty and Efficient Pipeline Parallelism for Long-Context LLM Training},
    year = {2025},
    url = {https://doi.org/10.1145/3712285.3759855},
    booktitle = {Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
}
```

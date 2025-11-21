#!/bin/bash

pip install nltk

python3 make_dataset.py

python3 ../../tools/preprocess_data.py \
       --input ./owt-sample.json \
       --workers 32 \
       --json-keys text \
       --output-prefix ./dataset/owt-sample \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model ./Llama-3.2-1B \
        --chunk-size 25 \
       --append-eod 



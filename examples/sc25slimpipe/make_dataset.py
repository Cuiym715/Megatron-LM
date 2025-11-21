from datasets import load_dataset

train_data = load_dataset("stanford-cs336/owt-sample", split='train')

train_data.to_json("owt-sample.json", lines=True)


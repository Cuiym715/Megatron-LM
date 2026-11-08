"""Construct reproducible CP document datasets; never overwrite existing data."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from megatron.data import indexed_dataset


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--repeats',type=int,default=16)
    p.add_argument('--vocab-size',type=int,default=32001)
    p.add_argument('--lengths',type=int,nargs='+',default=[800,224,600,424,1500,548,700,700,648,4608,768,768,6401,2305,97,31])
    cfg=p.parse_args()
    output=Path(cfg.output)
    if output.with_suffix('.bin').exists() or output.with_suffix('.idx').exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    builder=indexed_dataset.make_builder(str(output)+'.bin',impl='mmap',vocab_size=cfg.vocab_size)
    for repeat in range(cfg.repeats):
        for index,length in enumerate(cfg.lengths):
            tokens=(torch.arange(length+1,dtype=torch.int32)+repeat*313+index*19)%(cfg.vocab_size-1)+1
            builder.add_item(tokens)
            builder.end_document()
    builder.finalize(str(output)+'.idx')
    print(dict(output=str(output),training_lengths=cfg.lengths,repeats=cfg.repeats))


if __name__=='__main__':
    main()

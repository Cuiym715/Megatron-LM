from argparse import Namespace
import os
import sys
import shutil

import numpy as np
import torch
import torch.multiprocessing as mp


MEGATRON_PATH = '/home/liuyining/code/Megatron-LM/'
sys.path.insert(0, MEGATRON_PATH)

# We currently do not support rng_state in ckp
CKP_IGNORE_KEYS = {'rng_state'}

# We currently do not support these arguments
ARGS_IGNORE_KEYS = {
    'master_addr', 'save', 'no_load_optim', 'no_save_optim', 'no_save_rng', 'lr_decay_iters',
    'use_cpu_initialization', 'perform_initialization', 'train_iters', 'save_interval',
    'lr_warmup_iters', 'sequence_parallel', 'bias_dropout_fusion', 'iteration', 'do_test',
    'curr_iteration', 'do_valid', 'do_train', 'allow_transformer_engine', 'rank', 'local_rank',
}
MODEL_FILE = 'model_optim_rng.pt'
OPTIM_FILE = 'distrib_optim.pt'
# We currently do not support these files
IGNORE_CKP_PATHS = {'tokenizer_config.json', 'tokenizer.model',
                    'eval_result', 'dataloader.pt', 'config.json'}


def diff_dict(name, c1: dict, c2: dict, ignore_keys: set = set(), tensor_diff=None):
    assert (isinstance(c1, dict))
    assert (isinstance(c2, dict))
    diff = []
    missing_keys = set(c1.keys()) - set(c2.keys()) - ignore_keys
    if len(missing_keys) > 0:
        diff.append(f'{name} missing keys: {missing_keys}')
    unexpected_keys = set(c2.keys()) - set(c1.keys()) - ignore_keys
    if len(unexpected_keys) > 0:
        diff.append(f'{name} unexpected keys: {unexpected_keys}')
    for k in c1.keys() & c2.keys() - ignore_keys:
        diff += diff_ckp(f'{name}[{k}]', c1[k],
                         c2[k], tensor_diff=tensor_diff)
    return diff


def default_tensor_diff(name, t1, t2):
    if t1.shape != t2.shape:
        return f'{name} shape mismatch: {t1.shape} != {t2.shape}'
    elif t1.dtype != t2.dtype:
        return f'{name} dtype mismatch: {t1.dtype} != {t2.dtype}'
    d = torch.max(torch.abs(t1 - t2)).item()
    if d > 0:
        return f'{name} max diff: {d}'
    return None


def buffer_tensor_diff_pad_zeros(name: str, t1: torch.Tensor, t2: torch.Tensor):
    """
    t1 and t2 are contiguous buffer, compare them ignoring tailing padding elements
    """
    if t1.dtype != t2.dtype:
        return f'{name} dtype mismatch: {t1.dtype} != {t2.dtype}'
    # compare shape, ignore tailing padding elements
    if len(t1.shape) != 1 or len(t2.shape) != 1:
        return f'{name} buffer shape must be 1D'
    real_size = min(t1.shape[0], t2.shape[0])
    if t1.shape[0] > real_size:
        print(
            f'{name} ignore tailing padding elements num: {t1.shape[0] - real_size}')
        t1 = t1[:real_size]
    if t2.shape[0] > real_size:
        print(
            f'{name} ignore tailing padding elements num: {t2.shape[0] - real_size}')
        t2 = t2[:real_size]
    # compare values
    d = torch.max(torch.abs(t1 - t2)).item()
    if d > 0:
        return f'{name} max diff: {d}'
    return None


def diff_ckp(name, c1, c2, tensor_diff=None):
    diff = []
    if isinstance(c1, dict) and isinstance(c2, dict):
        diff += diff_dict(name, c1, c2, tensor_diff=tensor_diff)
    elif (isinstance(c1, list) and isinstance(c2, list)) or \
            (isinstance(c1, tuple) and isinstance(c2, tuple)):
        if len(c1) != len(c2):
            diff.append(f'{name} length mismatch: {len(c1)} != {len(c2)}')
        else:
            for i in range(len(c1)):
                diff += diff_ckp(f'{name}[{i}]', c1[i],
                                 c2[i], tensor_diff=tensor_diff)
    elif isinstance(c1, torch.Tensor) and isinstance(c2, torch.Tensor):
        d = tensor_diff(name, c1, c2) \
            if tensor_diff is not None else default_tensor_diff(name, c1, c2)
        if d is not None:
            diff.append(d)
    elif isinstance(c1, Namespace) and isinstance(c2, Namespace):
        diff += diff_dict(name, vars(c1), vars(c2),
                          ignore_keys=ARGS_IGNORE_KEYS)
    elif isinstance(c1, np.ndarray) and isinstance(c2, np.ndarray):
        if not np.array_equal(c1, c2):
            diff.append(f'{name} np array not equal')
    elif type(c1) != type(c2):
        diff.append(f'{name} type mismatch: {type(c1)} != {type(c2)}')
    else:
        # print(name, type(c1), type(c2))
        if c1 != c2:
            diff.append(f'{name} mismatch: {c1} != {c2}')
    return diff


def remove_keys(ckp: dict, ignore_args=False):
    keys = {'args'} if ignore_args else set()
    keys |= CKP_IGNORE_KEYS
    # print(f'Removing keys: {keys}')
    for k in keys:
        if k in ckp.keys():
            ckp.pop(k)
    return ckp


def diff_ckp_model_file(file1, file2, name, ignore_args=False):
    ckp1 = torch.load(file1, map_location='cpu')
    ckp2 = torch.load(file2, map_location='cpu')

    # Remove items we currently not support
    ckp1 = remove_keys(ckp1, ignore_args)
    ckp2 = remove_keys(ckp2, ignore_args)

    diff = diff_ckp('model', ckp1, ckp2)
    print(f'Comparing {file1} and {file2}, diff cnt: {len(diff)}')
    diff = [f'{name}: {d}' for d in diff]
    return diff


def diff_ckp_optimizer_file(file1, file2, name):
    ckp1 = torch.load(file1, map_location='cpu')
    ckp2 = torch.load(file2, map_location='cpu')

    diff = diff_ckp('optimizer', ckp1, ckp2,
                    tensor_diff=buffer_tensor_diff_pad_zeros)
    print(f'Comparing {file1} and {file2}, diff cnt: {len(diff)}')
    diff = [f'{name}: {d}' for d in diff]
    return diff


def diff_ckp_dir(subdirs1, subdirs2, ignore_args=False):
    diff = []
    missing_dirs = set(subdirs1.keys()) - set(subdirs2.keys())
    unexpected_dirs = set(subdirs2.keys()) - set(subdirs1.keys())
    if len(missing_dirs) > 0:
        diff.append(f'missing dirs: {missing_dirs}')
    if len(unexpected_dirs) > 0:
        diff.append(f'unexpected dirs: {unexpected_dirs}')
    for subdir in subdirs1.keys() & subdirs2.keys():
        diff += diff_ckp_model_file(
            os.path.join(subdirs1[subdir], MODEL_FILE),
            os.path.join(subdirs2[subdir], MODEL_FILE),
            subdir, ignore_args)
        if len(diff) > 0:
            break
        diff += diff_ckp_optimizer_file(
            os.path.join(subdirs1[subdir], OPTIM_FILE),
            os.path.join(subdirs2[subdir], OPTIM_FILE),
            subdir)
        if len(diff) > 0:
            break

    if len(diff) > 0:
        print('Below are diffs:')
        for d in diff:
            print(d)
        assert False
    else:
        print(f'No diffs found')


def diff_checkpoint(path1, path2, ignore_args=False):
    def get_iteration(path):
        from megatron.checkpointing import get_checkpoint_tracker_filename, read_metadata
        # Read the tracker file and set the iteration.
        tracker_filename = get_checkpoint_tracker_filename(path)
        assert (os.path.isfile(tracker_filename))
        # Read the tracker file and either set the iteration or
        # mark it as a release checkpoint.
        iteration, _ = read_metadata(tracker_filename)
        return iteration

    # Get iteration
    iteration = get_iteration(path1)
    assert (iteration == get_iteration(path2))
    # Diff path1/iter_{iteration} and path2/iter_{iteration}
    directory = 'iter_{:07d}'.format(iteration)
    ckp_path1 = os.path.join(path1, directory)
    ckp_path2 = os.path.join(path2, directory)

    # Get subdirectories of ckp_path1 and ckp_path2 as a dict
    subdirs1 = {o: os.path.join(ckp_path1, o) for o in os.listdir(ckp_path1)
                if o not in IGNORE_CKP_PATHS}
    subdirs2 = {o: os.path.join(ckp_path2, o) for o in os.listdir(ckp_path2)
                if o not in IGNORE_CKP_PATHS}

    print(f'Comparing {path1} and {path2}')
    # We should work on a separate process to avoid some global variable conflicts
    new_process_run(diff_ckp_dir, [subdirs1, subdirs2, ignore_args])


def run_concert_ckp_main(argv):
    sys.argv = argv
    sys.path.insert(0, os.path.join(MEGATRON_PATH, 'tools'))
    from checkpoint_util import main as checkpoint_util_main
    checkpoint_util_main()


def convert_ckp(input_path, output_path, tp_size, pp_size, vp_size):
    argv = ['script.py',
            '--megatron-path', MEGATRON_PATH,
            '--loader', 'megatron',
            '--saver', 'megatron',
            '--model-type', 'LLAMA',
            '--load-dir', input_path,
            '--save-dir', output_path,
            '--target-tensor-parallel-size', str(tp_size),
            '--target-pipeline-parallel-size', str(pp_size),
            '--target-virtual-pipeline-parallel-size', str(vp_size),
            '--process-optimizer',
            ]
    # We should work on a separate process to avoid some global variable conflicts
    new_process_run(run_concert_ckp_main, [argv])


def new_process_run(work, args=()):
    proc = mp.Process(target=work, args=args)
    proc.start()
    proc.join()
    assert (proc.exitcode == 0)


def main():
    torch.multiprocessing.set_start_method('spawn')

    # 13B LLaMA checkpoint with 2 TP, 4 PP, 5 VPP
    input_2_4_5 = '/nlp_group/liuyining/glb/input_megatron/13b_2_4_5'
    output_path = '/nlp_group/liuyining/glb/output_megatron'
    input_8_4_5 = '/nlp_group/liuyining/glb/input_megatron/13b_8_4_5'
    shutil.rmtree(output_path, ignore_errors=True)

    # Convert to 8 TP, 4 PP, 5 VPP
    # output_8_4_5 = os.path.join(output_path, '8_4_5')
    # convert_ckp(input_2_4_5, output_8_4_5, 8, 4, 5)
    # diff_checkpoint(input_8_4_5, output_8_4_5, ignore_args=True)

    # Convert to 4 TP, 8 PP, 1 VPP
    output_4_8_1 = os.path.join(output_path, '4_8_1')
    convert_ckp(input_2_4_5, output_4_8_1, 4, 8, 1)
    # Convert to 2 TP, 4 PP, 5 VPP
    output_2_4_5 = os.path.join(output_path, '2_4_5')
    convert_ckp(output_4_8_1, output_2_4_5, 2, 4, 5)
    # Diff 2_4_5 with original
    diff_checkpoint(input_2_4_5, output_2_4_5)
    # Convert to 8 TP, 4 PP, 5 VPP
    output_8_4_5 = os.path.join(output_path, '8_4_5')
    convert_ckp(output_4_8_1, output_8_4_5, 8, 4, 5)
    # Diff 8_4_5 with original
    diff_checkpoint(input_8_4_5, output_8_4_5, ignore_args=True)


if __name__ == '__main__':
    main()

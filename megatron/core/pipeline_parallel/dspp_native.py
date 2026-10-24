"""CPU C++ ready construction, adapted from read-only research 027/028.

Compilation is content-addressed and locked, outside measured warm planning.
No torch extension, GPU dependency, research import or algorithm selection.
"""
import ctypes as ct
import fcntl
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shlex
import subprocess
import tempfile


@lru_cache(maxsize=1)
def library():
    source = Path(__file__).with_name('dspp_planner.cpp')
    compiler = shlex.split(os.environ.get('CXX', 'c++'))
    version = subprocess.check_output([*compiler, '--version'])
    key = hashlib.sha256(source.read_bytes() + version + platform.machine().encode()).hexdigest()
    cache = Path(os.environ.get('DSPP_PLANNER_CACHE', Path(tempfile.gettempdir()) / 'dspp-planner'))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f'{key}.so'
    with (cache / f'{key}.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = cache / f'{key}.{os.getpid()}.so'
            try:
                subprocess.run([*compiler, '-std=c++17', '-O3', '-shared', '-fPIC',
                                str(source), '-o', str(temporary)], check=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    lib = ct.CDLL(str(target))  # CDLL releases the GIL while constructing.
    lib.dspp_construct.argtypes = [ct.c_char_p, ct.POINTER(ct.c_int),
                                  ct.POINTER(ct.c_int), ct.c_double]
    lib.dspp_construct.restype = ct.c_int
    return lib


def construct(nodes, messages, succ, relays, ids, costs, priority, slack, reference_end,
              owners, capacities, tokens, token_limits, *, supply, feedback, deadline):
    """Same graph and counter inputs as _plan_unsplit's Python construction."""
    keys = tuple(succ)
    index = {k: i for i, k in enumerate(keys)}
    outmsg = {k: [] for k in nodes}
    for m, (source, target) in messages.items():
        outmsg[source].append((m, target))
    rows = [f'{len(keys)} {len(capacities)} {len(owners)} {int(supply)} {int(feedback)}']
    rows.extend(f'{cap!r} {limit!r}' for cap, limit in zip(capacities, token_limits))
    def sequence(values):
        values = list(values)
        return f'{len(values)} ' + ' '.join(str(index[k]) for k in values)
    for k in keys:
        if k in nodes:
            node = nodes[k]
            kind = {'F': 0, 'B': 1, 'D': 1, 'W': 2, 'H': 3}[node.kind]
            rank = node.stage
            forward = index[(rank, 'F', node.chunk, node.microbatch, node.split)] if kind == 1 else -1
        else:
            kind = 5 if k in relays else 4
            rank = nodes[relays[k] if kind == 5 else messages[k][0]].stage
            forward = -1
        mapped = relays.get(k, k)
        outgoing = outmsg.get(k, ())
        rows.append(' '.join(map(str, (kind, rank, ids[k], costs.get(k, 0.),
            priority.get(k, 0.), slack.get(k, 0.), reference_end[mapped],
            index[relays[k]] if k in relays else -1, forward, tokens.get(k, 0.)))) + ' ' +
            sequence(sorted(succ[k], key=ids.__getitem__)) + ' ' + str(len(outgoing)) + ' ' +
            ' '.join(f'{index[m]} {index[t]}' for m, t in outgoing))
    for owner in owners:
        rows.append(f'{owner.bytes!r} ' + sequence(dict.fromkeys(owner.born)) + ' ' +
                    sequence(dict.fromkeys(owner.users)))
    output = (ct.c_int * len(nodes))()
    sizes = (ct.c_int * len(capacities))()
    code = library().dspp_construct('\n'.join(rows).encode(), output, sizes, deadline or 0.)
    if code < 0:
        raise RuntimeError('native DSPP planner rejected its graph payload')
    if code:
        return None, 'feedback-deadline' if code == 2 else 'construction-capacity-deadlock'
    orders, offset = [], 0
    for size in sizes:
        orders.append([nodes[keys[output[i]]] for i in range(offset, offset + size)])
        offset += size
    if offset != len(nodes):
        raise RuntimeError('native DSPP planner returned an incomplete task set')
    return orders, ''

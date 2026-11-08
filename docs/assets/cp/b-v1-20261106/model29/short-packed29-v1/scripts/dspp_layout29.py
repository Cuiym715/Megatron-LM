"""Pure helpers for DSPP V-pipeline Transformer-layer layouts."""


def logical_layer_ranges(num_layers, pipeline_size, layout):
    """Return zero-based half-open Transformer-layer ranges by logical stage."""
    num_layers = int(num_layers)
    pipeline_size = int(pipeline_size)
    logical_stages = 2 * pipeline_size
    if pipeline_size < 1 or num_layers < 1:
        raise ValueError('invalid DSPP layer layout dimensions')

    if layout == 'balanced':
        if num_layers % logical_stages:
            raise ValueError('balanced DSPP layout requires equal logical stages')
        counts = [num_layers // logical_stages] * logical_stages
    elif layout == 'legacy-output-slot':
        if (num_layers + 1) % logical_stages:
            raise ValueError('legacy DSPP layout requires num_layers + 1 slots')
        width = (num_layers + 1) // logical_stages
        counts = [width] * logical_stages
        counts[-1] -= 1
    elif layout == 'endpoint-3-2':
        if (pipeline_size, num_layers) not in ((8, 61), (4, 29)):
            raise ValueError('endpoint-3-2 requires PP8/61 or PP4/29 Transformer layers')
        counts = [3] + [4] * (logical_stages - 2) + [2]
    else:
        raise ValueError(f'unknown DSPP V-layer layout {layout!r}')

    ranges = []
    begin = 0
    for count in counts:
        if count <= 0:
            raise ValueError('every DSPP logical stage must retain a Transformer layer')
        ranges.append((begin, begin + count))
        begin += count
    if begin != num_layers:
        raise ValueError('DSPP layer layout does not cover the model exactly')
    return tuple(ranges)


def logical_stage_for_physical(pipeline_size, stage, chunk):
    if chunk not in (0, 1) or not 0 <= stage < pipeline_size:
        raise ValueError('invalid DSPP physical stage/chunk')
    return stage if chunk == 0 else 2 * pipeline_size - stage - 1


def physical_layer_range(num_layers, pipeline_size, layout, stage, chunk):
    logical = logical_stage_for_physical(pipeline_size, stage, chunk)
    return logical_layer_ranges(num_layers, pipeline_size, layout)[logical]


def physical_layer_count(num_layers, pipeline_size, layout, stage):
    ranges = logical_layer_ranges(num_layers, pipeline_size, layout)
    first = ranges[stage]
    second = ranges[2 * pipeline_size - stage - 1]
    return first[1] - first[0] + second[1] - second[0]

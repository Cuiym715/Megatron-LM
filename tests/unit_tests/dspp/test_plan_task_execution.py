"""CPU numerical check of actual _DsppVTaskBatch B/W binding.

Tiny dense stage only; this does not replace FlashAttention/NCCL/GPU checks.
"""
from collections import deque
from dataclasses import replace
import random

import torch

from megatron.core.datasets.dspp_batch_plan import build_dspp_batch_plan
from megatron.core.datasets.dspp_training import DsppTrainingBatch
from megatron.core.pipeline_parallel.dspp_plan import (
    PlanBinding, node_key, receive_orders, template_plan,
)
from megatron.core.pipeline_parallel.slice_v import build_slice_v_schedule
from megatron.core.pipeline_parallel.schedules import _DsppVTaskBatch
from megatron.core.zbpp_utils import WeightGradStore

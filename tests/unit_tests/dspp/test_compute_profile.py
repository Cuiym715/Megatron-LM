from megatron.core.datasets.dspp_compute_profile import (
    DsppCapacityProbeResult,
    DsppComputeProfileRecord,
    DsppComputeShape,
    build_compute_shape_plan,
    discover_capacity_range,
    generate_candidate_menu,
    suggest_refinement_capacities,
)


def _record(capacity, category, duration, *, operation="F"):
    if category == "full":
        q, h = (capacity,), (0,)
    elif category == "tail":
        q, h = (capacity // 2,), (0,)
    elif category == "middle":
        q, h = (capacity,), (capacity,)
    else:
        q, h = (capacity // 2, capacity // 2), (0, 0)
    shape = DsppComputeShape(0, operation, capacity, q, h, work_units=float(capacity))
    return DsppComputeProfileRecord(shape, duration, duration, 3)


def test_shape_plan_contains_full_tail_middle_and_pack():
    shapes = build_compute_shape_plan(
        min_capacity=256,
        max_capacity=1024,
        alignment=256,
        max_shapes=1000,
    )
    assert {shape.category for shape in shapes} == {"full", "tail", "middle", "pack"}
    assert {shape.capacity for shape in shapes} == {256, 512, 1024}


def test_menu_rejects_capacity_with_bad_tail_even_if_full_is_fast():
    records = []
    for capacity in (1000, 1234, 1500):
        for category in ("full", "tail", "middle", "pack"):
            duration = capacity / 1000.0
            if capacity == 1234 and category == "full":
                duration *= 2.0
            if capacity == 1000 and category == "tail":
                duration *= 8.0
            records.append(_record(capacity, category, duration))
    menu = generate_candidate_menu(records, efficiency_threshold=0.8, min_coverage=1.0)
    assert 1234 not in menu.candidates
    assert 1000 not in menu.candidates
    assert menu.candidates == (1500,)


def test_profile_menu_keeps_robust_capacity():
    records = []
    for capacity in (512, 1024, 2048):
        for category in ("full", "tail", "middle", "pack"):
            throughput = 0.5 if capacity == 512 else (0.95 if capacity == 1024 else 1.0)
            duration = capacity / throughput
            records.append(_record(capacity, category, duration))
    menu = generate_candidate_menu(records, efficiency_threshold=0.90, min_coverage=1.0)
    assert menu.candidates
    assert 1024 in menu.candidates or 2048 in menu.candidates


def test_menu_requires_model_work_units_by_default():
    shape = DsppComputeShape(0, "F", 512, (512,), (0,))
    record = DsppComputeProfileRecord(shape, 1.0, 1.0, 3)
    try:
        generate_candidate_menu([record])
    except ValueError as error:
        assert "work_units" in str(error)
    else:
        raise AssertionError("missing work units must not silently rank unlike shapes")


def test_refinement_is_profile_driven_and_aligned():
    records = []
    for capacity, throughput in ((256, 0.4), (512, 0.7), (1024, 1.0), (2048, 1.0)):
        for category in ("full", "tail", "middle", "pack"):
            records.append(_record(capacity, category, capacity / throughput))
    refined = suggest_refinement_capacities(
        records,
        min_capacity=256,
        max_capacity=2048,
        alignment=128,
        max_new_capacities=4,
    )
    assert refined
    assert all(value % 128 == 0 for value in refined)
    assert not set(refined).intersection({256, 512, 1024, 2048})


def test_menu_preserves_non_dominated_shape_specialists():
    records = []
    capacities = (640, 1280)
    throughputs = {
        640: {"full": 1.0, "tail": 0.82, "middle": 0.95, "pack": 0.82},
        1280: {"full": 0.82, "tail": 1.0, "middle": 0.82, "pack": 0.95},
    }
    for capacity in capacities:
        for category, throughput in throughputs[capacity].items():
            records.append(_record(capacity, category, capacity / throughput))
    menu = generate_candidate_menu(records, efficiency_threshold=0.90, min_coverage=1.0)
    assert menu.candidates == capacities
    payload = menu.to_dict()
    assert [row["capacity"] for row in payload["profile_bank"]] == list(capacities)
    assert payload["pareto_capacities"] == list(capacities)


def test_capacity_discovery_finds_legal_min_and_aligned_memory_max():
    def probe(capacity):
        legal = capacity >= 5
        fits = capacity <= 1000
        return DsppCapacityProbeResult(
            capacity=capacity,
            feasible=legal and fits,
            peak_reserved_bytes=capacity,
            total_memory_bytes=1000,
            reason=None if legal and fits else "synthetic boundary",
        )

    discovery = discover_capacity_range(
        probe,
        legal_start_capacity=1,
        search_alignment=64,
        memory_utilization=0.90,
    )
    assert discovery.min_capacity == 5
    assert discovery.max_capacity == 896
    assert discovery.probes


def test_capacity_discovery_accepts_explicitly_narrow_feasible_interval():
    def probe(capacity):
        return DsppCapacityProbeResult(
            capacity=capacity,
            feasible=3 <= capacity <= 7,
        )

    discovery = discover_capacity_range(
        probe,
        legal_start_capacity=1,
        search_alignment=16,
    )
    assert (discovery.min_capacity, discovery.max_capacity) == (3, 3)

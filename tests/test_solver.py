import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models import Consumer, Generator
from solver_milp import solve as milp_solve
from solver_greedy import solve as greedy_solve


def _load(path: str):
    with open(path) as f:
        data = json.load(f)
    consumers = [Consumer(id=r["id"], hourly_demand=r["hourly_demand"]) for r in data["consumers"]]
    generators = [
        Generator(id=r["id"], type=r["type"], max_power=r["max_power"],
                  cost_per_kwh=r["cost_per_kwh"], generation_profile=r["generation_profile"])
        for r in data["generators"]
    ]
    return consumers, generators


CASES_DIR = os.path.join(os.path.dirname(__file__), "..", "testcases")


@pytest.mark.parametrize("case_file", ["case_a_surplus.json", "case_b_deficit.json"])
def test_milp_greedy_agree(case_file):
    consumers, generators = _load(os.path.join(CASES_DIR, case_file))
    milp_result = milp_solve(consumers, generators)
    greedy_result = greedy_solve(consumers, generators)
    assert milp_result.total_consumer_hours_served == greedy_result.total_consumer_hours_served, (
        f"Served hours mismatch: MILP={milp_result.total_consumer_hours_served} "
        f"Greedy={greedy_result.total_consumer_hours_served}"
    )
    assert abs(milp_result.total_cost - greedy_result.total_cost) < 0.01, (
        f"Cost mismatch: MILP={milp_result.total_cost:.4f} Greedy={greedy_result.total_cost:.4f}"
    )


def test_zero_demand():
    consumers = [Consumer(id="C1", hourly_demand=[0.0] * 24),
                 Consumer(id="C2", hourly_demand=[0.0] * 24)]
    generators = [Generator(id="G1", type="diesel", max_power=100, cost_per_kwh=0.3,
                            generation_profile=[0.0] * 24)]
    for solve in [milp_solve, greedy_solve]:
        result = solve(consumers, generators)
        assert result.total_cost == 0.0
        for hr in result.hours:
            assert set(hr.served_consumers) == {"C1", "C2"}


def test_zero_generation():
    consumers = [Consumer(id="C1", hourly_demand=[10.0] * 24),
                 Consumer(id="C2", hourly_demand=[5.0] * 24)]
    generators = [Generator(id="G1", type="diesel", max_power=0, cost_per_kwh=0.3,
                            generation_profile=[0.0] * 24)]
    for solve in [milp_solve, greedy_solve]:
        result = solve(consumers, generators)
        for hr in result.hours:
            assert set(hr.served_consumers) == set()
            assert set(hr.unserved_consumers) == {"C1", "C2"}


def test_single_giant_consumer():
    # Giant consumer demanding more than capacity; smaller consumers should be served
    consumers = [
        Consumer(id="Giant", hourly_demand=[200.0] * 24),
        Consumer(id="Small1", hourly_demand=[10.0] * 24),
        Consumer(id="Small2", hourly_demand=[10.0] * 24),
    ]
    generators = [Generator(id="G1", type="diesel", max_power=30, cost_per_kwh=0.2,
                            generation_profile=[0.0] * 24)]
    for solve in [milp_solve, greedy_solve]:
        result = solve(consumers, generators)
        for hr in result.hours:
            assert "Giant" not in hr.served_consumers
            assert "Small1" in hr.served_consumers
            assert "Small2" in hr.served_consumers


def test_knapsack_correctness():
    # Capacity = 100, demands = [50, 40, 30, 70]. Optimal: serve {30, 40} (count 2).
    # Naive "drop only largest (70)" leaves 50+40+30 = 120 > 100, so it must shed more.
    consumers = [
        Consumer(id="C50", hourly_demand=[50.0] + [0.0] * 23),
        Consumer(id="C40", hourly_demand=[40.0] + [0.0] * 23),
        Consumer(id="C30", hourly_demand=[30.0] + [0.0] * 23),
        Consumer(id="C70", hourly_demand=[70.0] + [0.0] * 23),
    ]
    generators = [Generator(id="G1", type="diesel", max_power=100, cost_per_kwh=0.1,
                            generation_profile=[0.0] * 24)]
    for solve in [milp_solve, greedy_solve]:
        result = solve(consumers, generators)
        hr0 = result.hours[0]
        assert len(hr0.served_consumers) == 2, (
            f"{solve.__module__}: expected 2 served, got {hr0.served_consumers}"
        )
        assert set(hr0.served_consumers) == {"C30", "C40"}, (
            f"{solve.__module__}: expected {{C30, C40}}, got {hr0.served_consumers}"
        )

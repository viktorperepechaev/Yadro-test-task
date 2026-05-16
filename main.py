import argparse
import json
import os

from models import Consumer, Generator, ScheduleResult
from visualize import plot_schedule


def load_input(path: str) -> tuple[list[Consumer], list[Generator]]:
    with open(path) as f:
        data = json.load(f)

    consumers = []
    for raw in data["consumers"]:
        if len(raw["hourly_demand"]) != 24:
            raise ValueError(f"{raw['id']}: hourly_demand must have 24 values")
        consumers.append(Consumer(id=raw["id"], hourly_demand=raw["hourly_demand"]))

    generators = []
    for raw in data["generators"]:
        if len(raw["generation_profile"]) != 24:
            raise ValueError(f"{raw['id']}: generation_profile must have 24 values")
        if raw["type"] not in ("diesel", "solar"):
            raise ValueError(f"{raw['id']}: type must be 'diesel' or 'solar'")
        generators.append(Generator(
            id=raw["id"],
            type=raw["type"],
            max_power=raw["max_power"],
            cost_per_kwh=raw["cost_per_kwh"],
            generation_profile=raw["generation_profile"],
        ))

    return consumers, generators


def print_summary(result: ScheduleResult, consumers: list[Consumer]) -> None:
    n_consumers = len(consumers)
    header = f"{'Hour':>4}  {'Gen (kWh)':>10}  {'Cost ($)':>10}  {'Served':>8}  {'Shed':>6}"
    print(header)
    print("-" * len(header))
    for hr in result.hours:
        total_gen = sum(hr.generator_output.values())
        n_served = len(hr.served_consumers)
        n_shed = len(hr.unserved_consumers)
        print(f"{hr.hour:>4}  {total_gen:>10.2f}  {hr.cost:>10.4f}  {n_served:>4}/{n_consumers:<3}  {n_shed:>6}")
    print("-" * len(header))
    print(f"Total cost: ${result.total_cost:.4f}  |  Consumer-hours served: {result.total_consumer_hours_served}")

    shed_hours = [(hr.hour, hr.unserved_consumers) for hr in result.hours if hr.unserved_consumers]
    if shed_hours:
        print("\nLoad shedding occurred:")
        for hour, unserved in shed_hours:
            print(f"  Hour {hour:>2}: shed {unserved}")
    else:
        print("\nNo load shedding — all consumers served every hour.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Power grid simulator")
    parser.add_argument("input", help="Path to JSON input file")
    parser.add_argument("--solver", choices=["milp", "greedy"], default="milp")
    args = parser.parse_args()

    consumers, generators = load_input(args.input)

    if args.solver == "milp":
        from solver_milp import solve
    else:
        from solver_greedy import solve

    result = solve(consumers, generators)
    print_summary(result, consumers)

    input_dir = os.path.dirname(args.input)
    base = os.path.splitext(os.path.basename(args.input))[0]
    png_path = os.path.join(input_dir, f"output_{base}.png")
    plot_schedule(result, consumers, generators, base, png_path)
    print(f"\nVisualization saved to: {png_path}")


if __name__ == "__main__":
    main()

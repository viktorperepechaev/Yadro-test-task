from models import Consumer, Generator, HourResult, ScheduleResult


def solve(consumers: list[Consumer], generators: list[Generator]) -> ScheduleResult:
    diesels = [g for g in generators if g.type == "diesel"]
    solars = [g for g in generators if g.type == "solar"]
    sorted_diesels = sorted(diesels, key=lambda g: g.cost_per_kwh)
    diesel_capacity = sum(g.max_power for g in diesels)

    hour_results: list[HourResult] = []
    total_cost = 0.0
    total_served = 0

    for t in range(24):
        solar_available = sum(s.generation_profile[t] for s in solars)
        total_capacity = solar_available + diesel_capacity

        served_indices: set[int] = set()
        used = 0.0
        for idx, c in sorted(enumerate(consumers), key=lambda x: x[1].hourly_demand[t]):
            d = c.hourly_demand[t]
            if used + d <= total_capacity + 1e-6:
                served_indices.add(idx)
                used += d

        served = [c.id for i, c in enumerate(consumers) if i in served_indices]
        unserved = [c.id for i, c in enumerate(consumers) if i not in served_indices]
        total_demand = used  # accumulated in the greedy loop above

        solar_used = min(solar_available, total_demand)
        residual = total_demand - solar_used

        gen_output: dict[str, float] = {}
        if solar_available > 1e-9:
            for s in solars:
                gen_output[s.id] = s.generation_profile[t] / solar_available * solar_used
        else:
            for s in solars:
                gen_output[s.id] = 0.0

        cost = 0.0
        for g in sorted_diesels:
            if residual <= 1e-9:
                gen_output[g.id] = 0.0
            else:
                output = min(g.max_power, residual)
                gen_output[g.id] = output
                cost += output * g.cost_per_kwh
                residual -= output

        hour_results.append(HourResult(
            hour=t,
            generator_output=gen_output,
            served_consumers=served,
            unserved_consumers=unserved,
            cost=cost,
        ))
        total_cost += cost
        total_served += len(served)

    return ScheduleResult(
        hours=hour_results,
        total_cost=round(total_cost, 6),
        total_consumer_hours_served=total_served,
    )

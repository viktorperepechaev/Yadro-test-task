import pulp
from models import Consumer, Generator, HourResult, ScheduleResult


def _val(var) -> float:
    return max(0.0, pulp.value(var) or 0.0)


def _make_vars(prob: pulp.LpProblem, consumers: list[Consumer],
               diesels: list[Generator], solars: list[Generator]):
    v = {c.id: prob.add_variable(f"v_{c.id}", cat="Binary") for c in consumers}
    p = {g.id: prob.add_variable(f"p_{g.id}", lowBound=0) for g in diesels}
    u = {g.id: prob.add_variable(f"u_{g.id}", cat="Binary") for g in diesels}
    s = {g.id: prob.add_variable(f"s_{g.id}", lowBound=0) for g in solars}
    return v, p, u, s


def _add_constraints(prob: pulp.LpProblem, t: int,
                     consumers: list[Consumer], diesels: list[Generator],
                     solars: list[Generator], v, p, u, s) -> None:
    prob += (
        pulp.lpSum(p[g.id] for g in diesels) + pulp.lpSum(s[g.id] for g in solars)
        == pulp.lpSum(c.hourly_demand[t] * v[c.id] for c in consumers)
    )
    for g in diesels:
        prob += p[g.id] <= g.max_power * u[g.id]
    for g in solars:
        prob += s[g.id] <= g.generation_profile[t]


def _solve_hour(t: int, consumers: list[Consumer], diesels: list[Generator],
                solars: list[Generator]) -> tuple[dict[str, float], list[str], list[str], float]:
    # Stage 1: maximize served consumers
    prob1 = pulp.LpProblem(f"stage1_h{t}", pulp.LpMaximize)
    v1, p1, u1, s1 = _make_vars(prob1, consumers, diesels, solars)
    prob1 += pulp.lpSum(v1[c.id] for c in consumers)
    _add_constraints(prob1, t, consumers, diesels, solars, v1, p1, u1, s1)
    prob1.solve(pulp.PULP_CBC_CMD(msg=0))
    v_star = round(pulp.value(prob1.objective))

    # Stage 2: minimize cost subject to v_star
    prob2 = pulp.LpProblem(f"stage2_h{t}", pulp.LpMinimize)
    v2, p2, u2, s2 = _make_vars(prob2, consumers, diesels, solars)
    prob2 += pulp.lpSum(g.cost_per_kwh * p2[g.id] for g in diesels)
    _add_constraints(prob2, t, consumers, diesels, solars, v2, p2, u2, s2)
    prob2 += pulp.lpSum(v2[c.id] for c in consumers) >= v_star
    prob2.solve(pulp.PULP_CBC_CMD(msg=0))

    gen_output: dict[str, float] = {g.id: _val(p2[g.id]) for g in diesels}
    gen_output.update({g.id: _val(s2[g.id]) for g in solars})
    served = [c.id for c in consumers if (pulp.value(v2[c.id]) or 0) > 0.5]
    unserved = [c.id for c in consumers if (pulp.value(v2[c.id]) or 0) <= 0.5]
    cost = pulp.value(prob2.objective) or 0.0

    return gen_output, served, unserved, cost


def solve(consumers: list[Consumer], generators: list[Generator]) -> ScheduleResult:
    diesels = [g for g in generators if g.type == "diesel"]
    solars = [g for g in generators if g.type == "solar"]

    hour_results: list[HourResult] = []
    total_cost = 0.0
    total_served = 0

    for t in range(24):
        gen_output, served, unserved, cost = _solve_hour(t, consumers, diesels, solars)
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

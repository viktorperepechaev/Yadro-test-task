import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from models import Consumer, Generator, ScheduleResult


def plot_schedule(
    result: ScheduleResult,
    consumers: list[Consumer],
    generators: list[Generator],
    case_name: str,
    output_path: str,
) -> None:
    hours = list(range(24))
    diesels = sorted([g for g in generators if g.type == "diesel"], key=lambda g: g.cost_per_kwh)
    solars = [g for g in generators if g.type == "solar"]
    ordered_gens = solars + diesels

    solar_colors = ["#FFD700", "#FFA500", "#FF8C00", "#FF6347"]
    diesel_colors = ["#6699CC", "#4477AA", "#335588", "#224466"]

    # Pre-compute served sets once to avoid O(C) list scans per lookup
    served_sets = [set(hr.served_consumers) for hr in result.hours]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # --- Top subplot: generation stack ---
    bottom = np.zeros(24)
    n_solars = len(solars)
    for i, g in enumerate(ordered_gens):
        output = np.array([result.hours[t].generator_output.get(g.id, 0.0) for t in hours])
        if g.type == "solar":
            color = solar_colors[i % len(solar_colors)]
        else:
            color = diesel_colors[(i - n_solars) % len(diesel_colors)]
        label = f"{g.id} ({'solar' if g.type == 'solar' else f'diesel ${g.cost_per_kwh}/kWh'})"
        ax1.fill_between(hours, bottom, bottom + output, alpha=0.8, color=color, label=label, step="mid")
        bottom += output

    served_demand = np.array([
        sum(c.hourly_demand[t] for c in consumers if c.id in served_sets[t])
        for t in hours
    ])
    total_demand = np.array([sum(c.hourly_demand[t] for c in consumers) for t in hours])

    ax1.step(hours, served_demand, where="mid", color="black", linestyle="--", linewidth=2,
             label="Served demand")
    ax1.step(hours, total_demand, where="mid", color="gray", linestyle=":", linewidth=2,
             label="Total requested demand")
    ax1.set_xlim(-0.5, 23.5)
    ax1.set_xticks(hours)
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("kWh")
    ax1.set_title(f"{case_name} — Total cost: ${result.total_cost:.2f}")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

    # --- Bottom subplot: consumer heatmap ---
    peaks = {c.id: max(c.hourly_demand) for c in consumers}
    consumer_order = sorted(consumers, key=lambda c: peaks[c.id], reverse=True)

    demand_grid = np.zeros((len(consumer_order), 24))
    served_mask = np.zeros((len(consumer_order), 24), dtype=bool)
    for row_i, c in enumerate(consumer_order):
        demand_grid[row_i] = c.hourly_demand
        for t in hours:
            served_mask[row_i, t] = c.id in served_sets[t]

    global_max = demand_grid.max() or 1.0
    intensity = demand_grid / global_max

    rgba = np.zeros((*demand_grid.shape, 4))
    rgba[..., 3] = 1.0
    rgba[served_mask, 1] = 0.3 + 0.7 * intensity[served_mask]
    rgba[served_mask, 0] = 0.1
    rgba[served_mask, 2] = 0.1
    rgba[~served_mask, 0] = 0.3 + 0.7 * intensity[~served_mask]
    rgba[~served_mask, 1] = 0.1
    rgba[~served_mask, 2] = 0.1

    ax2.imshow(rgba, aspect="auto", interpolation="nearest")
    ax2.set_xticks(hours)
    ax2.set_xticklabels([str(h) for h in hours], fontsize=8)
    ax2.set_yticks(range(len(consumer_order)))
    ax2.set_yticklabels(
        [f"{c.id} (peak {peaks[c.id]:.0f} kWh/h)" for c in consumer_order],
        fontsize=9,
    )
    ax2.set_xlabel("Hour")
    ax2.set_title("Consumer service schedule  (brighter = higher demand, green = served, red = shed)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)

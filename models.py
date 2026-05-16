from dataclasses import dataclass


@dataclass
class Consumer:
    id: str
    hourly_demand: list[float]  # length 24


@dataclass
class Generator:
    id: str
    type: str  # "diesel" or "solar"
    max_power: float           # for diesel; ignored for solar
    cost_per_kwh: float        # for diesel; 0 for solar
    generation_profile: list[float]  # length 24; for solar; zeros for diesel


@dataclass
class HourResult:
    hour: int
    generator_output: dict[str, float]   # gen_id -> kWh
    served_consumers: list[str]
    unserved_consumers: list[str]
    cost: float


@dataclass
class ScheduleResult:
    hours: list[HourResult]
    total_cost: float
    total_consumer_hours_served: int

"""Privileged geometry is used here for offline supervision, never in the head."""
from dataclasses import asdict, dataclass
import math
import numpy as np


@dataclass(frozen=True)
class ReferenceConfig:
    distance_scale: float = 15.0
    lateral_scale: float = 3.0
    time_scale: float = 3.0
    distance_weight: float = 0.5
    speed_epsilon: float = 0.1

    def __post_init__(self):
        values = asdict(self)
        if not all(math.isfinite(float(x)) for x in values.values()):
            raise ValueError("Reference parameters must be finite")
        if min(self.distance_scale, self.lateral_scale, self.time_scale, self.speed_epsilon) <= 0:
            raise ValueError("Reference scales must be positive")
        if not 0 <= self.distance_weight <= 1:
            raise ValueError("distance_weight must lie in [0, 1]")


def continuous_reference(ego_xy, heading_xy, speed, pedestrian_xy, config=None):
    """Return the maximum soft proximity reference; arrival time is NOT true TTC."""
    config = config or ReferenceConfig()
    ego = np.asarray(ego_xy, dtype=np.float64)
    heading = np.asarray(heading_xy, dtype=np.float64)
    people = np.asarray(pedestrian_xy, dtype=np.float64)
    if ego.shape != (2,) or heading.shape != (2,):
        raise ValueError("ego_xy and heading_xy must have shape (2,)")
    if people.size == 0:
        people = np.empty((0, 2), dtype=np.float64)
    if people.ndim != 2 or people.shape[1] != 2:
        raise ValueError("pedestrian_xy must have shape (N, 2)")
    if not all(np.isfinite(x).all() for x in (ego, heading, people)) or not math.isfinite(speed) or speed < 0:
        raise ValueError("Invalid simulator state; discard sample rather than label it safe")
    norm = np.linalg.norm(heading)
    if norm < 1e-8:
        raise ValueError("Heading has zero length")
    if len(people) == 0:
        return 0.0
    relative = people - ego
    longitudinal = relative.dot(heading / norm)
    lateral = np.sqrt(np.maximum(0.0, np.sum(relative ** 2, axis=1) - longitudinal ** 2))
    time = np.where(longitudinal > 0, longitudinal / max(speed, config.speed_epsilon), np.inf)
    distance_term = np.clip(1.0 - longitudinal / config.distance_scale, 0, 1)
    time_term = np.clip(1.0 - time / config.time_scale, 0, 1)
    scores = (longitudinal > 0) * np.exp(-lateral ** 2 / (2 * config.lateral_scale ** 2))
    scores *= config.distance_weight * distance_term + (1 - config.distance_weight) * time_term
    return float(np.max(scores))


def reference_from_env(base_env, config=None):
    """Explicit privileged adapter for collection/reference baselines only."""
    vehicle = base_env.vehicle
    p = vehicle.get_location()
    h = vehicle.get_transform().get_forward_vector()
    v = vehicle.get_velocity()
    actors = {}
    for attr in ("walker_probes", "walker_probe", "walkers", "walker_actors", "walker"):
        value = getattr(base_env, attr, None)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for actor in candidates:
            if getattr(actor, "is_alive", True):
                actors[getattr(actor, "id", id(actor))] = actor
    for actor in getattr(base_env, "actor_list", []) or []:
        if "walker" in getattr(actor, "type_id", "") and getattr(actor, "is_alive", True):
            actors[getattr(actor, "id", id(actor))] = actor
    positions = []
    for actor in actors.values():
        pos = actor.get_location()  # Failure invalidates this sample; do not silently drop a walker.
        positions.append([pos.x, pos.y])
    return continuous_reference([p.x, p.y], [h.x, h.y], math.hypot(v.x, v.y), positions, config)

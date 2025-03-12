from collections.abc import Callable
from typing import Any, NamedTuple

from flax import struct
from jaxtyping import Array, PRNGKeyArray

Observation = Array
Action = Array
Extra = dict[str, Array]
Policy = Callable[[Observation, PRNGKeyArray], tuple[Action, Extra]]
Metrics = dict[str, Array]

Params = Any
PreprocessObservationFn = Callable[[Observation, Params], Observation]

NestedArray = Array


@struct.dataclass
class SimState:
    pos: Array
    vel: Array


@struct.dataclass
class State:
    sim_state: SimState
    obs: Observation
    reward: Array
    done: Array
    metrics: Metrics = struct.field(default_factory=Metrics)


class Transition(NamedTuple):
    observation: NestedArray
    action: NestedArray
    reward: NestedArray
    discount: NestedArray
    next_observation: NestedArray
    extras: dict[str, NestedArray] = {}


class Trajectory(NamedTuple):
    position: Array
    reward: Array
    discount: Array
    metrics: Metrics

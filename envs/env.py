import abc

import jax
from flax import struct
from jaxtyping import Array
from training import types


@struct.dataclass
class Env(abc.ABC):
    @abc.abstractmethod
    def reset(self, rng: jax.Array) -> types.State:
        """Resets the environment to an initial state."""

    @abc.abstractmethod
    def step(self, state: types.State, action: Array) -> types.State:
        """Run one timestep of the environment's dynamics."""

    @abc.abstractmethod
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""

    @abc.abstractmethod
    def action_size(self) -> int:
        """The size of the action vector returned in step and reset."""

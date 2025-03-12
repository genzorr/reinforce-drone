import jax
import jax.numpy as jnp
from envs.env import Env
from jaxtyping import Array
from training import types


class Drone(Env):
    def __init__(
        self,
        dt: float = 0.05,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (-10.0, -10.0, -10.0),
            (10.0, 10.0, 10.0),
        ),
        drone_radius: float = 0.2,
        start_position_bounds: tuple[tuple[float, float, float], tuple[float, float, float]] = (
            (-8.0, -8.0, -8.0),
            (-6.0, -6.0, -6.0),
        ),
        target_position: tuple[float, float, float] = (8.0, 8.0, 8.0),
        obstacles: list[tuple[float, float, float, float]] = [
            (0.0, 0.0, 0.0, 3.0),
            (-3.0, 4.0, 2.0, 1.5),
        ],
        max_acceleration: float = 12.0,
        target_threshold: float = 0.5,
        coeff_target_reward: float = 1000.0,
        coeff_distance_reward: float = 10.0,
        coeff_time_reward: float = -10.0,
        coeff_boundary_reward: float = -100.0,
        coeff_collision_reward: float = -100.0,
    ) -> None:
        self.dt = dt
        self.bounds = jnp.array(bounds)
        self.drone_radius = drone_radius
        self.start_position_bounds = jnp.array(start_position_bounds)
        self.target_position = jnp.array(target_position)
        self.obstacles = jnp.array(obstacles)
        self.max_acceleration = max_acceleration
        self.target_threshold = target_threshold
        self.coeff_target_reward = coeff_target_reward
        self.coeff_distance_reward = coeff_distance_reward
        self.coeff_time_reward = coeff_time_reward
        self.coeff_boundary_reward = coeff_boundary_reward
        self.coeff_collision_reward = coeff_collision_reward

        self.diagonal_length = jnp.linalg.norm(self.bounds[1] - self.bounds[0])
        self._gravity = jnp.array([0.0, 0.0, -9.81])

    def action_size(self) -> int:
        return 3

    def observation_size(self) -> int:
        return self._get_obs(
            types.SimState(
                pos=jnp.zeros(3),
                vel=jnp.zeros(3),
            )
        ).shape[-1]

    def get_distance_to_obstacles(self, pos: Array, obstacles: Array) -> Array:
        return jnp.linalg.norm(pos - obstacles[:, :3], axis=-1) - (self.drone_radius + obstacles[:, 3])

    def reset(self, rng: jax.Array) -> types.State:
        start_position = jax.random.uniform(
            rng, (3,), minval=self.start_position_bounds[0], maxval=self.start_position_bounds[1]
        )
        vel = jnp.zeros(3)
        sim_state = types.SimState(
            pos=start_position,
            vel=vel,
        )

        obs = self._get_obs(sim_state)
        reward, done, zero = jnp.array(0.0), jnp.array(0.0), jnp.array(0.0)

        metrics = {
            "target_reached": zero,
            "collision": zero,
            "out_of_bounds": zero,
            "distance": zero,
            "target_reward": zero,
            "collision_reward": zero,
            "oob_reward": zero,
            "distance_reward": zero,
            "time_reward": zero,
        }
        return types.State(sim_state, obs, reward, done, metrics)

    def step(self, state: types.State, action: Array) -> types.State:
        # Scale action from [-1, 1] to [-max_acceleration, max_acceleration].
        acc = action * self.max_acceleration
        acc = acc + self._gravity

        # Update drone parameters
        new_vel = state.sim_state.vel + acc * self.dt
        new_pos = state.sim_state.pos + new_vel * self.dt + 0.5 * acc * self.dt**2

        # Compute distance to the target
        distance = jnp.linalg.norm(new_pos - self.target_position)
        prev_distance = jnp.linalg.norm(state.sim_state.pos - self.target_position)

        # Termination conditions
        def check_collision(obstacles: Array):
            dist_to_obstacles = self.get_distance_to_obstacles(new_pos, obstacles)
            return (dist_to_obstacles < 0).any()

        collision = jnp.where(check_collision(self.obstacles), 1.0, 0.0)
        target_reached = jnp.where(distance < self.target_threshold, 1.0, 0.0)
        out_of_bounds = jnp.where(jnp.any(new_pos < self.bounds[0]) | jnp.any(new_pos > self.bounds[1]), 1.0, 0.0)

        done = jnp.where(
            (target_reached > 0.0) | (collision > 0.0) | (out_of_bounds > 0.0),
            1.0,
            0.0,
        )

        # Termination rewards
        target_reward = jax.lax.select(distance < self.target_threshold, self.coeff_target_reward, 0.0)
        collision_reward = jnp.where(collision > 0.0, self.coeff_collision_reward, 0.0)
        oob_reward = jnp.where(out_of_bounds > 0.0, self.coeff_boundary_reward, 0.0)

        # Distance reward
        distance_reward = self.coeff_distance_reward * (prev_distance - distance)

        # Time reward
        time_reward = self.coeff_time_reward * jnp.array(self.dt)

        # Combine rewards
        reward = target_reward + collision_reward + oob_reward + distance_reward + time_reward

        new_sim_state = types.SimState(
            pos=new_pos,
            vel=new_vel,
        )
        obs = self._get_obs(new_sim_state)

        return types.State(
            sim_state=new_sim_state,
            obs=obs,
            reward=reward,
            done=done,
            metrics={
                "target_reached": target_reached,
                "collision": collision,
                "out_of_bounds": out_of_bounds,
                "distance": distance,
                "target_reward": target_reward,
                "collision_reward": collision_reward,
                "oob_reward": oob_reward,
                "distance_reward": distance_reward,
                "time_reward": time_reward,
            },
        )

    def _get_obs(self, sim_state: types.SimState) -> Array:
        pos = sim_state.pos
        vel = sim_state.vel
        target_distance = self.target_position - pos
        obstacle_distance = self.get_distance_to_obstacles(pos, self.obstacles)

        return jnp.concatenate(
            [
                vel,
                target_distance,
                obstacle_distance,
            ]
        )

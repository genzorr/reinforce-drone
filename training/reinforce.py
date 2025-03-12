from collections.abc import Callable, Sequence
from functools import partial

import jax
import jax.random as jr
import optax
from envs.env import Env
from flax import linen as nn
from flax import struct
from jax import numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from training import distribution, networks, types, utils


@struct.dataclass
class REINFORCENetwork:
    parametric_action_distribution: distribution.ParametricDistribution
    policy_network: networks.FeedForwardNetwork


@struct.dataclass
class TrainingState:
    optimizer_state: optax.OptState
    policy_params: types.Params
    baseline: Array


def make_policy_network(
    param_size: int,
    obs_size: int,
    hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = nn.relu,
    kernel_init: networks.Initializer = jax.nn.initializers.glorot_normal(),
) -> networks.FeedForwardNetwork:
    policy_module = networks.MLP(
        layer_sizes=[*list(hidden_layer_sizes), param_size],
        activation=activation,
        kernel_init=kernel_init,
    )

    def apply(policy_params: types.Params, obs: types.Observation) -> types.Action:
        return policy_module.apply(policy_params, obs)

    dummy_obs = jnp.zeros((obs_size,))
    return networks.FeedForwardNetwork(
        init=lambda key: policy_module.init(key, dummy_obs),
        apply=apply,
    )


def make_reinforce_network(
    observation_size: int,
    action_size: int,
    policy_hidden_layer_sizes: Sequence[int] = (32,) * 4,
    activation: networks.ActivationFn = nn.relu,
) -> REINFORCENetwork:
    parametric_action_distribution = distribution.NormalTanhDistribution(event_size=action_size)
    policy_network = make_policy_network(
        parametric_action_distribution.param_size,
        observation_size,
        hidden_layer_sizes=policy_hidden_layer_sizes,
        activation=activation,
    )

    return REINFORCENetwork(
        parametric_action_distribution=parametric_action_distribution,
        policy_network=policy_network,
    )


def make_policy_fn(
    reinforce_network: REINFORCENetwork,
) -> Callable[[types.Params, bool], types.Policy]:
    def make_policy(params: types.Params, deterministic: bool = False) -> types.Policy:
        policy_network = reinforce_network.policy_network
        parametric_action_distribution = reinforce_network.parametric_action_distribution

        def policy(observation: types.Observation, key: PRNGKeyArray) -> tuple[types.Action, types.Extra]:
            logits = policy_network.apply(params, observation)
            if deterministic:
                return parametric_action_distribution.mode(logits), {}

            raw_actions = parametric_action_distribution.sample_no_postprocessing(logits, key)
            postprocessed_actions = parametric_action_distribution.postprocess(raw_actions)
            return postprocessed_actions, {
                "raw_action": raw_actions,
            }

        return policy

    return make_policy


def generate_unroll(
    key: PRNGKeyArray,
    env: Env,
    policy: types.Policy,
    unroll_length: int,
) -> tuple[types.State, types.Transition]:
    def unroll_fn(
        carry: tuple[PRNGKeyArray, types.State], unused_t
    ) -> tuple[tuple[PRNGKeyArray, types.State], types.Transition]:
        key, state = carry
        policy_key, key = jax.random.split(key)
        action, policy_extras = policy(state.obs, policy_key)
        new_state = env.step(state, action)
        return (key, new_state), types.Transition(
            observation=state.obs,
            action=action,
            reward=new_state.reward,
            discount=1 - new_state.done,
            next_observation=new_state.obs,
            extras={"policy_extras": policy_extras, "metrics": new_state.metrics},
        )

    reset_key, key = jr.split(key)
    env_state = env.reset(reset_key)
    (_, final_state), data = jax.lax.scan(unroll_fn, (key, env_state), (), length=unroll_length)
    return final_state, data


def compute_returns(
    rewards: Array,
    discounts: Array,
    gamma: float,
) -> tuple[Array, Array]:
    terminal_states = discounts < 1.0

    # Create a mask for post-terminal states
    def identify_post_terminal(carry: Array, is_terminal: Array) -> tuple[Array, Array]:
        has_seen_terminal = carry
        new_has_seen_terminal = jnp.where(jnp.logical_or(has_seen_terminal, is_terminal), 1.0, 0.0)
        is_post_terminal = has_seen_terminal
        return new_has_seen_terminal, is_post_terminal

    _, post_terminal_mask = jax.lax.scan(identify_post_terminal, jnp.array(0.0), terminal_states)

    # Create modified discounts
    modified_discounts = jnp.where(post_terminal_mask, 0.0, 1.0)
    episode_reward = jnp.sum(rewards * modified_discounts)

    def get_returns(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        return_value = carry
        reward, discount = x
        return_value = discount * (gamma * return_value + reward)
        return return_value, return_value

    _, returns = jax.lax.scan(get_returns, jnp.array(0.0), (rewards[::-1], modified_discounts[::-1]))
    return returns[::-1], episode_reward


def compute_loss(
    params: types.Params,
    baseline: Array,
    data: types.Transition,
    network: REINFORCENetwork,
    gamma: float,
) -> tuple[Array, types.Metrics]:
    # Compute returns for each unroll in the batch
    compute_returns_batch = jax.vmap(partial(compute_returns, gamma=gamma), in_axes=(0, 0))
    returns, episode_reward = compute_returns_batch(data.reward, data.discount)

    # Compute log probs from scratch
    logits = jax.vmap(
        network.policy_network.apply,
        in_axes=(None, 0),
    )(params, data.observation)
    log_probs = jax.vmap(network.parametric_action_distribution.log_prob, in_axes=(0, 0))(
        logits, data.extras["policy_extras"]["raw_action"]
    )

    # Compute loss
    advantage = returns - baseline
    advantage = (advantage - jnp.mean(advantage)) / (jnp.std(advantage) + 1e-8)
    total_loss = -jnp.mean(jnp.sum(log_probs * advantage, axis=1))

    return total_loss, {
        "total_loss": total_loss,
        "returns": returns,
        "episode_reward": episode_reward,
    }


def training_step(
    key: PRNGKeyArray,
    env: Env,
    training_state: TrainingState,
    network: REINFORCENetwork,
    make_policy: Callable[[types.Params, bool], types.Policy],
    optimizer: optax.GradientTransformation,
    unroll_length: int,
    batch_size: int,
    gamma: float,
) -> tuple[TrainingState, types.Metrics]:
    policy = make_policy(training_state.policy_params, False)

    # Collect batch of trajectories
    unroll_keys = jr.split(key, batch_size)
    _, batch_data = jax.vmap(
        partial(generate_unroll, env=env, policy=policy, unroll_length=unroll_length),
        in_axes=(0,),
    )(unroll_keys)

    assert batch_data.discount.shape[1:] == (unroll_length,)

    # Compute loss and gradients
    (_, train_metrics), grads = jax.value_and_grad(
        partial(
            compute_loss,
            baseline=training_state.baseline,
            data=batch_data,
            network=network,
            gamma=gamma,
        ),
        has_aux=True,
    )(training_state.policy_params)

    # Apply gradients
    updates, optimizer_state = optimizer.update(grads, training_state.optimizer_state)
    updated_params = optax.apply_updates(training_state.policy_params, updates)

    new_training_state = TrainingState(
        policy_params=updated_params,
        optimizer_state=optimizer_state,
        baseline=training_state.baseline,
    )
    return new_training_state, train_metrics


def train(
    env: Env,
    num_epochs: int = 100,
    unroll_length: int = 200,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    baseline_decay: float = 0.95,
    seed: int = 0,
) -> tuple[
    Callable[[types.Params, bool], types.Policy],
    TrainingState,
    types.Metrics,
]:
    key = jr.PRNGKey(seed)

    # Create network, initialize parameters
    reinforce_network = make_reinforce_network(
        observation_size=env.observation_size(),
        action_size=env.action_size(),
        policy_hidden_layer_sizes=(128,) * 4,
    )
    init_key, key = jr.split(key)
    policy_params = reinforce_network.policy_network.init(init_key)

    # Create optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(policy_params)

    # Initialize training state
    training_state = TrainingState(
        policy_params=policy_params,
        optimizer_state=optimizer_state,
        baseline=jnp.zeros((unroll_length,)),
    )

    # Create policy function
    make_policy = make_policy_fn(reinforce_network)

    jit_training_step = jax.jit(
        partial(
            training_step,
            env=env,
            network=reinforce_network,
            make_policy=make_policy,
            optimizer=optimizer,
            unroll_length=unroll_length,
            batch_size=batch_size,
            gamma=gamma,
        )
    )

    total_metrics = []
    for i in range(num_epochs):
        step_key, key = jr.split(key)

        training_state, metrics = jit_training_step(key=step_key, training_state=training_state)

        # Update baseline
        mean_returns = jnp.mean(metrics["returns"], axis=0)
        if i > 0:
            new_baseline = baseline_decay * training_state.baseline + (1 - baseline_decay) * mean_returns
        else:
            new_baseline = mean_returns

        training_state = TrainingState(
            optimizer_state=training_state.optimizer_state,
            policy_params=training_state.policy_params,
            baseline=new_baseline,
        )

        metrics = utils.aggregate_metrics(metrics)
        total_metrics.append(metrics)

    total_metrics = jax.tree_util.tree_map(lambda *xs: jnp.array(xs), *total_metrics)

    return make_policy, training_state, total_metrics


def evaluate(
    key: PRNGKeyArray,
    env: Env,
    params: types.Params,
    make_policy: Callable[[types.Params, bool], types.Policy],
    unroll_length: int,
    num_evals: int,
):
    policy = make_policy(params, True)

    def unroll_fn(
        carry: tuple[PRNGKeyArray, types.State], unused_t
    ) -> tuple[tuple[PRNGKeyArray, types.State], types.Trajectory]:
        key, state = carry
        policy_key, key = jax.random.split(key)
        action, _ = policy(state.obs, policy_key)
        new_state = env.step(state, action)
        return (key, new_state), types.Trajectory(
            position=new_state.sim_state.pos,
            reward=new_state.reward,
            discount=1 - new_state.done,
            metrics=new_state.metrics,
        )

    successes = []
    episode_rewards = []
    trajectories = []

    for _ in range(num_evals):
        key, reset_key = jr.split(key)
        env_state = env.reset(reset_key)
        _, trajectory = jax.lax.scan(unroll_fn, (key, env_state), (), length=unroll_length)

        _, episode_reward = compute_returns(trajectory.reward, trajectory.discount, 1.0)

        # Find terminal state
        terminal_state = jnp.where(trajectory.discount < 1.0)[0][0]

        # Slice everything in trajectory including terminal state
        trajectory = jax.tree_util.tree_map(lambda x: x[: terminal_state + 1], trajectory)

        success = trajectory.metrics["target_reached"][-1] == 1.0

        successes.append(success)
        episode_rewards.append(episode_reward)
        trajectories.append(trajectory)

    return successes, episode_rewards, trajectories

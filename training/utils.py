import jax
import jax.numpy as jnp
from envs.drone import Drone
from jaxtyping import Array
from training import types


@jax.jit
def aggregate_metrics(train_metrics: types.Metrics) -> types.Metrics:
    mean_episode_reward = jnp.mean(train_metrics["episode_reward"])
    std_episode_reward = jnp.std(train_metrics["episode_reward"])
    mean_returns_over_time = jnp.mean(train_metrics["returns"], axis=(0,))
    total_loss = train_metrics["total_loss"]

    return {
        "mean_episode_reward": mean_episode_reward,
        "std_episode_reward": std_episode_reward,
        "mean_returns_over_time": mean_returns_over_time,
        "total_loss": total_loss,
    }


def visualize_trajectory_3d(env: Drone, episode_reward: float, positions_array: Array):
    """Create a 3D visualization of the drone's trajectory.

    Args:
        env: Drone environment
        trajectory: Dictionary containing trajectory data
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D

    # Extract trajectory data
    positions = np.array(positions_array)

    # Create figure
    fig = plt.figure(figsize=(15, 10))

    # Create 3D plot
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.set_title(f"Drone Trajectory (Reward: {episode_reward:.2f})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")

    # Set axis limits based on world bounds
    world_min = env.bounds[0]
    world_max = env.bounds[1]
    ax1.set_xlim([world_min[0] - 0.2, world_max[0] + 0.2])
    ax1.set_ylim([world_min[1] - 0.2, world_max[1] + 0.2])
    ax1.set_zlim([world_min[2] - 0.2, world_max[2] + 0.2])

    # Draw world bounds as wireframe box
    r = [world_min[0], world_max[0]]
    s = [world_min[1], world_max[1]]
    t = [world_min[2], world_max[2]]

    # Combine the line segments for the wireframe box
    for i, j, k in [(i, j, k) for i in range(2) for j in range(2) for k in range(2)]:
        if i < 1:
            ax1.plot3D([r[i], r[i + 1]], [s[j], s[j]], [t[k], t[k]], "gray", alpha=0.3)
        if j < 1:
            ax1.plot3D([r[i], r[i]], [s[j], s[j + 1]], [t[k], t[k]], "gray", alpha=0.3)
        if k < 1:
            ax1.plot3D([r[i], r[i]], [s[j], s[j]], [t[k], t[k + 1]], "gray", alpha=0.3)

    # Plot trajectory
    ax1.plot3D(positions[:, 0], positions[:, 1], positions[:, 2], "b-", label="Trajectory", linewidth=2)

    # Mark start and end positions
    ax1.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color="green", s=100, label="Start")
    ax1.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color="red", s=100, label="End")

    # Mark target position
    target = env.target_position
    ax1.scatter(target[0], target[1], target[2], color="purple", s=100, marker="*", label="Target")

    # Draw obstacles as spheres
    obstacles = env.obstacles
    for obs in obstacles:
        # Create a wireframe sphere for each obstacle
        center = obs[:3]
        radius = obs[3]

        # Create sphere mesh
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        # Plot sphere
        ax1.plot_surface(x, y, z, color="red", alpha=0.3)

    # Add legend
    ax1.legend()

    # Create 2D projections plot
    ax2 = fig.add_subplot(122)
    ax2.set_title("Position over Time")

    time = np.arange(len(positions))
    ax2.plot(time, positions[:, 0], "r-", label="X Position")
    ax2.plot(time, positions[:, 1], "g-", label="Y Position")
    ax2.plot(time, positions[:, 2], "b-", label="Z Position")

    # Mark target position lines
    ax2.axhline(y=target[0], color="r", linestyle="--", alpha=0.5)
    ax2.axhline(y=target[1], color="g", linestyle="--", alpha=0.5)
    ax2.axhline(y=target[2], color="b", linestyle="--", alpha=0.5)

    # Add obstacle regions as shaded areas
    for obs in obstacles:
        center = obs[:3]
        radius = obs[3]

        # For each dimension, shade the obstacle region
        for i, color in enumerate(["r", "g", "b"]):
            ax2.axhspan(center[i] - radius, center[i] + radius, alpha=0.2, color=color)

    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Position")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    # Optional: Create an animation of the drone's movement
    def create_animation():
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Set up the plot
        ax.set_xlim([world_min[0] - 0.2, world_max[0] + 0.2])
        ax.set_ylim([world_min[1] - 0.2, world_max[1] + 0.2])
        ax.set_zlim([world_min[2] - 0.2, world_max[2] + 0.2])
        ax.set_title("Drone Navigation")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        # Draw the target
        ax.scatter(target[0], target[1], target[2], color="purple", s=100, marker="*")

        # Draw obstacles
        for obs in obstacles:
            center = obs[:3]
            radius = obs[3]

            # Create sphere mesh
            u = np.linspace(0, 2 * np.pi, 20)
            v = np.linspace(0, np.pi, 10)
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))

            # Plot sphere
            ax.plot_surface(x, y, z, color="red", alpha=0.3)

        # Initialize drone position
        drone = ax.scatter([], [], [], color="blue", s=80)

        # Initialize trail
        (trail,) = ax.plot([], [], [], "b-", alpha=0.5)

        def init():
            drone._offsets3d = (np.array([]), np.array([]), np.array([]))
            trail.set_data([], [])
            trail.set_3d_properties([])
            return [drone, trail]

        def animate(i):
            # Update drone position
            drone._offsets3d = (np.array([positions[i, 0]]), np.array([positions[i, 1]]), np.array([positions[i, 2]]))

            # Update trail
            trail.set_data(positions[: i + 1, 0], positions[: i + 1, 1])
            trail.set_3d_properties(positions[: i + 1, 2])

            return [drone, trail]

        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(positions), interval=50, blit=True)
        return anim

    # # Uncomment to create and show animation
    # return create_animation()

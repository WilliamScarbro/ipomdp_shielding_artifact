import gymnasium as gym
import numpy as np
from tqdm import trange

def collect_cartpole_data(
        num_episodes: int = 200,
        max_steps_per_episode: int = 200,
        render_mode: str = "rgb_array",
        seed: int = 0,
    ):
    env = gym.make("CartPole-v1", render_mode=render_mode)
    rng = np.random.default_rng(seed)

    frame_pairs = []   # list of (frame_{t-1}, frame_t)
    states      = []   # list of true state vectors [x, x_dot, theta, theta_dot]

    for ep in trange(num_episodes, desc="Collecting data"):
        obs, info = env.reset(seed=int(rng.integers(0, 10_000)))
        prev_frame = env.render()

        for t in range(max_steps_per_episode):
            # random policy is fine for state coverage to start with
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            cur_frame = env.render()

            # env.unwrapped.state is a 4-vector [x, x_dot, theta, theta_dot]
            true_state = np.array(env.unwrapped.state, dtype=np.float32)

            frame_pairs.append((prev_frame, cur_frame))
            states.append(true_state)

            prev_frame = cur_frame
            if terminated or truncated:
                break

    env.close()

    frame_pairs = np.array(frame_pairs, dtype=np.uint8)  # (N, 2, H, W, C)
    states      = np.array(states, dtype=np.float32)     # (N, 4)

    return frame_pairs, states


import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


import torch
from torch.optim import Adam


import matplotlib.pyplot as plt
import numpy as np

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

class CartPoleFramesDataset(Dataset):
    def __init__(self, frame_pairs: np.ndarray, states: np.ndarray):
        """
        frame_pairs: (N, 2, H, W, C) uint8
        states:      (N, 4) float32
        """
        assert frame_pairs.shape[0] == states.shape[0]
        self.frame_pairs = frame_pairs
        self.states = states

        # We will:
        # - convert to PIL
        # - to grayscale
        # - resize to 84x84
        # - to tensor (C,H,W)
        self.frame_transform = T.Compose([
            T.ToPILImage(),
            T.Grayscale(num_output_channels=1),
            T.Resize((84, 84)),
            T.ToTensor(),  # -> (1,84,84) float in [0,1]
        ])

    def __len__(self):
        return self.frame_pairs.shape[0]

    def __getitem__(self, idx):
        pair = self.frame_pairs[idx]   # (2, H, W, C)
        state = self.states[idx]       # (4,)

        f1 = pair[0]  # (H, W, C)
        f2 = pair[1]

        f1_t = self.frame_transform(f1)  # (1,84,84)
        f2_t = self.frame_transform(f2)  # (1,84,84)

        # stack along channel dimension -> (2,84,84)
        x = torch.cat([f1_t, f2_t], dim=0)
        y = torch.from_numpy(state)      # (4,)

        return x, y

def make_dataloaders(frame_pairs, states, batch_size=64, val_split=0.1, test_split=0.1):
    N = frame_pairs.shape[0]
    idx = np.random.permutation(N)
    split_val = int(N * (1 - val_split - test_split))
    split_test = int(N * (1 - test_split))
    train_idx, val_idx, test_idx = idx[:split_val], idx[split_val:split_test], idx[split_test:]

    train_ds = CartPoleFramesDataset(frame_pairs[train_idx], states[train_idx])
    val_ds   = CartPoleFramesDataset(frame_pairs[val_idx],   states[val_idx])
    test_ds  = CartPoleFramesDataset(frame_pairs[test_idx],   states[test_idx])
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader   = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


class CartPoleStateNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: (2, 84, 84)
        self.conv = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=5, stride=2, padding=2),  # -> (16, 42, 42)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # -> (32, 21, 21)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # -> (64, 11, 11)
            nn.ReLU(),
        )

        # 64 * 11 * 11 = 7744
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 11 * 11, 256),
            nn.ReLU(),
            nn.Linear(256, 4),   # x, x_dot, theta, theta_dot
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

def train_state_net(train_loader, val_loader, epochs=20, lr=1e-3, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = CartPoleStateNet().to(device)
    opt = Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)   # (B,2,84,84)
            y = y.to(device)   # (B,4)

            opt.zero_grad()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            opt.step()
            train_loss += loss.item() * x.size(0)

        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                val_loss += loss.item() * x.size(0)
        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch:02d} | train MSE: {train_loss:.4f} | val MSE: {val_loss:.4f}")

    return model

def collect_predictions(model, test_loader, device="cuda"):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ys, y_hats = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y_hat = model(x).cpu().numpy()   # (B, 4)
            ys.append(y.numpy())             # (B, 4)
            y_hats.append(y_hat)

    y_true = np.concatenate(ys, axis=0)      # (N, 4)
    y_pred = np.concatenate(y_hats, axis=0)  # (N, 4)
    return y_true, y_pred

# Example usage:
# y_true, y_pred = collect_predictions(model, test_loader)

def make_confusion_matrices(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        min_avg_count: int = 50,
        max_bins: int = 15,
        min_bins: int = 3,
    ):
    """
    Build one confusion matrix per state dimension.

    y_true, y_pred: arrays of shape (N, 4)
    Returns:
      cms: list of length 4, each an array (k_d, k_d) of counts
      edges_list: list of length 4, each the bin edges used for that dimension
    """
    assert y_true.shape == y_pred.shape
    N, dims = y_true.shape
    assert dims == 4

    cms = []
    edges_list = []

    for d in range(dims):
        # Decide number of bins for this dimension
        # Require N / k^2 >= min_avg_count  =>  k <= sqrt(N / min_avg_count)
        # k_max_due_to_counts = int(np.floor(np.sqrt(max(N / min_avg_count, 1))))
        k = 7 #max(min_bins, min(max_bins, k_max_due_to_counts))

        # If N is very small, this may give k == min_bins; that's fine.
        # Compute bin edges using combined range (true + predicted)
        v_min = float(min(y_true[:, d].min(), y_pred[:, d].min()))
        v_max = float(max(y_true[:, d].max(), y_pred[:, d].max()))
        if v_min == v_max:
            # Degenerate case: everything is the same
            v_min -= 1e-6
            v_max += 1e-6

        edges = np.linspace(v_min, v_max, num=k + 1)

        # Bin indices: 0..k-1
        true_bins = np.clip(np.digitize(y_true[:, d], edges) - 1, 0, k - 1)
        pred_bins = np.clip(np.digitize(y_pred[:, d], edges) - 1, 0, k - 1)

        cm = np.zeros((k, k), dtype=int)
        for tb, pb in zip(true_bins, pred_bins):
            cm[tb, pb] += 1

        cms.append(cm)
        edges_list.append(edges)

        avg_count = cm.sum() / (k * k)
        print(f"Dim {d}: k={k}, avg count per cell ~ {avg_count:.1f}")

    return cms, edges_list

# Example:
# cms, edges_list = make_confusion_matrices(y_true, y_pred, min_avg_count=50)

def plot_confusion_matrices(cms, edges_list, dim_names=None, figsize=(12, 10)):
    """
    cms: list of confusion matrices, one per dimension
    edges_list: list of bin edges per dimension
    dim_names: optional list of dimension names, length = len(cms)
    """
    D = len(cms)
    if dim_names is None:
        dim_names = [f"dim {i}" for i in range(D)]

    cols = 2
    rows = int(np.ceil(D / cols))
    plt.figure(figsize=figsize)

    for d in range(D):
        cm = cms[d]
        k = cm.shape[0]

        plt.subplot(rows, cols, d + 1)
        plt.imshow(cm, cmap="Blues", interpolation="nearest")
        plt.title(f"Confusion Matrix: {dim_names[d]}")
        plt.xlabel("Estimated bin")
        plt.ylabel("True bin")
        plt.colorbar()

        # Annotate each cell with count
        for i in range(k):
            for j in range(k):
                plt.text(j, i, str(cm[i, j]),
                         ha="center", va="center",
                         color="black" if cm[i, j] < cm.max() / 2 else "white",
                         fontsize=8)

        # optional: set ticks to bin midpoints
        plt.xticks(range(k))
        plt.yticks(range(k))

    plt.tight_layout()
    plt.show()

if __name__=="__main__":
    # 1. collect data
    frame_pairs, states = collect_cartpole_data(num_episodes=80)

    print(f"finished collecting data, collected {len(frame_pairs)} observations")
    
    # 2. loaders
    train_loader, val_loader, test_loader = make_dataloaders(frame_pairs, states, batch_size=64)

    print("finished building loaders")
    
    # 3. train
    model = train_state_net(train_loader, val_loader, epochs=10, lr=1e-3)

    
    # 4. Collect performance data
    y_true, y_pred = collect_predictions(model, test_loader)

    # 5. Build confusion matrices
    cms, edges_list = make_confusion_matrices(y_true, y_pred, min_avg_count=50)

    dim_names = ["x", "x_dot", "theta", "theta_dot"]
    plot_confusion_matrices(cms, edges_list, dim_names)


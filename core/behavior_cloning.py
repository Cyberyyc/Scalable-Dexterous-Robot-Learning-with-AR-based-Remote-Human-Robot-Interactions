"""Stage I: behavior cloning of the SAC actor on aligned expert transitions."""
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def pretrain_actor(model, observations, actions, epochs=200, batch_size=128,
                   learning_rate=1e-3, actions_are_scaled=False):
    model.set_expert_data(observations, actions, actions_are_scaled=actions_are_scaled)
    dataset = TensorDataset(torch.as_tensor(model.expert_obs), torch.as_tensor(model.expert_acts))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.actor.parameters(), lr=learning_rate)
    history = []
    model.actor.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for obs, acts in loader:
            obs_np = model._normalize_expert_obs(obs.numpy())
            obs = torch.as_tensor(obs_np, device=model.device)
            acts = acts.to(model.device)
            prediction = model.actor(obs, deterministic=True)
            loss = torch.nn.functional.mse_loss(prediction, acts)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(obs)
            count += len(obs)
        history.append(total / count)
    model.actor.eval()
    return history


def load_transitions(path):
    with np.load(path, allow_pickle=False) as data:
        return data["observations"].astype(np.float32), data["actions"].astype(np.float32)

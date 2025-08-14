"""Behavior-cloned SAC with an expert-paired projection objective."""
import numpy as np
import torch as th
from torch import nn
import torch.nn.functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


class CustomSAC(SAC):
    def __init__(self, *args, lambda_smooth=0.5, projector_hidden_dim=256,
                 projector_output_dim=128, projector_lr=7.3e-4,
                 contrastive_temperature=0.1, **kwargs):
        self.lambda_smooth = lambda_smooth
        self.projector_hidden_dim = projector_hidden_dim
        self.projector_output_dim = projector_output_dim
        self.projector_lr = projector_lr
        self.contrastive_temperature = contrastive_temperature
        self.expert_obs = self.expert_acts = None
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        super()._setup_model()
        action_dim = int(np.prod(self.action_space.shape))
        width = self.projector_hidden_dim
        self.projector = nn.Sequential(
            nn.Linear(action_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, self.projector_output_dim),
        ).to(self.device)
        self.projector_optimizer = th.optim.Adam(self.projector.parameters(), lr=self.projector_lr)

    def set_expert_data(self, observations, actions, *, actions_are_scaled=False):
        observations = np.asarray(observations, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if observations.shape[1:] != self.observation_space.shape:
            raise ValueError("Expert observations must match the environment observation space.")
        if actions.shape[1:] != self.action_space.shape or len(observations) != len(actions):
            raise ValueError("Expert observations and actions must be aligned transitions.")
        if len(observations) < 2 or not np.isfinite(observations).all() or not np.isfinite(actions).all():
            raise ValueError("At least two finite expert transitions are required.")
        self.expert_obs = observations.copy()
        self.expert_acts = actions.copy() if actions_are_scaled else self.policy.scale_action(actions)

    def _normalize_expert_obs(self, observations):
        return self._vec_normalize_env.normalize_obs(observations) if self._vec_normalize_env else observations

    def contrastive_loss(self, policy_actions, expert_actions, temperature=None):
        temperature = self.contrastive_temperature if temperature is None else temperature
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        z_policy = F.normalize(self.projector(policy_actions), dim=1)
        z_expert = F.normalize(self.projector(expert_actions), dim=1)
        logits = z_policy @ z_expert.T / temperature
        labels = th.arange(logits.shape[0], device=logits.device)
        return F.cross_entropy(logits, labels)

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["expert_obs", "expert_acts"]

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts + ["projector", "projector_optimizer"], variables

    def train(self, gradient_steps: int, batch_size: int=64) -> None:
        """
            执行训练
            """
        if self.expert_obs is None:
            raise ValueError("Attach expert observations and actions before training.")
        self.policy.set_training_mode(True)
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]
        self._update_learning_rate(optimizers)
        ent_coef_losses, ent_coefs = ([], [])
        actor_losses, critic_losses = ([], [])
        contrastive_losses = []
        for gradient_step in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)
            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor
            ent_coefs.append(ent_coef.item())
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()
            with th.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
            current_q_values = self.critic(replay_data.observations, replay_data.actions)
            critic_loss = 0.5 * sum((F.mse_loss(current_q, target_q_values) for current_q in current_q_values))
            critic_losses.append(critic_loss.item())
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            expert_indices = np.random.randint(0, len(self.expert_obs), size=batch_size)
            expert_batch_obs = th.tensor(self._normalize_expert_obs(self.expert_obs[expert_indices]), dtype=th.float32).to(self.device)
            expert_batch_acts = th.tensor(self.expert_acts[expert_indices], dtype=th.float32).to(self.device)
            policy_actions_expert, _ = self.actor.action_log_prob(expert_batch_obs)
            contrastive_loss_value = self.contrastive_loss(policy_actions_expert, expert_batch_acts)
            contrastive_losses.append(contrastive_loss_value.item())
            total_actor_loss = actor_loss + self.lambda_smooth * contrastive_loss_value
            actor_losses.append(total_actor_loss.item())
            self.actor.optimizer.zero_grad()
            self.projector_optimizer.zero_grad()
            total_actor_loss.backward()
            self.actor.optimizer.step()
            self.projector_optimizer.step()
            if (self._n_updates + gradient_step + 1) % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)
        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates)
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        self.logger.record("train/contrastive_loss", np.mean(contrastive_losses))


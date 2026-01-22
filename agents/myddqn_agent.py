from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from copy import deepcopy
from xuance.torch.learners import REGISTRY_Learners
from learner.myddqn_learner import MyDDQNLearner

import torch
from torch import nn
from tqdm import tqdm
import numpy as np


class MyDDQNAgent(DQN_Agent):
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        # Registry your pre-defined learner BEFORE calling super().__init__
        REGISTRY_Learners['MyDDQNLearner'] = MyDDQNLearner
        super(MyDDQNAgent, self).__init__(config, envs, observation_space, action_space, callback)

        # ---- FIX: local RNG for masked random exploration ----
        base_seed = int(getattr(config, "seed", 0) or 0)
        env_seed = int(getattr(config, "env_seed", 0) or 0)
        rank = int(getattr(self, "rank", 0) or 0)
        self.np_rng = np.random.default_rng(base_seed + 1000 * rank + env_seed)

    # ---------------------------------------------------------------------
    # Masked action selection (critical for resilience / anti-damage)
    # ---------------------------------------------------------------------
    def _get_eps(self) -> float:
        """
        Get current epsilon for epsilon-greedy exploration.
        Xuance DQN agents usually maintain self.e_greedy, updated by _update_explore_factor().
        """
        eps = getattr(self, "e_greedy", None)
        if eps is None:
            eps = getattr(self, "explore_factor", 0.0)
        try:
            return float(eps)
        except Exception:
            return 0.0

    def _to_tensor_obs(self, obs):
        """Convert obs to torch.Tensor on self.device, without changing shape."""
        if torch.is_tensor(obs):
            return obs.to(self.device)
        return torch.as_tensor(obs, dtype=torch.float32, device=self.device)

    def _extract_action_mask_from_obs(self, obs):
        """
        Derive action mask from observation itself.

        Your obs layout:
          [current_node, dst_node, (nbr_id, delay, bw) * max_degree]
        nbr_id == -1 indicates invalid slot. So:
          mask[i, a] = (obs[i, 2 + 3*a] >= 0)
        """
        if torch.is_tensor(obs):
            nbr_ids = obs[:, 2::3]  # (N, max_degree)
            return nbr_ids >= 0.0
        else:
            obs_np = np.asarray(obs)
            nbr_ids = obs_np[:, 2::3]
            return nbr_ids >= 0.0

    def _masked_greedy_actions(self, q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        q_values: (N, A)
        mask: (N, A) bool
        return: greedy actions within valid set, shape (N,)
        """
        q_masked = q_values.clone()
        q_masked[~mask] = -1e9
        return torch.argmax(q_masked, dim=-1)

    def _sample_random_valid_actions(self, mask: np.ndarray) -> np.ndarray:
        """
        mask: (N, A) numpy bool
        return random actions only from valid actions, shape (N,)
        """
        N, A = mask.shape
        acts = np.zeros((N,), dtype=np.int64)
        for i in range(N):
            valid = np.flatnonzero(mask[i])
            if valid.size == 0:
                acts[i] = 0  # fallback
            else:
                acts[i] = int(self.np_rng.choice(valid))  # <-- FIX: use self.np_rng
        return acts

    def action(self, obs, test_mode: bool = False) -> dict:
        """
        Override action selection:
          - If use_actions_mask is False: fall back to parent implementation.
          - If True: apply action mask derived from obs to:
              * greedy selection (argmax over valid actions only)
              * random exploration (sample only from valid actions)
        """
        if not getattr(self, "use_actions_mask", False):
            return super().action(obs, test_mode=test_mode)

        obs_t = self._to_tensor_obs(obs)

        # policy(obs) -> (rep_out, argmax_action, evalQ)
        rep_out, argmax_action, evalQ = self.policy(obs_t)  # evalQ: (N, A)

        # Build mask from obs itself
        mask_t = self._extract_action_mask_from_obs(obs_t)
        if not torch.is_tensor(mask_t):
            mask_t = torch.as_tensor(mask_t, dtype=torch.bool, device=self.device)
        else:
            mask_t = mask_t.to(self.device)

        greedy_acts_t = self._masked_greedy_actions(evalQ, mask_t)  # (N,)

        if test_mode:
            final_acts_t = greedy_acts_t
        else:
            eps = self._get_eps()
            N = evalQ.shape[0]
            explore = (torch.rand((N,), device=self.device) < eps)

            mask_np = mask_t.detach().cpu().numpy()
            rand_acts_np = self._sample_random_valid_actions(mask_np)
            rand_acts_t = torch.as_tensor(rand_acts_np, dtype=torch.int64, device=self.device)

            final_acts_t = greedy_acts_t.clone()
            final_acts_t[explore] = rand_acts_t[explore]

        policy_out = {
            "actions": final_acts_t.detach().cpu().numpy(),
            "greedy_actions": greedy_acts_t.detach().cpu().numpy(),
            "q_values": evalQ.detach().cpu().numpy(),
            "action_mask": mask_t.detach().cpu().numpy(),
        }
        return policy_out

    # ---------------------------------------------------------------------
    # Train / Test (unchanged except calling self.action)
    # ---------------------------------------------------------------------
    def train(self, train_steps: int) -> dict:
        train_info = {}
        obs = self.train_envs.buf_obs
        for _ in tqdm(range(train_steps)):
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)

            policy_out = self.action(obs, test_mode=False)
            acts = policy_out['actions']
            next_obs, rewards, terminals, truncations, infos = self.train_envs.step(acts)

            self.callback.on_train_step(self.current_step, envs=self.train_envs, policy=self.policy,
                                        obs=obs, policy_out=policy_out, acts=acts, next_obs=next_obs, rewards=rewards,
                                        terminals=terminals, truncations=truncations, infos=infos,
                                        train_steps=train_steps)

            self.memory.store(obs, acts, self._process_reward(rewards), terminals, self._process_observation(next_obs))

            if self.current_step > self.start_training and self.current_step % self.training_frequency == 0:
                update_info = self.train_epochs(n_epochs=self.n_epochs)
                self.log_infos(update_info, self.current_step)
                train_info.update(update_info)
                self.callback.on_train_epochs_end(self.current_step, policy=self.policy, memory=self.memory,
                                                  current_episode=self.current_episode, train_steps=train_steps,
                                                  update_info=update_info)

            self.returns = self.gamma * self.returns + rewards
            obs = deepcopy(next_obs)
            for i in range(self.n_envs):
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        self.train_envs.buf_obs[i] = obs[i]
                        self.ret_rms.update(self.returns[i:i + 1])
                        self.returns[i] = 0.0
                        self.current_episode[i] += 1
                        if self.use_wandb:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}/env-{i}": infos[i]["episode_step"],
                                f"Train-Episode-Rewards/rank_{self.rank}/env-{i}": infos[i]["episode_score"]
                            }
                        else:
                            episode_info = {
                                f"Episode-Steps/rank_{self.rank}": {f"env-{i}": infos[i]["episode_step"]},
                                f"Train-Episode-Rewards/rank_{self.rank}": {f"env-{i}": infos[i]["episode_score"]}
                            }
                        self.log_infos(episode_info, self.current_step)
                        train_info.update(episode_info)
                        self.callback.on_train_episode_info(envs=self.train_envs, policy=self.policy, env_id=i,
                                                            infos=infos, rank=self.rank, use_wandb=self.use_wandb,
                                                            current_step=self.current_step,
                                                            current_episode=self.current_episode,
                                                            train_steps=train_steps)

            self.current_step += self.n_envs
            self._update_explore_factor()
            self.callback.on_train_step_end(self.current_step, envs=self.train_envs, policy=self.policy,
                                            train_steps=train_steps, train_info=train_info)
        return train_info

    def test(self,
             test_episodes: int,
             test_envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
             close_envs: bool = True) -> list:

        if test_envs is None:
            raise ValueError("`test_envs` must be provided for evaluation.")
        num_envs = test_envs.num_envs
        videos, episode_videos, images = [[] for _ in range(num_envs)], [], None
        current_episode, current_step, scores, best_score = 0, 0, [], -np.inf
        obs, infos = test_envs.reset()

        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=True)

            next_obs, rewards, terminals, truncations, infos = test_envs.step(policy_out['actions'])

            self.callback.on_test_step(envs=test_envs, policy=self.policy, images=images,
                                       obs=obs, policy_out=policy_out, next_obs=next_obs, rewards=rewards,
                                       terminals=terminals, truncations=truncations, infos=infos,
                                       current_train_step=self.current_step,
                                       current_step=current_step, current_episode=current_episode)

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or truncations[i]:
                    if self.atari and (~truncations[i]):
                        pass
                    else:
                        obs[i] = infos[i]["reset_obs"]
                        scores.append(infos[i]["episode_score"])
                        current_episode += 1
                        if best_score < infos[i]["episode_score"]:
                            best_score = infos[i]["episode_score"]

                    print(f"Info src: {infos[i].get('src')}")
                    print(f"Info dst: {infos[i].get('dst')}")
                    print(f"Info path: {infos[i].get('path')}")
                    print(f"Info path_delay: {infos[i].get('path_delay')}")
                    print(f"Info shortest_path: {infos[i].get('shortest_path')}")
                    print(f"Info shortest_path_delay: {infos[i].get('shortest_path_delay')}")

                    if "failure_happened" in infos[i]:
                        print(f"Info failure_mode: {infos[i].get('failure_mode')}")
                        print(f"Info fail_step: {infos[i].get('fail_step')}")
                        print(f"Info fail_num: {infos[i].get('fail_num')}")
                        print(f"Info dead_edges: {infos[i].get('dead_edges')}")
                        print(f"Info dead_nodes: {infos[i].get('dead_nodes')}")
                        print(f"Info is_connected_src_dst: {infos[i].get('is_connected_src_dst')}")

            current_step += num_envs

        self.callback.on_test_end(envs=test_envs, policy=self.policy,
                                  current_train_step=self.current_step,
                                  current_step=current_step, current_episode=current_episode,
                                  scores=scores, best_score=best_score)

        if close_envs:
            test_envs.close()

        return scores


class MyPolicy(nn.Module):
    """
    An example of self-defined policy.
    """

    def __init__(self, representation: nn.Module, hidden_dim: int, n_actions: int, device: torch.device):
        super(MyPolicy, self).__init__()
        self.representation = representation
        self.feature_dim = self.representation.output_shapes['state'][0]
        self.q_net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        ).to(device)
        self.target_q_net = deepcopy(self.q_net)

    def forward(self, observation):
        output_rep = self.representation(observation)
        output = self.q_net(output_rep['state'])
        argmax_action = output.argmax(dim=-1)
        return output_rep, argmax_action, output

    def target(self, observation):
        outputs_target = self.representation(observation)
        Q_target = self.target_q_net(outputs_target['state'])
        argmax_action = Q_target.argmax(dim=-1)
        return outputs_target, argmax_action.detach(), Q_target.detach()

    def copy_target(self):
        for ep, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            tp.data.copy_(ep)

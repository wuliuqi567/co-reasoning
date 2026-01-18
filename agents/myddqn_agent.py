from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from copy import deepcopy
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.learners import REGISTRY_Learners
from learner.myddqn_learner import MyDDQNLearner
import torch
from torch import nn
from tqdm import tqdm
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


class MyPolicy(nn.Module):
    """
    An example of self-defined policy.

    Args:
        representation (nn.Module): A neural network module responsible for extracting meaningful features
            from the raw observations provided by the environment.
        hidden_dim (int): Specifies the number of units in each hidden layer, determining the model’s
            capacity to capture complex patterns.
        n_actions (int): The total number of discrete actions available to the agent in the environment.
        device (torch.device): The calculating device.

    Note:
        The inputs to the __init__ method are not rigidly defined. You can extend or modify them as needed
        to accommodate additional settings or configurations specific to your application.
    """

    def __init__(self, representation: nn.Module, hidden_dim: int, n_actions: int, device: torch.device):
        super(MyPolicy, self).__init__()
        self.representation = representation  # Specify the representation.
        self.feature_dim = self.representation.output_shapes['state'][0]  # Dimension of the representation's output.
        self.q_net = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        ).to(device)  # The Q network.
        self.target_q_net = deepcopy(self.q_net)  # Target Q network.

    def forward(self, observation):
        output_rep = self.representation(observation)  # Get the output of the representation module.
        output = self.q_net(output_rep['state'])  # Get the output of the Q network.
        argmax_action = output.argmax(dim=-1)  # Get greedy actions.
        return output_rep, argmax_action, output

    def target(self, observation):
        outputs_target = self.representation(observation)  # Get the output of the representation module.
        Q_target = self.target_q_net(outputs_target['state'])  # Get the output of the target Q network.
        argmax_action = Q_target.argmax(dim=-1)  # Get greedy actions that output by target Q network.
        return outputs_target, argmax_action.detach(), Q_target.detach()

    def copy_target(self):  # Reset the parameters of target Q network as the Q network.
        for ep, tp in zip(self.q_net.parameters(), self.target_q_net.parameters()):
            tp.data.copy_(ep)
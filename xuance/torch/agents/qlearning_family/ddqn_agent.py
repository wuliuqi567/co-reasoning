from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from copy import deepcopy

class DDQN_Agent(DQN_Agent):
    """The implementation of Double DQN agent.

    Args:
        config: the Namespace variable that provides hyperparameters and other settings.
        envs: the vectorized environments.
        callback: A user-defined callback function object to inject custom logic during training.
    """
    def __init__(
            self,
            config: Namespace,
            envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            observation_space: Optional[Space] = None,
            action_space: Optional[Space] = None,
            callback: Optional[BaseCallback] = None
    ):
        super(DDQN_Agent, self).__init__(config, envs, observation_space, action_space, callback)

    def test(self,
            test_episodes: int,
            test_envs: Optional[DummyVecEnv | SubprocVecEnv] = None,
            close_envs: bool = True) -> list:

        if test_envs is None:
            raise ValueError("`test_envs` must be provided for evaluation.")
        num_envs = test_envs.num_envs
        current_episode, current_step, paths, infos = 0, 0, [], None
        obs, infos = test_envs.reset()
        while current_episode < test_episodes:
            self.obs_rms.update(obs)
            obs = self._process_observation(obs)
            policy_out = self.action(obs, test_mode=True)
            next_obs, rewards, terminals, truncations, infos = test_envs.step(policy_out['actions'])

            obs = deepcopy(next_obs)
            for i in range(num_envs):
                if terminals[i] or truncations[i]:
                    paths.append(infos[i]["path"])
                    infos.append(infos[i])
                    current_episode += 1

            current_step += num_envs

        if close_envs:
            test_envs.close()
        
        print(f"Paths: {paths}")
        print(f"Infos: {infos}")

        return paths, infos
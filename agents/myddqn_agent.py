from argparse import Namespace
from gymnasium.spaces import Space
from xuance.common import Optional, BaseCallback
from xuance.environment import DummyVecEnv, SubprocVecEnv
from xuance.torch.agents.qlearning_family.dqn_agent import DQN_Agent
from copy import deepcopy
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.learners import REGISTRY_Learners
from learner.myddqn_learner import MyDDQNLearner

class MyAgent(OffPolicyAgent):
    def __init__(self, config, envs):
        super(MyAgent, self).__init__(config, envs)
        self.policy = self._build_policy()  # Build the policy module.
        self.memory = self._build_memory()  # Build the replay buffer.
        REGISTRY_Learners['MyLearner'] = MyDDQNLearner  # Registry your pre-defined learner.
        self.learner = self._build_learner(self.config, self.policy)  # Build the learner.

    def _build_policy(self):
        # First create the representation module.
        representation = self._build_representation("Basic_MLP", self.observation_space, self.config)
        # Build your custom policy module.
        policy = MyPolicy(representation, 64, self.action_space.n, self.config.device)
        return policy


class MyPolicy(nn.Module):
"""
An example of self-defined policy.

Args:
    representation (nn.Module): A neural network module responsible for extracting meaningful features from the raw observations provided by the environment.
    hidden_dim (int): Specifies the number of units in each hidden layer, determining the model’s capacity to capture complex patterns.
    n_actions (int): The total number of discrete actions available to the agent in the environment.
    device (torch.device): The calculating device.


Note: The inputs to the __init__ method are not rigidly defined. You can extend or modify them as needed to accommodate additional settings or configurations specific to your application.
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
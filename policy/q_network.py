import os
import torch
import numpy as np
import torch.nn as nn
from xuance.common import Sequence, Optional, Callable, Union
from copy import deepcopy
from gymnasium.spaces import Space, Discrete
from xuance.torch import Module, Tensor, DistributedDataParallel
from xuance.torch.utils import ModuleType
from xuance.torch.policies import BasicQhead


class Q_Network(Module):
    """
    The base class to implement DQN based policy

    Args:
        action_space (Discrete): The action space, which type is gym.spaces.Discrete.
        representation (Module): The representation module.
        hidden_size (Sequence[int]): List of hidden units for fully connect layers.
        normalize (Optional[ModuleType]): The layer normalization over a minibatch of inputs.
        initialize (Optional[Callable[..., Tensor]]): The parameters initializer.
        activation (Optional[ModuleType]): The activation function for each layer.
        device (Optional[Union[str, int, torch.device]]): The calculating device.
        use_distributed_training (bool): Whether to use multi-GPU for distributed training.
    """

    def __init__(self,
                 action_space: Discrete,
                 representation: Module,
                 hidden_size: Sequence[int] = None,
                 normalize: Optional[ModuleType] = None,
                 initialize: Optional[Callable[..., Tensor]] = None,
                 activation: Optional[ModuleType] = None,
                 device: Optional[Union[str, int, torch.device]] = None,
                 use_distributed_training: bool = False):
        super(Q_Network, self).__init__()
        self.action_dim = action_space.n
        self.representation = representation
        self.target_representation = deepcopy(representation)
        self.representation_info_shape = self.representation.output_shapes
        self.eval_Qhead = BasicQhead(self.representation.output_shapes['state'][0], self.action_dim, hidden_size,
                                     normalize, initialize, activation, device)
        self.target_Qhead = deepcopy(self.eval_Qhead)

        # Prepare DDP module.
        self.distributed_training = use_distributed_training
        if self.distributed_training:
            self.rank = int(os.environ["RANK"])
            if self.representation._get_name() != "Basic_Identical":
                self.representation = DistributedDataParallel(module=self.representation, device_ids=[self.rank])
            self.eval_Qhead = DistributedDataParallel(module=self.eval_Qhead, device_ids=[self.rank])

    def forward(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs: The hidden state output by the representation.
            argmax_action: The greedy actions.
            evalQ: The evaluated Q-values.
        """
        outputs = self.representation(observation)
        evalQ = self.eval_Qhead(outputs['state'])
        argmax_action = evalQ.argmax(dim=-1)
        return outputs, argmax_action, evalQ

    def target(self, observation: Union[np.ndarray, dict]):
        """
        Returns the output of the representation, greedy actions, and the evaluated Q-values via target networks.

        Parameters:
            observation: The original observation input.

        Returns:
            outputs_target: The hidden state output by the representation.
            argmax_action: The greedy actions from target networks.
            targetQ: The evaluated Q-values output by target Q-network.
        """
        outputs_target = self.target_representation(observation)
        targetQ = self.target_Qhead(outputs_target['state'])
        argmax_action = targetQ.argmax(dim=-1)
        return outputs_target, argmax_action.detach(), targetQ.detach()

    def copy_target(self):
        for ep, tp in zip(self.representation.parameters(), self.target_representation.parameters()):
            tp.data.copy_(ep)
        for ep, tp in zip(self.eval_Qhead.parameters(), self.target_Qhead.parameters()):
            tp.data.copy_(ep)
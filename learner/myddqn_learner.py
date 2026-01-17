import torch
from torch import nn
from xuance.torch.learners import Learner
from argparse import Namespace


class MyDDQNLearner(Learner):
    def __init__(self, config, policy):
        super(MyDDQNLearner, self).__init__(config, policy)
        # Build the optimizer.
        self.optimizer = torch.optim.Adam(self.policy.parameters(), self.config.learning_rate, eps=1e-5)
        self.loss = nn.MSELoss()  # Build a loss function.
        self.sync_frequency = config.sync_frequency  # The period to synchronize the target network.

    def update(self, **samples):
        info = {}
        self.iterations += 1
        '''Get a batch of training samples.'''
        obs_batch = torch.as_tensor(samples['obs'], device=self.device)
        act_batch = torch.as_tensor(samples['actions'], device=self.device)
        next_batch = torch.as_tensor(samples['obs_next'], device=self.device)
        rew_batch = torch.as_tensor(samples['rewards'], device=self.device)
        ter_batch = torch.as_tensor(samples['terminals'], dtype=torch.float, device=self.device)

        # Feedforward steps.
        _, _, q_eval = self.policy(obs_batch)
        _, _, q_next = self.policy.target(next_batch)
        q_next_action = q_next.max(dim=-1).values
        q_eval_action = q_eval.gather(-1, act_batch.long().unsqueeze(-1)).reshape(-1)
        target_value = rew_batch + (1 - ter_batch) * self.gamma * q_next_action
        loss = self.loss(q_eval_action, target_value.detach())

        # Backward and optimizing steps.
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Synchronize the target network
        if self.iterations % self.sync_frequency == 0:
            self.policy.copy_target()

        # Set the variables you need to observe.
        info.update({'loss': loss.item(),
                     'iterations': self.iterations,
                     'q_eval_action': q_eval_action.mean().item()})

        return info
import sys
import os

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from agents.myddqn_agent import MyDDQNAgent

from environment.net_tupu_inner import NetTupuInner
from xuance.environment import REGISTRY_ENV


def parse_args():
    parser = argparse.ArgumentParser("Inner-field Double DQN for NetEnv.")
    parser.add_argument("--env-id", type=str, default="InnerNetEnv-Net-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="../config/inner_ddqn.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)

    configs = argparse.Namespace(**configs_dict)
    REGISTRY_ENV[configs.env_name] = NetTupuInner

    if configs.test:
        configs.logger = "tensorboard"
        configs.benchmark = 0
        configs.parallels = 1
        configs.vectorize = "DummyVecEnv"

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = MyDDQNAgent(config=configs, envs=envs)

    train_information = {
        "Deep learning toolbox": configs.dl_toolbox,
        "Calculating device": configs.device,
        "Algorithm": configs.agent,
        "Environment": configs.env_name,
        "Scenario": configs.env_id,
        "Graph source": configs.graph_source,
    }
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.benchmark:
        def env_fn():
            configs_test = deepcopy(configs)
            configs_test.parallels = configs_test.test_episode
            return make_envs(configs_test)

        train_steps = configs.running_steps // configs.parallels
        eval_interval = configs.eval_interval // configs.parallels
        test_episode = configs.test_episode
        num_epoch = int(train_steps / eval_interval)

        test_scores = Agent.test(test_episode, env_fn())
        Agent.save_model(model_name="best_model.pth")
        best_scores_info = {
            "mean": np.mean(test_scores),
            "std": np.std(test_scores),
            "step": Agent.current_step,
        }
        for i_epoch in range(num_epoch):
            print("Epoch: %d/%d:" % (i_epoch, num_epoch))
            Agent.train(eval_interval)
            test_scores = Agent.test(test_episode, env_fn())

            if np.mean(test_scores) > best_scores_info["mean"]:
                best_scores_info = {
                    "mean": np.mean(test_scores),
                    "std": np.std(test_scores),
                    "step": Agent.current_step,
                }
                Agent.save_model(model_name="best_model.pth")

        print(
            "Best Model Score: %.2f, std=%.2f"
            % (best_scores_info["mean"], best_scores_info["std"])
        )
    else:
        if configs.test:
            def env_fn():
                configs.parallels = configs.test_episode
                return make_envs(configs)

            Agent.load_model(path=Agent.model_dir_load, model=getattr(configs, "load_model", ""))
            Agent.test(configs.test_episode, env_fn())
        else:
            Agent.train(configs.running_steps // configs.parallels)
            Agent.save_model("final_train_model.pth")
            print("Finish inner-field training!")

    Agent.finish()

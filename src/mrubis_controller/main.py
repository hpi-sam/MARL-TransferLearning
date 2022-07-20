import argparse
import os
import sys
from time import sleep

import wandb
from marl.agent.actor import LinearActor
from marl.agent.critic import LinearConcatCritic
from marl.mrubis_env import MrubisEnv
from marl.runner import ReplayBufferRunner, Runner
from marl.shop_agent_controller import ShopAgentController
from marl.master_project.master_baseline_runner import MasterBaselineRunner
from marl.master_project.multi_agent_controller import MultiAgentController
from marl.agent_new import Agent


class MrubisStarter:
    def __init__(self):
        pass

    def __enter__(self):
        print("Starting mRUBiS")
        os.system(
            "tmux new-session -d -s mrubis -n mrubis 'CWD=$(pwd) && cd ../mRUBiS/ML_based_Control/ && java -jar mRUBiS.jar > ${CWD}/mrubis.log'")

    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing mRUBiS")
        os.system("tmux kill-session -t mrubis")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", action="store_true",
                        default=False, help="start rl runner with dl")
    parser.add_argument("--runner-rp", action="store_true", default=False,
                        help="start rl runner with replay buffer with dl")
    parser.add_argument("--runner-master", action="store_true", default=False,
                        help="start rl runner with the baseline of the masters project")
    parser.add_argument("--wandb", action="store_true", help="Log with wandb")
    args = parser.parse_args()
    if not (args.runner ^ args.runner_rp ^ args.runner_master):
        raise Exception(
            "Specify exactly one runner type. See main.py -h for help.")
    with MrubisStarter():
        sleep(2)
        wandb.init(project="mrubis_test", entity="mrubis",
                   mode="online" if args.wandb else "disabled")

        episodes = 400
        num_shops = 10
        env = MrubisEnv(
            episodes=episodes,
            negative_reward=-1,
            propagation_probability=0.5,
            shops=num_shops,
            injection_mean=5,
            injection_variance=2,
            trace="",
            trace_length=0,
            send_root_issue=True,
            reward_variance=0)
        # shops = {f'mRUBiS #{i+1}' for i in range(num_shops)}
        shops = [['mRUBiS #1', 'mRUBiS #2', 'mRUBiS #3', 'mRUBiS #4', 'mRUBiS #5',
                  'mRUBiS #6', 'mRUBiS #7', 'mRUBiS #8', 'mRUBiS #9', 'mRUBiS #10']]
        actor = LinearActor()
        critic = LinearConcatCritic()
        #agent_controller = ShopAgentController(actor, critic, shops)
        agents = [Agent(shop_set, actor.clone(), critic.clone())
                  for shop_set in shops]
        if args.runner_rp:
            ReplayBufferRunner(env, agents).run(episodes)
        elif args.runner:
            pass
            #Runner(env, agent_controller).run(episodes)
        elif args.runner_master:
            MasterBaselineRunner(env).run(episodes)
        else:
            raise Exception("Specify a runner type. See main.py -h for help.")


if __name__ == '__main__':
    main()

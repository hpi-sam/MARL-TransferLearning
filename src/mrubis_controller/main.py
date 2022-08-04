import os
from time import sleep

import wandb
from marl.mrubis_env import MrubisEnv
from marl.runner import ReplayBufferRunner, Runner
from marl.master_project.master_baseline_runner import MasterBaselineRunner
from marl.shop_agent_controller import NewActorCritic

from marl.options import args

class MrubisStarter:
    def __init__(self):
        pass

    def __enter__(self):
        print("Starting mRUBiS")
        os.system(
            f"tmux new-session -d -s mrubis{int(os.getenv('MRUBIS_PORT', 8080))} -n mrubis 'MRUBIS_PORT={int(os.getenv('MRUBIS_PORT', 8080))}  CWD=$(pwd) && cd ../mRUBiS/ML_based_Control/ && java -jar mRUBiS.jar > ${{CWD}}/mrubis.log'")

    def __exit__(self, exc_type, exc_value, traceback):
        print("Closing mRUBiS")
        os.system(f"tmux kill-session -t mrubis{int(os.getenv('MRUBIS_PORT', 8080))}")


def main():
    if not (args.runner ^ args.runner_rp ^ args.runner_master):
        raise Exception(
            "Specify exactly one runner type. See main.py -h for help.")
    with MrubisStarter():
        sleep(60)
        wandb.init(project="sweep", entity="mrubis",
                   mode="online" if args.wandb else "disabled", config=args)

        episodes = args.episodes
        num_shops = 10
        env = MrubisEnv(
            episodes=episodes,
            negative_reward=0,
            propagation_probability=0.5,
            shops=num_shops,
            injection_mean=5,
            injection_variance=2,
            trace="",
            trace_length=0,
            send_root_issue=True,
            alternating_trace=False,
            constricted=True,
            reward_variance=0)
        shops = {f'mRUBiS #{i+1}' for i in range(num_shops)}
        agent_controller = NewActorCritic(shops)
        if args.runner_rp:
            ReplayBufferRunner(env, agent_controller).run(episodes)
        elif args.runner:
            Runner(env, agent_controller).run(episodes)
        elif args.runner_master:
            MasterBaselineRunner(env).run(episodes)
        else:
            raise Exception("Specify a runner type. See main.py -h for help.")


if __name__ == '__main__':
    main()

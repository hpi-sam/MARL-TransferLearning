import argparse
from marl.mrubis_env import MrubisEnv
from marl.runner import Runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runner", action="store_true", default=False, help="start rl runner with dl")
    args = parser.parse_args()
    if args.runner:
        episodes = 400
        # mock_env = MrubisMockEnv(number_of_shops=5, shop_config=[1, 0, False])
        env = MrubisEnv(
            episodes=episodes,
            negative_reward=-1,
            propagation_probability=0.5,
            shops=20,
            injection_mean=5,
            injection_variance=2,
            trace="",
            root_causes="",
            trace_length=0)
        shop_distribution_example = [{'mRUBiS #1'}]
        load_model = {0: None, 1: None, 2: None, 3: None, 4: None, 5: None}
        # load_model = {0: {'start_time': 'test_robustness', 'episode': 500}, 1: None}
        Runner(None, env, shop_distribution_example, save_model=True, load_models_data=load_model,
            robustness_activated=False).run(episodes, train=True)
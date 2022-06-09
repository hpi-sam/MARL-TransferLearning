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
            shops=10,
            injection_mean=5,
            injection_variance=2,
            trace="",
            trace_length=0,
            send_root_issue=True,
            reward_variance=5)
        shop_distribution_example = [{'mRUBiS #1', 'mRUBiS #2', 'mRUBiS #3', 'mRUBiS #4', 'mRUBiS #5',
                                    'mRUBiS #6', 'mRUBiS #7', 'mRUBiS #8', 'mRUBiS #9', 'mRUBiS #10'}]
        load_model = {0: None, 1: None, 2: None, 3: None, 4: None,
                    5: None, 6: None, 7: None, 8: None, 9: None,
                    10: None, 11: None, 12: None, 13: None, 14: None,
                    15: None, 16: None, 17: None, 18: None, 19: None,
                    20: None, 21: None, 22: None, 23: None, 24: None,
                    25: None, 26: None, 27: None, 28: None, 29: None,
                    30: None, 31: None, 32: None, 33: None, 34: None,
                    35: None, 36: None, 37: None, 38: None, 39: None,
                    40: None, 41: None, 42: None, 43: None, 44: None,
                    45: None, 46: None, 47: None, 48: None, 49: None,
                    50: None, 51: None, 52: None, 53: None, 54: None,
                    55: None, 56: None, 57: None, 58: None, 59: None,
                    60: None, 61: None, 62: None, 63: None, 64: None,
                    65: None, 66: None, 67: None, 68: None, 69: None,
                    70: None, 71: None, 72: None, 73: None, 74: None,
                    75: None, 76: None, 77: None, 78: None, 79: None}
        # load_model = {0: {'start_time': 'trace_experiments_length_5', 'episode': 300}}
        Runner(None, env, shop_distribution_example, save_model=True, load_models_data=load_model,
            robustness_activated=False, training_activated=True).run(episodes)


if __name__ == '__main__':
    main()
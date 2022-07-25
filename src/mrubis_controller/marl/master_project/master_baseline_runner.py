import wandb
# from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from marl.master_project.multi_agent_controller import MultiAgentController


class MasterBaselineRunner:
    def __init__(self, env, save_model=False, robustness_activated=False, training_activated=True):
        self.load_models_data = {0: None, 1: None, 2: None, 3: None, 4: None,
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
        # self.shop_distribution =
        self.env = env

        self.mac = MultiAgentController(
            self.load_models_data, robustness_activated)
        self.episode = 0
        self.step = 0
        self.save_model = save_model
        self.training_activated = training_activated

    def reset(self):
        """ reset all variables and init env """
        self.episode = 0
        self.env.reset()
        self.mac.init(self.env.action_space, self.training_activated)

    def close_env(self):
        self.env.close()

    def run(self, episodes):
        """ runs the simulation """
        rewards = []
        metrics = []
        logs = []
        self.reset()
        # wandb.init(project="mrubis_test", entity="mrubis",
        #           mode="online")
        while self.episode < episodes:
            terminated = False
            observations = self.env.reset()
            logs.append([])
            wandb.log({'episode': self.episode},
                      commit=False, step=self.step)

            while not terminated:
                actions, regret, root_cause = self.mac.select_actions(observations)
                reward, observations_, terminated, env_info = self.env.step(actions)
                if actions is not None:
                    rewards.append(reward)
                    # old: self.mac.learn(observations, actions, reward, observations_, terminated)
                    self.agent_controller.add_to_replay_buffer(observations, actions, reward, observations_, terminated)
                    res = self.agent_controller.learn(10)
                    metrics.append(res)
                    for shop in regret[0].keys():
                        wandb.log(
                            {f'Regret_{shop}': regret[0][shop]}, step=self.step)
                    for shop in reward:
                        wandb.log(
                            {f"Reward_{shop}": reward[shop]}, step=self.step)
                observations = observations_
                self.step += 1

                if terminated:
                    for shop, count in env_info['stats'].items():
                        print(f"Fixed_{shop}: {count}")
                        if count != -1:
                            wandb.log({f"Fixed_{shop}": count},
                                      step=self.step)
            self.episode += 1
            print(f"episode {self.episode} done")

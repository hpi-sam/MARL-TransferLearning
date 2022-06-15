import wandb
from entities.observation import SystemObservation
from marl.mrubis_env import MrubisEnv
from marl.shop_agent_controller import ShopAgentController

class Runner:
    def __init__(self, env: MrubisEnv, agent_controller: ShopAgentController):
        self.env = env
        self.agent_controller = agent_controller # TODO create agent controller

    def reset(self):
        """ reset all variables and init env """
        self.env.reset()
        # TODO reset agent controller

    def close_env(self):
        self.env.close()

    def run(self, episodes):
        """ runs the simulation """
        self.reset()
        step = 0
        for episode in range(episodes):
            print("Episode: {}".format(episode))
            wandb.log({'episode': episode}, commit=False, step=step)
            terminated = False
            current_observations = SystemObservation.from_dict(self.env.reset())
            while not terminated:
                step += 1
                actions = self.agent_controller.choose_actions(current_observations)
                sendable_actions = list(map(lambda x: x.action, actions.values()))
                reward, next_observations, terminated, env_info = self.env.step(sendable_actions)
                self.agent_controller.learn(actions, reward)
                current_observations = next_observations
                if terminated:
                    for shop, count in env_info['stats'].items():
                        print(f"{shop} fixed after: {count}")
                        wandb.log({f"{shop}_fixed": count}, step=step)
            print(f"episode {episode} done")

if __name__ == "__main__":
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
    shops = {'mRUBiS #1', 'mRUBiS #2', 'mRUBiS #3', 'mRUBiS #4', 'mRUBiS #5', 'mRUBiS #6', 'mRUBiS #7', 'mRUBiS #8', 'mRUBiS #9', 'mRUBiS #10'}
    # load_model = {0: {'start_time': 'trace_experiments_length_5', 'episode': 300}}
    Runner(env).run(episodes)

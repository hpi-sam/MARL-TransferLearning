import wandb
from entities.components import Components
from marl.mrubis_data_helper import has_shop_remaining_issues
from marl.mrubis_env import MrubisEnv
from marl.shop_agent_controller import ShopAgentController
import alive_progress as ap
from marl.options import args
import plotly.express as px

from marl.utils import distance_rp_buffers
import math


class Runner:
    def __init__(self, env: MrubisEnv, agent_controller: ShopAgentController):
        self.env = env
        self.agent_controller = agent_controller  # TODO create agent controller

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
        with ap.alive_bar(title="Running") as bar:
            for episode in range(episodes):
                wandb.log({'episode': episode}, commit=False, step=step)
                terminated = False
                current_observations = self.env.reset()
                while not terminated:
                    bar()
                    bar.text = f"Episode: {episode} Step: {step}"
                    step += 1
                    actions = self.agent_controller.choose_actions(
                        current_observations)
                    # for action in actions.values():
                    #     print(action.action_tensor.max().item(), action.action_tensor.argmax().item(), end=' ')
                    # print()
                    # exit()
                    valid_actions = filter(lambda x: has_shop_remaining_issues(
                        current_observations, x.action.shop), list(actions.values()))
                    sendable_actions = list(
                        map(lambda x: x.action.to_sendable_json(), valid_actions))
                    sendable_actions = {i: e for i,
                                        e in enumerate(sendable_actions)}
                    reward, next_observations, terminated, env_info = self.env.step(
                        sendable_actions)
                    self.agent_controller.learn(
                        actions, reward, next_observations)
                    current_observations = next_observations
                    if terminated:
                        for shop, count in env_info['stats'].items():
                            print(f"{shop} fixed after: {count}")
                            if count != -1:
                                wandb.log({f"{shop}_fixed": count}, step=step)
                        self.env.reset()
                print(f"episode {episode} done")


class ReplayBufferRunner:
    def __init__(self, env: MrubisEnv, agent_controller: ShopAgentController):
        self.env = env
        self.agent_controller = agent_controller  # TODO create agent controller

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
        with ap.alive_bar(title="Running") as bar:
            for episode in range(episodes):
                wandb.log({'episode': episode}, commit=False, step=step)
                terminated = False
                current_observations = self.env.reset()
                self.agent_controller.reset_shield()
                self.agent_controller.handle_episode_observation(current_observations)
                for shop_name, buffer in self.agent_controller.replay_buffers.items():
                    wandb.log({f'{shop_name} Conditional Probabilities': px.imshow(buffer.get_dist_probs()[0], text_auto=True, x=Components.value_list(), y=Components.value_list())}, step=step)
                distances, buffer_names = distance_rp_buffers(self.agent_controller.replay_buffers)
                wandb.log({f'Replay Buffer Distances': px.imshow(distances / math.sqrt(18**2), text_auto=True, x=buffer_names, y=buffer_names)}, step=step)
                while not terminated:
                    bar()
                    bar.text = f"Episode: {episode} Step: {step}"
                    step += 1
                    actions, action_probs = self.agent_controller.choose_actions(current_observations)
                    print(action_probs)
                    valid_actions = filter(lambda x: has_shop_remaining_issues(current_observations, x.action.shop), list(actions.values()))
                    sendable_actions = list(map(lambda x: x.action.to_sendable_json(), valid_actions))
                    sendable_actions = {i: e for i, e in enumerate(sendable_actions)}
                    reward, next_observations, terminated, env_info = self.env.step(sendable_actions)
                    for shop in action_probs.keys():
                        wandb.log({
                            f"{shop}_reward": reward[shop],
                            f"{shop}_action_p": action_probs[shop],
                        }, step=step)
                    # self.agent_controller.learn(actions, reward, next_observations)
                    self.agent_controller.add_to_replaybuffer(actions, reward, next_observations)
                    
                    current_observations = next_observations
                    if terminated:
                        for shop, count in env_info['stats'].items():
                            print(f"{shop} fixed after: {count}")
                            self.agent_controller.reset_shield()
                            if count != -1:
                                wandb.log({f"{shop}_fixed": count}, step=step)
                self.agent_controller.learn_from_replaybuffer(args.batch_size)
                print(f"episode {episode} done")

import math
from typing import List, Tuple
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import wandb
from entities.components import Components
from marl.master_project.agent import Agent
# from mrubis_controller.marl.mrubis_mock_env import MrubisMockEnv
from marl.master_project.multi_agent_controller import MultiAgentController
from marl.options import args
import plotly.express as px

from marl.utils import distance_rp_buffers


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

    def get_groups(self, agents: List[Agent]) -> Tuple[List[Agent]]:
        print("agents", agents)
        sorted_agents = sorted(agents, key=lambda agent: list(agent.shops)[0])
        print("sorted_agents", sorted_agents)
        return sorted_agents[:int(len(agents) / 2)], sorted_agents[int(len(agents) / 2):]

    def transfer_knowledge_pairwise(self, pairs: List[Tuple[Agent]]):
        if args.transfer_strategy == "off":
            return
        for pair in pairs:
            print("======== TRANSFERING KNOWLEDGE =========")
            if args.transfer_strategy == "replace":
                self.transfer_knowledge_replace(*pair)
                self.knowledge_retrain(pair[1])
            elif args.transfer_strategy == "combine":
                self.transfer_knowledge_combine(*pair)
                self.knowledge_retrain(pair[1])
            elif args.transfer_strategy == "knowledge_destillation":
                self.transfer_knowledge_combine(*pair)
                self.knowledge_destillation(*pair)
        
    def transfer_knowledge_replace(self, source: Agent, target: Agent):
        source_replay_buffer = next(iter(source.replay_buffers.values()))
        target_replay_buffer = next(iter(target.replay_buffers.values()))
        target_replay_buffer.set_state(*source_replay_buffer.get_state())
    
    def transfer_knowledge_combine(self, source: Agent, target: Agent):
        source_replay_buffer = next(iter(source.replay_buffers.values()))
        target_replay_buffer = next(iter(target.replay_buffers.values()))
        source_state = source_replay_buffer.get_state()
        target_state = target_replay_buffer.get_state()
        combined_state = [target_state[i] + source_state[i] for i in range(len(source_state))]
        target_replay_buffer.set_state(*combined_state)
    
    def knowledge_retrain(self, agent: Agent):
        for _ in range(args.n_retrain):
            if args.retrain_cat:
                agent.retrain_off_policy_cat(args.batch_size)
            else:
                agent.retrain_off_policy(args.batch_size)

    def knowledge_destillation(self, source: Agent, target: Agent):
        source_replay_buffer = next(iter(source.replay_buffers.values()))
        observations = source_replay_buffer.get_state()[0]
        dataloader = DataLoader(observations, batch_size=args.batch_size)
        for _ in range(args.n_retrain):
            for observations in dataloader:
                with torch.no_grad():
                    source_output = source.model(observations)[0]
                target.model.actor_optimizer.zero_grad()
                target_output = target.model(observations)[0]
                F.cross_entropy(target_output, source_output).backward()
                target.model.actor_optimizer.step()
    def run(self, episodes):
        """ runs the simulation """
        rewards = []
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
            all_replay_buffers = {}
            for agent in self.mac.agents:
                all_replay_buffers = {**all_replay_buffers, **agent.replay_buffers}
                for shop_name, buffer in agent.replay_buffers.items():
                    wandb.log({f'{shop_name} Conditional Probabilities': px.imshow(buffer.get_dist_probs()[0].numpy(), text_auto=True, x=Components.value_list(), y=Components.value_list())}, step=self.step)
            distances, buffer_names = distance_rp_buffers(all_replay_buffers)
            wandb.log({f'Replay Buffer Distances': px.imshow((distances / math.sqrt(18**2)).numpy(), text_auto=True, x=buffer_names, y=buffer_names)}, step=self.step)
            if self.episode == math.ceil(episodes / 2):
                group1, group2 = self.get_groups(self.mac.agents)
                print("group1: ", group1)
                print("group2: ", group2)
                # For every element in group1 find partner in group2
                pairs = zip(group1, group2)
                print("pairs: ", pairs)
                # Transfer knowledge between partners
                self.transfer_knowledge_pairwise(pairs)
            while not terminated:
                actions, regret, root_cause, probabilities = self.mac.select_actions(observations)
                print("Actions:")
                print(actions)
                reward, observations_, terminated, env_info = self.env.step(actions)
                if actions is not None:
                    rewards.append(reward)
                    #print("reward:", reward)
                    self.mac.add_to_replay_buffer(observations, probabilities , actions, reward, observations_, terminated)
                    if args.on_policy:
                        self.mac.learn_on_policy(observations, actions, reward, observations_, terminated)
                    elif args.on_off_policy:
                        self.mac.learn_on_off_policy(observations, actions, reward, observations_, terminated, args.batch_size)
                    else:
                        self.mac.learn_off_policy(args.batch_size)
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
            if args.use_exploration:
                self.mac.reset_sampled_actions_mem()
            self.episode += 1
            print(f"episode {self.episode} done")

from marl.master_project.agent import encode_observations
from typing import Dict, Union
from entities.component_failure import ComponentFailure
from entities.observation import ComponentObservation, ShopObservation, SystemObservation
from entities.shop import Shop


def build_observations(agents, agent_index, observation, shops=None):
    """ extracts the relevant observations of the env per agent """
    relevant_shops = agents[agent_index].shops if shops is None else shops
    return {shop: observation.shops[shop] for shop in observation.shops.keys()}


def build_rewards(agents, agent_index, rewards):
    """ extracts the relevant rewards per agent
        probably not needed as mRubis is returning reward per shop already
    """
    # eigentlich der zweite Parameter: rewards[1] (was eine einzelne Zahl ist, aber das wird nirgends benutzt)
    return {shop: rewards[shop] for shop in agents[agent_index].shops}, 1


def build_actions(agents, agent_index, actions):
    """ extracts the relevant actions taken per agent
    """
    return {index: action
            for shop in agents[agent_index].shops
            for index, action in enumerate(actions.values())
            if action['shop'] == shop}


def build_replay_buffer(agents, index, replay_buffer):
    """ extracts only the latest history """
    for history in replay_buffer[-1:]:
        return {
            'observations': build_observations(agents, index, history['observations']),
            'actions': build_actions(agents, index, history['actions']),
            'rewards': build_rewards(agents, index, history['rewards']),
            'observations_': build_observations(agents, index, history['observations_']),
            'dones': history['dones']
        }


def get_failing_component(current_shop):
    """ returns the current failing component or None if no failure available """
    for component in current_shop:
        if current_shop[component]['failure_name'] != "None":
            return component
    return None


def get_current_utility(observation):
    return {shop: float(list(components.items())[0][1]['shop_utility']) for shop, components in
            observation.items()}


def has_shop_remaining_issues(observations, shop):
    """ checks whether a shop as remaining issues """
    one_hot = encode_observations(observations[shop])
    return sum(one_hot) > 0


def get_failing_component(current_shop: ShopObservation):
    """ returns the current failing component or None if no failure available """
    for component in current_shop.components.values():
        if component.failure_name != ComponentFailure.NONE:
            return component
    return None

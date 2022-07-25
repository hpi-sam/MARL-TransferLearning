import numpy as np


def prepare_step(inputs):
    return dict(enumerate(inputs))


def _rank_actions(inputs):
    filtered_inputs = [i for i in inputs if i]  # remove empty arrays
    if len(filtered_inputs) == 0:
        return None
    sorted_inputs = sorted(np.hstack(filtered_inputs), key=lambda d: d['predicted_utility'], reverse=True)
    return prepare_step(sorted_inputs)


class RankLearner:
    def __init__(self, stage, args):
        # stage 0 = no sorting as a baseline
        # stage 1 = sorting of agents
        self.stage = stage
        self.args = args

    def _build_network(self):
        """ build the ranking network """
        raise NotImplementedError

    def sort_actions(self, inputs):
        """ Sorts the actions of each agents """
        if self.stage == 0:
            return prepare_step(np.array(inputs).flatten())
        else:
            return _rank_actions(inputs)

    def learn(self):
        """ learn to rank """
        raise NotImplementedError

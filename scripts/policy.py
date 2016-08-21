
import numpy as np

class EpsilonGreedy(object):

    def __init__(self, opts):
        self.opts = opts
        self.action_idxs = range(opts.num_actions)

    def random_action_index(self):
        return np.random.choice(self.action_idxs)

    def choose_action_index(self, q_values):
        exploration_prob = self.opts.exploration_prob
        if self.opts.in_validation:
            exploration_prob = self.opts.validation_exploration_prob

        # e-greedy action selection
        if np.random.random() < exploration_prob :
            action_idx = np.random.choice(self.action_idxs)
        else:
            action_idx = np.argmax(q_values)

        # update parameter values
        self.update_parameters()
        return action_idx

    def update_parameters(self):
        updated_exploration_prob = self.opts.exploration_prob \
                                        - self.opts.exploration_reduction
        self.opts.exploration_prob = max(self.opts.min_exploration_prob, \
                                        updated_exploration_prob)
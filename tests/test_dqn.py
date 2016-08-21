
import numpy as np
import os
import sys

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import qnetwork
import replay_memory

class TestOptions(object):

    def __init__(self):
        self.state_dim = 7
        self.num_actions = 5
        self.random_seed = 1
        self.rng = np.random.RandomState(self.random_seed)
        self.in_validation = False
        self.batch_size = 16
        self.num_hidden = 256
        self.num_hidden_layers = 2
        self.nonlinearity = 'leaky_relu'
        self.freeze_interval = 10
        self.regularization = 0.0
        self.learning_rate = 0.0001
        self.min_learning_rate = 0.0001
        self.decay_lr_every = 20000
        self.decay_lr_ratio = 0.97
        self.discount = 0.95
        self.max_norm = 5
        self.max_quadratic_loss = 50.0
        self.hidden_layer_sizes = []
        self.dropout_prob = 0.0
        self.increase_batch_size = False
        self.use_skip_connection = False
        self.dropout_prob = 0.0

def test_overfit_simple_artificial_dataset():
        opts = TestOptions()
        opts.state_dim = 1
        opts.batch_size = 2 ** 6
        opts.replay_capacity = 2 ** 16
        opts.num_hidden = 128
        opts.num_hidden_layers = 2
        opts.discount = 0
        opts.num_actions = 2
        opts.regularization = 1e-5
        opts.dropout = 0.5

        network = qnetwork.QNetwork(opts)
        rm = replay_memory.ReplayMemory(opts)

        num_samples_each = 2 ** 14
        state = np.array([0], dtype=np.float32)
        next_state = np.array([0], dtype=np.float32)
        terminal = 1
        for idx in range(num_samples_each):
            action = 0
            reward = np.random.choice([0,5])
            rm.store((state, action, reward, next_state, terminal))
            action = 1
            reward = np.random.choice([1,3])
            rm.store((state, action, reward, next_state, terminal))

        counter = 0
        losses = []
        max_iterations = 10000
        Q = {}
        for _ in range(max_iterations):
            counter += 1
            states, actions, rewards, next_states, terminals = rm.sample_batch()
            loss = network.train(states, actions, rewards, next_states, terminals)
            losses.append(loss)

            if counter % 100 == 0:
                q_values = network.get_q_values(np.array([0]))
                print q_values


test_overfit_simple_artificial_dataset()
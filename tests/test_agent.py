
import gym
import numpy as np
import os
import sys
import time
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import agent
import default_options
import experiment
import logger
import policy
import qnetwork
import recurrent_qnetwork
import replay_memory
import run_experiment
import state_adapter

# high, low, state_dim, num_actions
MDP_CONST_MAP = {'OneRoundDeterministicReward-v0': (np.array([0]), np.array([0]), 1, 2),
                'TwoRoundDeterministicReward-v0': (np.array([0]), np.array([2]), 1, 2),
                'OneRoundNondeterministicReward-v0': (np.array([0]), np.array([0]), 1, 2),
                'TwoRoundDeterministicReward-v0': (np.array([0]), np.array([2]), 1, 2)}

def get_options():
    opts = default_options.DefaultOptions()
    opts.log_weights_every = 1e50
    opts.log_stats_every = 1e50
    opts.log_directory = '../../../../data/snapshots/test_run/'
    opts.options_filepath = os.path.join(opts.log_directory, 'options.pkl')
    opts.weights_filepath = os.path.join(opts.log_directory, 'weights_{}.npz')
    opts.replay_capacity = 2000
    opts.validation_every = 100000
    opts.batch_size = 512
    opts.learning_rate = 0.01
    opts.min_learning_rate = 0.0001
    opts.decay_lr_every = 50
    opts.freeze_interval = 200
    opts.num_hidden = 64
    opts.num_hidden_layers = 2
    opts.dropout_prob = 0.0 
    opts.regularization = 0.0
    opts.print_every = 25
    opts.sequence_length = 1
    opts.subnetwork_type = 'single_layer_lstm'
    opts.rnn_grad_clip = 2
    return opts

class TestSequenceAgent(unittest.TestCase):

    def test_sequence_agent_on_basic_non_sequential_mdp(self):
        opts = get_options()
        opts.sequence_length = 1
        opts.env_name = 'TwoRoundDeterministicReward-v0'

        # environment
        env = gym.make(opts.env_name)
        env.monitor.start(opts.env_directory, force=True, seed=0)
        (opts.high, opts.low, opts.state_dim, 
            opts.num_actions) = MDP_CONST_MAP[opts.env_name]

        # agent
        qnet = qnetwork.SequenceQNetwork(opts)
        p = policy.EpsilonGreedy(opts)
        rm = replay_memory.SequenceReplayMemory(opts)
        sa = state_adapter.RangeAdapter(opts)
        # sa = state_adapter.IdentityAdapter(opts)
        l = logger.Logger(opts)
        a = agent.SequenceAgent(
                network=qnet, policy=p, replay_memory=rm, 
                state_adapter=sa, log=l, opts=opts)

        # experiment
        exp = experiment.Experiment(env, a, opts)
    
        # run it
        exp.run()

    def test_sequence_agent_on_advanced_mdp(self):
        opts = get_options()
        opts.replay_capacity = 10000
        opts.log_weights_every = 100
        opts.log_stats_every = 1e50
        opts.validation_every = 50
        opts.batch_size = 32
        opts.learning_rate = 0.01
        opts.min_learning_rate = 0.00005
        opts.decay_lr_every = 1000
        opts.freeze_interval = 10
        opts.num_hidden = 128
        opts.num_hidden_layers = 3
        opts.discount = .999
        opts.dropout_prob = 0.0
        opts.regularization = 0.0
        opts.train_updates_per_episode = 1
        opts.exploration_prob = 0.5
        opts.min_exploration_prob = 0.05
        opts.max_quadratic_loss = 50.0 
        opts.exploration_reduction = 1e-7
        opts.env_name = 'CartPole-v0' # 'LunarLander-v2'
        opts.sequence_length = 4

        # environment
        env = gym.make(opts.env_name)
        env.monitor.start(opts.env_directory, force=True, seed=0)

        # set const values 
        opts.state_dim = env.observation_space.shape[0]
        opts.num_actions = env.action_space.n

        # state range information
        opts.high = env.observation_space.high
        opts.low = env.observation_space.low

        # agent
        qnet = qnetwork.SequenceQNetwork(opts)
        p = policy.EpsilonGreedy(opts)
        rm = replay_memory.SequenceReplayMemory(opts)
        # sa = state_adapter.RangeAdapter(opts)
        sa = state_adapter.IdentityAdapter(opts)
        l = logger.Logger(opts)
        a = agent.SequenceAgent(
                network=qnet, policy=p, replay_memory=rm, 
                state_adapter=sa, log=l, opts=opts)

        # experiment
        exp = experiment.Experiment(env, a, opts)
    
        # run it
        exp.run()

class TestSequenceAgentWithRecurrentNetwork(unittest.TestCase):

    def test_recurrent_q_net(self):
        opts = get_options()
        opts.replay_capacity = 10000
        opts.log_weights_every = 100
        opts.log_stats_every = 1e50
        opts.validation_every = 50
        opts.batch_size = 32
        opts.learning_rate = 0.01
        opts.min_learning_rate = 0.00005
        opts.decay_lr_every = 500
        opts.freeze_interval = 10
        opts.num_hidden = 128
        opts.discount = .999
        opts.dropout_prob = 0.0
        opts.regularization = 1e-5
        opts.train_updates_per_episode = 1
        opts.exploration_prob = 0.3
        opts.min_exploration_prob = 0.05
        opts.max_quadratic_loss = 50.0 
        opts.exploration_reduction = 1e-6
        opts.subnetwork_type = 'single_layer_lstm'
        opts.rnn_grad_clip = 10
        opts.env_name = 'CartPole-v0' # 'LunarLander-v2'
        opts.sequence_length = 3

        # environment
        env = gym.make(opts.env_name)
        env.monitor.start(opts.env_directory, force=True, seed=0)

        # set const values 
        opts.state_dim = env.observation_space.shape[0]
        opts.num_actions = env.action_space.n

        # state range information
        opts.high = env.observation_space.high
        opts.low = env.observation_space.low

        # agent
        print 'building agent...'
        qnet = recurrent_qnetwork.RecurrentQNetwork(opts)
        p = policy.EpsilonGreedy(opts)
        rm = replay_memory.SequenceReplayMemory(opts)
        # sa = state_adapter.RangeAdapter(opts)
        sa = state_adapter.IdentityAdapter(opts)
        l = logger.Logger(opts)
        a = agent.SequenceAgent(
                network=qnet, policy=p, replay_memory=rm, 
                state_adapter=sa, log=l, opts=opts)

        # experiment
        print 'building experiment...'
        exp = experiment.Experiment(env, a, opts)
        
        # run it
        print 'running experiment...'
        exp.run()

if __name__ == '__main__':
    unittest.main()
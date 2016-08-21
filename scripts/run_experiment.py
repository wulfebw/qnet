
import gym
import numpy as np
import os
import sys

import agent
import default_options
import experiment
import logger
import policy
import qnetwork
import replay_memory
import state_adapter

def get_env(opts):
    print 'building environment...'
    env = gym.make(opts.env_name)
    env.monitor.start(opts.env_directory, force=True, seed=0)
    # after creating the environment, transfer information
    # about that environment to the options object that 
    # is used in creating the other classes in the experiment

    # # add stuff to debugging mdps
    # if 'shape' not in env.observation_space.__dict__:
    #     env.observation_space.shape = (1,)

    if len(env.observation_space.shape) > 1:
        msg = 'multidimensional observation space not implemented'
        raise ValueError(msg)
    opts.state_dim = env.observation_space.shape[0]
    opts.num_actions = env.action_space.n

    # state range information
    opts.high = env.observation_space.high
    opts.low = env.observation_space.low

    return env

def get_agent(opts):
    print 'building agent...'
    qnet = qnetwork.QNetwork(opts)

    if opts.load_weights_filepath != '':
        if not os.path.exists(opts.load_weights_filepath):
            raise ValueError('weights filepath invalid: {}'.format(
                opts.load_weights_filepath))
        print 'loading weights from {}'.format(opts.load_weights_filepath)
        qnet.load_weights_from_file(opts.load_weights_filepath)

    # policy and replay memory
    p = policy.EpsilonGreedy(opts)
    rm = replay_memory.ReplayMemory(opts)
    
    # state adapter
    sa = state_adapter.RangeAdapter(opts)
    #sa = state_adapter.IdentityAdapter()
    # if loading weights, load in the means and ranges
    if opts.load_weights_filepath != '':
        d = np.load(opts.load_weights_filepath)
        sa.state_means = d['state_means']
        sa.state_stds = d['state_stds']

    l = logger.Logger(opts)
    a = agent.Agent(network=qnet, policy=p, replay_memory=rm, 
            state_adapter=sa, log=l, opts=opts)
    return a

def run():
    opts = default_options.parse_args(sys.argv[1:])
    env = get_env(opts)
    ag = get_agent(opts)
    print 'building experiment...'
    exp = experiment.Experiment(env, ag, opts)
    print 'running experiment...'
    exp.run()

if __name__ == '__main__':
    run()

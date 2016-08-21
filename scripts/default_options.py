
import argparse
import numpy as np
import os
import sys

class DefaultOptions(object):

    def __init__(self):

        # env
        # 'TwoRoundNondeterministicReward-v0' 'OneRoundNondeterministicReward-v0' 'TwoRoundDeterministicReward-v0' 'OneRoundDeterministicReward-v0' 'MountainCar-v0' 'CartPole-v0' 'LunarLander-v2'
        self.env_name = 'LunarLander-v2'
        self.env_directory = '/Users/wulfebw/Desktop/gym/'
        # environment is used to set the variables listed
        # below this comment. The reason for this is that 
        # this implementation is an adaption of another 
        # one that does not interact with gym. These values
        # must be set after creating the environment, which
        # is done in the file run_experiment.py in the 
        # function get_env
        self.state_dim = None
        self.num_actions = None
        self.high = None
        self.low = None

        # network
        self.framework = 'theano'
        self.batch_size = 2 ** 8
        # next two only used if no specific layer sizes
        self.num_hidden = 128
        self.num_hidden_layers = 3
        # hidden_layer_sizes used if it exists else, 
        # num_hidden_layers and num_hidden
        self.hidden_layer_sizes = [128,128] 
        # {relu, tanh, leaky_relu}
        self.nonlinearity = 'leaky_relu' 
        # time to train for each episode run
        self.train_updates_per_episode = 1
        # how many training updates between target network update
        self.freeze_interval = 10
        # l2 weight reg
        self.regularization = 1e-6
        self.learning_rate = 0.01
        # mininum lr allowed
        self.min_learning_rate = 0.00001
        # number of training updates between lr decay
        self.decay_lr_every = 1000
        # exponential lr decay
        self.decay_lr_ratio = 0.9
        # dropout prob = prob of dropping _not_ keeping 
        self.dropout_prob = 0.2
        self.discount = 1.0
        # max norm used with tensorflow network
        self.max_norm = 5
        # max quadratic used with theano network
        self.max_quadratic_loss = 50.0 
        # filepath from which to load network params
        self.load_weights_filepath = ''
        # tensorflow only
        self.use_multi_gpu = False
        self.gpu_ids = [1,2,3]
        
        # policy
        # initil exploration probability
        self.exploration_prob = 1.0
        # minimum allowed exploration probability
        self.min_exploration_prob = 0.1
        # exploration prob to use in validation
        self.validation_exploration_prob = 0.0
        # amount exploration is reduced every time 
        # _an action is selected_
        # so it's based on actions not training updates
        self.exploration_reduction = 5e-7

        # replay memory
        self.replay_capacity = 1e5

        # logger
        self.log_steps_back = 10000
        # epochs between logging weights
        self.log_weights_every = 1e50
        # epochs between logging stats
        self.log_stats_every = 1e50
        # where to log stats
        self.log_directory = '../../../../data/snapshots/test_run/'
        # actions take up a lot of space so limit number tracked
        self.max_actions_tracked = 1e6
        self.verbose = True
        self.print_every = 100
            
        # experiment
        # in_validation flag used to trigger validation
        # settings during experiment
        self.in_validation = False
        # episodes between validation run
        self.validation_every = 10
        # number of episodes to run
        self.num_episodes = 100000
        # max number of steps in an episode
        self.max_steps = 1000

def parse_args(args):
    opts = DefaultOptions()
    parser = argparse.ArgumentParser()

    # general args
    parser.add_argument('-a', '--framework', dest='framework', 
        help='framework to use {theano, tensorflow}',
        type=str, default=opts.framework)
    parser.add_argument('-b', '--batch_size', dest='batch_size', 
        help='batch size to use', 
        type=int, default=opts.batch_size)
    parser.add_argument('-d', '--log_directory', dest='log_directory', 
        help='directory to log training information and network weights',
        type=str, default=opts.log_directory)
    parser.add_argument('-e', '--num_hidden_layers', dest='num_hidden_layers', 
        help='number of hidden layers in the network',
        type=int, default=opts.num_hidden_layers)
    parser.add_argument('-f', '--freeze_interval', dest='freeze_interval', 
        help='network weight updates between target network reset',
        type=int, default=opts.freeze_interval)
    parser.add_argument('-g', '--regularization', dest='regularization', 
        help='regularization weight',
        type=float, default=opts.regularization)
    parser.add_argument('-i', '--max_steps', dest='max_steps', 
        help='maximum steps in a single episode',
        type=int, default=opts.max_steps)
    parser.add_argument('-l', '--learning_rate', dest='learning_rate', 
        help='initial learning rate',
        type=float, default=opts.learning_rate)
    parser.add_argument('-m', '--replay_capacity', dest='replay_capacity', 
        help='capacity of replay memory',
        type=int, default=opts.replay_capacity)
    parser.add_argument('-n', '--nonlinearity', dest='nonlinearity', 
        help='nonlinearity to use in the network, one of \{relu, linear\}',
        type=str, default=opts.nonlinearity)
    parser.add_argument('-o', '--load_weights_filepath', dest='load_weights_filepath', 
        help='filepath from which to load weights',
        type=str, default=opts.load_weights_filepath)
    parser.add_argument('-p', '--dropout_prob', dest='dropout_prob', 
        help='dropout prob for hidden layers',
        type=float, default=opts.dropout_prob)
    parser.add_argument('-u', '--num_hidden', dest='num_hidden', 
        help='number of hidden units in each hidden layer',
        type=int, default=opts.num_hidden)
    parser.add_argument('-x', dest='hidden_layer_sizes', 
        help='list of hidden layer sizes to use',
        nargs='*', default=opts.hidden_layer_sizes)

    parsed_args = parser.parse_args(args)
    for k, v in parsed_args.__dict__.iteritems():
        opts.__dict__[k] = v

    # format filepaths
    opts.weights_filepath = os.path.join(opts.log_directory, 'qnetwork_weights_{}')
    opts.stats_filepath = os.path.join(opts.log_directory, 'stats')
    opts.options_filepath = os.path.join(opts.log_directory, 'options.pkl')

    # reformat the output filepaths and create the directory if necessary
    if not os.path.exists(opts.log_directory):
        try:
            os.mkdir(opts.log_directory)
        except Exception as e:
            print "unable to create logging directory: {}".format(opts.log_directory)
            raise e
            
    # convert hidden layer sizes to ints from strings
    opts.hidden_layer_sizes = [int(v) for v in opts.hidden_layer_sizes]
    
    return opts


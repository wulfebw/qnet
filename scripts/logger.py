
import cPickle
import collections
import copy
import numpy as np
import os
import sys
import time

class Logger(object):

    def __init__(self, opts):
        self.opts = opts
        self.info = collections.defaultdict(lambda: list())
        self.episode = 0
        self.start_time = time.time()
        self.episode_start_time = time.time()
        self.log_options()

    def log_value(self, key, value):
        if self.opts.in_validation:
            key = 'validation_' + key
        self.info[key].append(value)

    def finish_episode(self, network, adapter, replay_memory):
        # increment episodes only on non validation runs
        if not self.opts.in_validation:
            self.episode += 1

        # how long have we been doing this (too long)
        cur_time = time.time()
        episode_time = cur_time - self.episode_start_time
        self.episode_start_time = cur_time
        total_time = cur_time - self.start_time

        # log progress to terminal
        if self.opts.verbose:
            if self.opts.in_validation:
                mean_reward = np.mean(self.info['validation_reward'][-5000:])
                txt = '\rvalidation reward: {:.6f}\n'.format(mean_reward)
            else: 
                # compute mean loss and rewards over number of past steps
                mean_loss = np.mean(self.info['loss'][-self.opts.log_steps_back:])
                mean_reward = np.mean(self.info['reward'][-self.opts.log_steps_back:])
                txt = '\repisode: {}\tloss: {:.6}\treward: {:.6f}\ttime: {:.4f}\ttotal_time: {:.3f}\tlr: {:.5f}\n'.format(self.episode, mean_loss, mean_reward, episode_time, total_time, self.opts.learning_rate)

            # write the message out
            if self.opts.in_validation or self.episode % self.opts.print_every == 0:
                sys.stdout.write(txt)
                sys.stdout.flush()

        # saving weights
        # only write to file if replay memory is full and we are training
        if self.episode % self.opts.log_weights_every == 0 and replay_memory.is_full():
            self.save_weights(network, adapter)

        # saving stats and hyperparams
        if self.episode % self.opts.log_stats_every == 0:
            # convert the dictionary to tuples and save
            np.savez(self.opts.stats_filepath, **self.info)

    def save_weights(self, network, adapter):
        # format filepath for this episode
        filepath = self.opts.weights_filepath.format(self.episode)

        # collect weights 
        weights = network.get_params()

        # collect state means and stds and save with weights
        state_means = adapter.means
        state_ranges = adapter.ranges
        np.savez(filepath, *weights, state_means=state_means, state_ranges=state_ranges)

    def log_options(self):
        """
        Log the options used for this run for later reference.
        """
        outfile = open(self.opts.options_filepath, 'wb')
        cPickle.dump(self.opts, outfile)

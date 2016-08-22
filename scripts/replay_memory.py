"""
There are two replay memory implementations in this file.

The first is a dictionary style memory that makes few
assumptions about the form or consistency of the state. 
It's not efficient, but can be extended for easy 
prioritized sampling.

The second is a buffer style memory that's efficient
and works well with recurrent networks.
"""

import numpy as np
import theano

class ReplayMemory(object):
    """
    Nonsequential replay memory. Good if you want 
    to implement weighted sampling or don't know much
    about the states.
    """

    def __init__(self, opts):
        self.opts = opts
        self.memory = {}
        self.first_index = -1
        self.last_index = -1

        # allocate space for batch a single time
        self.states = np.empty((self.opts.batch_size, 
                        self.opts.state_dim), dtype=theano.config.floatX)
        self.action_idxs = np.empty((self.opts.batch_size, 1), 
                            dtype=np.int32)
        self.rewards = np.empty((self.opts.batch_size, 1), 
                            dtype=theano.config.floatX)
        self.next_states = np.empty((self.opts.batch_size, 
                            self.opts.state_dim), dtype=theano.config.floatX)
        self.terminals = np.empty((self.opts.batch_size, 1), 
                            dtype=np.int32)

    def store(self, sample):
        if self.first_index == -1:
            self.first_index = 0
        self.last_index += 1
        self.memory[self.last_index] = sample 
        if (self.last_index - self.first_index + 1) > self.opts.replay_capacity:
            self.discard_sample()

    def discard_sample(self):
        del self.memory[self.first_index]
        self.first_index += 1

    def is_full(self):
        return (self.last_index - self.first_index + 1) >= self.opts.replay_capacity

    def is_empty(self):
        return self.first_index == -1

    def sample_batch(self):
        # get random indices to select
        rand_idxs = np.random.randint(self.first_index, 
                    self.last_index + 1, size=self.opts.batch_size)

        # sample batch
        for idx, rand_idx in enumerate(rand_idxs):
            state, action_idx, reward, next_state, terminal = self.memory[rand_idx]
            self.states[idx] = state
            self.action_idxs[idx] = action_idx
            self.rewards[idx] = reward
            self.next_states[idx] = next_state
            self.terminals[idx] = terminal

        return self.states, self.action_idxs, self.rewards, \
                    self.next_states, self.terminals

class SequenceReplayMemory(object):
    """
    This is from https://github.com/spragunr/deep_q_rl
    Sequential replay memory. Good if you want to concat 
    frames for multi-frame states or recurrent networks.
    """
    
    def __init__(self, opts):
        self.opts = opts
        self.bottom = 0
        self.top = 0
        self.size = 0

        # unpack some values for easier reference
        batch_size = opts.batch_size
        state_dim = opts.state_dim
        capacity = opts.replay_capacity
        sequence_length = opts.sequence_length

        # allocate buffers
        self.states = np.zeros(((capacity,) + (state_dim,)), 
                        dtype='float32')
        self.actions = np.zeros(capacity, dtype='int32')
        self.rewards = np.zeros(capacity, 
                        dtype='float32')
        self.terminals = np.zeros(capacity, dtype='int32')

        # use tuple concatenation for multidimensional states
        self.sequence_shape = (sequence_length,) + (state_dim,)
        batch_shape = (batch_size, ) + self.sequence_shape

        # allocate batch containers
        self.batch_states = np.empty(batch_shape, 
                            dtype=np.float32)
        self.batch_actions = np.empty((batch_size, 1), 
                            dtype=np.int32)
        self.batch_rewards = np.empty((batch_size, 1), 
                            dtype=np.float32)
        self.batch_next_states = np.empty(batch_shape, 
                            dtype=np.float32)
        self.batch_terminals = np.empty((batch_size, 1), 
                            dtype=np.int32)

    def store(self, state, action, reward, terminal):
        """
        Stores a state, the action taken in that state,
        and the reward received for taking that action 
        in the state.
    
        args:
            - state: the current state
            - action: the action taken in this state
            - reward: the reward received 
            - terminal: whether this state is terminal
        """

        # place new sample on top
        self.states[self.top] = np.float32(state)
        self.actions[self.top] = np.int32(action)
        self.rewards[self.top] = np.float32(reward)
        self.terminals[self.top] = np.int32(terminal)

        # if replay memory is full, then replace the bottom
        # element with this new sample
        # this is accomplished by moding the incremented bottom
        # because if we are at capacity, then we want to add 
        # one to the bottom index
        # and if it happens that the bottom index is the last
        # index in the buffer, then we want to wrap it around
        if self.size == self.opts.replay_capacity:
            self.bottom = (self.bottom + 1) % self.opts.replay_capacity
        else:
            self.size += 1

        # same logic as for the bottom index
        # we always want to increment the top
        # it's possible that this will go beyond 
        # the length of the buffer, so we wrap it around
        self.top = (self.top + 1) % self.opts.replay_capacity

    def make_last_sequence(self, next_state):
        """
        Given a state, this method creates a sequence
        of sequence_length where the last state in that
        sequence is the passed in state. This is used 
        to pass to an agent to get an action.

        args:
            - next_state: the next state to be 
                inserted last into the sequence
        """

        # take states from the memory
        sequence = np.zeros(self.sequence_shape, 
                    dtype='float32')
        indexes = np.int64(np.arange(
                    self.top - self.opts.sequence_length + 1,
                    self.top))

        sequence[0:self.opts.sequence_length - 1] = self.states.take(
                    indexes, axis=0, mode='wrap')

        # set current states value in sequence
        sequence[-1] = next_state

        # take the same terminal values from the memory
        terminals = self.terminals.take(
                        indexes, axis=0, mode='wrap')
        
        # if any of those terminals are true, 
        # this means that some of the states we collected
        # are from a previous episode and should be ignored
        # to do this, we set indexes of the 
        # sequence up to and including the terminal 
        # index to zero
        true_terminals = np.argwhere(terminals == True)
        if len(true_terminals) > 0:
            real_start = true_terminals[-1][0] + 1
            # replace states with zeros
            # so what this means is that we 
            # will input states full of zeros 
            # to the network and basically just 
            # hope that it learns to ignore those
            # states
            sequence[:real_start] = 0

        return sequence

    def is_full(self):
        """
        Is the replay memory full
        """
        return self.size == self.opts.replay_capacity

    def sample_batch(self):
        """
        Sample a minibatch of data
        """

        # unpack values
        sequence_length = self.opts.sequence_length

        # sample batch_size times from the memory
        count = 0 
        while count < self.opts.batch_size:

            # randomly select a single index into the dataset
            # disregarding elements that are at the end and 
            # would not give a full sample
            index = np.random.randint(
                        self.bottom, 
                        self.bottom + self.size - sequence_length)
            
            # collect indices for the current and next states
            cur_state_indices = np.arange(
                index, index + sequence_length)
            next_state_indices = cur_state_indices + 1
            end_index = index + sequence_length - 1
            
            # original quote:
            # """
            # Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate.
            # """
            # in other words:
            # the agent uses the terminal indicator to zero out the 
            # future value, so here we return the next state, and 
            # just make sure terminals are set correctly. We do 
            # however ignore current states in which a terminal
            # exists in the middle of the sequence
            if np.any(self.terminals.take(
                    cur_state_indices[:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            self.batch_states[count] = self.states.take(
                    cur_state_indices, axis=0, mode='wrap')
            self.batch_actions[count] = self.actions.take(
                    [end_index], mode='wrap')[0]
            self.batch_rewards[count] = self.rewards.take(
                    [end_index], mode='wrap')[0]
            self.batch_terminals[count] = self.terminals.take(
                    [end_index], mode='wrap')[0]
            self.batch_next_states[count] = self.states.take(
                    next_state_indices, axis=0, mode='wrap')

            # increment batch count
            count += 1

        return (self.batch_states,
               self.batch_actions,
               self.batch_rewards,
               self.batch_next_states,
               self.batch_terminals)

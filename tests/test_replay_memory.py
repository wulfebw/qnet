
import numpy as np
import os
import sys
import time
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import replay_memory

class TestOptions(object):
     def __init__(self):
        self.batch_size = 10
        self.replay_capacity = 1000
        self.state_dim = 2

class TestReplayMemory(unittest.TestCase):

    def test_sample_batch(self):
        # confirm that the batch contains samples that were inserted
        np.random.seed(1)
        opts = TestOptions()
        opts.replay_capacity = 1
        opts.batch_size = 1
        rm = replay_memory.ReplayMemory(opts)

        state = np.float32(np.random.rand(opts.state_dim))
        action = np.random.randint(10)
        reward = np.random.randint(10)
        next_state = np.float32(np.random.rand(opts.state_dim))
        terminal = np.random.randint(1)
        rm.store((state, action, reward, next_state, terminal))

        # check that the only sample is sampled
        sstates, sactions, srewards, snext_states, sterminals = rm.sample_batch()
        np.testing.assert_array_equal(state, sstates[0])
        np.testing.assert_array_equal(action, sactions[0])
        np.testing.assert_array_equal(reward, srewards[0])
        np.testing.assert_array_equal(next_state, snext_states[0])
        np.testing.assert_array_equal(terminal, sterminals[0])
        self.assertEquals(1, len(rm.memory))

        # insert another sample
        new_state = np.float32(np.random.rand(opts.state_dim))
        new_action = np.random.randint(10)
        new_reward = np.random.randint(10)
        new_next_state = np.float32(np.random.rand(opts.state_dim))
        new_terminal = np.random.randint(1)
        rm.store((new_state, new_action, new_reward, new_next_state, new_terminal))

        # size the same
        self.assertEquals(1, len(rm.memory))
        # check that the first sample was discarded for the second
        sstates, sactions, srewards, snext_states, sterminals = rm.sample_batch()
        np.testing.assert_array_equal(new_state, sstates[0])
        np.testing.assert_array_equal(new_action, sactions[0])
        np.testing.assert_array_equal(new_reward, srewards[0])
        np.testing.assert_array_equal(new_next_state, snext_states[0])
        np.testing.assert_array_equal(new_terminal, sterminals[0])
        self.assertEquals(1, len(rm.memory))


class TestSequenceReplayMemory(unittest.TestCase):

    def test_make_last_sequence_basic_operation(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 3
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = False
        for idx in range(4):
            rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(
                    np.zeros(opts.state_dim))
        expected = [[1, 1], [1, 1], [0, 0]]
        np.testing.assert_array_equal(actual, expected)

    def test_make_last_sequence_preceding_state_terminal(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 3
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        state = np.ones(opts.state_dim)
        action = 0
        reward = 0

        terminal = False
        rm.store(state, action, reward, terminal)

        terminal = True
        rm.store(state, action, reward, terminal)

        cur_state = np.arange(opts.state_dim)
        actual = rm.make_last_sequence(cur_state)
        expected = [[0, 0], [0, 0], [0, 1]]
        np.testing.assert_array_equal(actual, expected)

    def test_make_last_sequence_some_previous_state_terminal_not_in_sequence(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 3
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = True
        rm.store(state, action, reward, terminal)
        terminal = False
        for idx in range(2):
            rm.store(state, action, reward, terminal)

        cur_state = np.arange(opts.state_dim)
        actual = rm.make_last_sequence(cur_state)
        expected = [[1, 1], [1, 1], [0, 1]]
        np.testing.assert_array_equal(actual, expected)

    def test_make_last_sequence_terminal_state_within_sequence_but_not_preceding(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 4
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        # tuple 1
        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = True
        rm.store(state, action, reward, terminal)

        # tuple 3
        terminal = False
        rm.store(state, action, reward, terminal)

        cur_state = np.arange(opts.state_dim)
        actual = rm.make_last_sequence(cur_state)
        expected = [[0, 0], [0, 0], [1, 1], [0, 1]]
        np.testing.assert_array_equal(actual, expected)

        # now terminal as first item
        # tuple 1
        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = True
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 3
        terminal = False
        rm.store(state, action, reward, terminal)

        cur_state = np.arange(opts.state_dim)
        actual = rm.make_last_sequence(cur_state)
        expected = [[0, 0], [1, 1], [1, 1], [0, 1]]
        np.testing.assert_array_equal(actual, expected)

    def test_make_last_sequence_terminal_state_first_in_made_sequence_wrap(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 4
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        # tuple 1
        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = False
        for i in range(opts.replay_capacity - 1):
            rm.store(state, action, reward, terminal)
        terminal = True
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 3
        terminal = False
        rm.store(state, action, reward, terminal)

        cur_state = np.arange(opts.state_dim)
        actual = rm.make_last_sequence(cur_state)
        expected = [[0, 0], [1, 1], [1, 1], [0, 1]]
        np.testing.assert_array_equal(actual, expected)


    def test_make_last_sequence_insufficient_samples_for_full_sequence(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 4
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        # tuple 1
        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        next_state = np.ones(opts.state_dim)
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = False
        rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(opts.state_dim))
        expected = [[0, 0], [1, 1], [1, 1], [0, 1]]
        np.testing.assert_array_equal(actual, expected)

    def test_make_last_sequence_empty(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 4
        opts.replay_capacity = 30
        rm = replay_memory.SequenceReplayMemory(opts)

        actual = rm.make_last_sequence(np.arange(opts.state_dim)).tolist()
        expected = [[0, 0], [0, 0], [0, 0], [0, 1]]
        self.assertEquals(actual, expected)

    def test_minibatch_sample_shapes_1D_state_sequence_length_1(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 100
        opts.state_dim = 2
        opts.sequence_length = 1
        opts.replay_capacity = 1000
        rm = replay_memory.SequenceReplayMemory(opts)

        # unpack
        batch_size = opts.batch_size
        sequence_length = opts.sequence_length

        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = False
        for idx in range(1000):
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        self.assertEquals(
            states.shape, (batch_size, sequence_length, opts.state_dim))
        self.assertEquals(
            actions.shape, (batch_size, 1))
        self.assertEquals(
            rewards.shape, (batch_size, 1))
        self.assertEquals(
            next_states.shape, (batch_size, sequence_length, opts.state_dim))
        self.assertEquals(
            terminals.shape, (batch_size, 1))

    def test_minibatch_sample_shapes_1D_state_sequence_length_2(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 10
        opts.state_dim = 2
        opts.sequence_length = 2
        opts.replay_capacity = 1000
        rm = replay_memory.SequenceReplayMemory(opts)

        # unpack
        batch_size = opts.batch_size
        sequence_length = opts.sequence_length

        state = np.ones(opts.state_dim)
        action = 0
        reward = 0
        terminal = False
        for idx in range(1000):
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        self.assertEquals(
            states.shape, (batch_size, sequence_length, opts.state_dim))
        self.assertEquals(
            states.sum(), batch_size * sequence_length * opts.state_dim)
        self.assertEquals(
            actions.shape, (batch_size, 1))
        self.assertEquals(
            rewards.shape, (batch_size, 1))
        self.assertEquals(
            next_states.shape, (batch_size, sequence_length, opts.state_dim))
        self.assertEquals(
            next_states.sum(), batch_size * sequence_length * opts.state_dim)
        self.assertEquals(
            terminals.shape, (batch_size, 1))

    def test_minibatch_sample_shapes_1D_state_terminal(self):
        np.random.seed(1)
        opts = TestOptions()
        opts.batch_size = 200
        opts.state_dim = 2
        opts.sequence_length = 2
        opts.replay_capacity = 1000
        rm = replay_memory.SequenceReplayMemory(opts)

        # unpack
        batch_size = opts.batch_size
        sequence_length = opts.sequence_length

        prev_state_terminal = False
        for idx in range(1, 1001):
            action = 0
            reward = 0
            state = np.ones(opts.state_dim) * idx
            state = state if not prev_state_terminal else np.zeros(opts.state_dim)
            prev_state_terminal = False if np.random.random() < .8 else True
            rm.store(state, action, reward, prev_state_terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        for state, next_state, terminal in zip(states, next_states, terminals):
            if terminal:
                self.assertEquals(
                    next_state.tolist()[-1], 
                    np.zeros(opts.state_dim).tolist())

if __name__ == '__main__':
    unittest.main()
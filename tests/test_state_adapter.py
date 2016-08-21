
import numpy as np
import os
import sys
import time
import unittest

path = os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')
sys.path.append(os.path.abspath(path))

import state_adapter

class TestOptions(object):
     def __init__(self):
        self.state_dim = 1
        self.high = np.array([2])
        self.low = np.array([0])

class TestRangeAdapter(unittest.TestCase):

    def test_normalize_state(self):
        opts = TestOptions()
        sa = state_adapter.RangeAdapter(opts)

        # basic 
        states = np.array([[0],[1],[2]])
        expected = [[-.5], [0], [.5]]
        for idx, s in enumerate(states):
            norm_state = sa.normalize_state(s)
            np.testing.assert_array_equal(expected[idx], norm_state)

        # case where range is zero
        opts = TestOptions()
        opts.high = [1]
        opts.low = [1]
        sa = state_adapter.RangeAdapter(opts)
        state = [1]
        norm_state = sa.normalize_state(state)
        np.testing.assert_array_equal(norm_state, [0])


if __name__ == '__main__':
    unittest.main()
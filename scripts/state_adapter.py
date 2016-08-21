
import numpy as np

class NormalizingAdapter(object):

    def __init__(self, opts):
        self.opts = opts
        self.count = 1.
        self.state_means = np.zeros(opts.state_dim)
        # initialize to ones for numeric stability
        self.state_diff_sums = np.ones(opts.state_dim)
        self.state_stds = np.ones(opts.state_dim)

    def normalize_state(self, state):
        # update statistics
        temp_means = self.state_means[:]
        diff = np.subtract(state, temp_means)
        self.state_means += diff / self.count
        new_diff = np.subtract(state, self.state_means)
        self.state_diff_sums += np.multiply(diff, new_diff)
        self.count += 1
        k = self.count - 2 if self.count > 2 else 1
        self.state_stds = np.sqrt(self.state_diff_sums / k)

        # normalize and return state
        state = np.subtract(state, self.state_means)
        state = np.divide(state, self.state_stds)
        return state

class RangeAdapter(object):

    def __init__(self, opts):
        self.opts = opts
        self.means = (np.array(opts.high) + opts.low) / 2.
        self.ranges = np.float32((np.array(opts.high) - opts.low))
        # if the range for any state variable is zero, then 
        # just set the range to be one. The reason being 
        # that this value will be set to zero during mean
        # subtraction, so it should not matter what the range
        # is, so long as it is not zero since this gives 
        # 0/0 which is undefined, yo.
        self.ranges[self.ranges == 0] = 1
        
        # if opts.high / low is invalid then ranges
        # or means might be nan or inf, so confirm
        # that that is not the case
        print self.ranges
        print self.means
        print opts.high
        print opts.low
        assert not np.any(np.isnan(self.ranges))
        assert not np.any(np.isinf(self.ranges))
        assert not np.any(np.isnan(self.means))
        assert not np.any(np.isinf(self.means))

    def normalize_state(self, state):
        state = np.subtract(state, self.means)
        state = np.divide(state, self.ranges)
        return state

class IdentityAdapter(object):

    def __init__(self, opts=None):
        self.means = []
        self.ranges = []

    def normalize_state(self, state):
        if len(np.shape(state)) == 0:
            return np.array([state])
        return np.array(state)
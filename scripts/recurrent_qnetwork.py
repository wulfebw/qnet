"""
General discussion of recurrent q network:
There are a few different ways you could implement an rqn, 
and it's not obvious which way would be the best. 
See http://arxiv.org/pdf/1507.06527v3.pdf for discussion.

The rqn is implemented as follows:
1. define the current state as frames {i,i+1,i+2}
    (assume a training sequence length (i.e., BPTT 
    rollout length) of 3)
2. define the next state as frames {i+1,i+2,i+3}
3. next fprop these sequences to get respective 
    qvalues for each.
4. use the usual q-learning loss with these qvalues.
"""

import lasagne
from lasagne.regularization import regularize_network_params, l2
import numpy as np
import sys
import theano
import theano.tensor as T

import qnetwork

class RecurrentQNetwork(qnetwork.SequenceQNetwork):
    """
    A recurrent QNetwork. 
    """

    def __init__(self, opts):
        super(RecurrentQNetwork, self).__init__(opts)

    def initialize_network(self):
        """
        :description: this method initializes the network, updates, and theano functions for training and 
            retrieving q values. Here's an outline: 

            1. build the q network and target q network
            2. initialize theano symbolic variables used for compiling functions
            3. initialize the theano numeric variables used as input to functions
            4. formulate the symbolic loss 
            5. formulate the symbolic updates 
            6. compile theano functions for training and for getting q_values
        """
        batch_size = self.opts.batch_size
        state_dim = self.opts.state_dim
        num_actions = self.opts.num_actions
        sequence_length = self.opts.sequence_length

        # 1. build the q network and target q network
        self.build_network = self.get_build_network()
        self.l_out = self.build_network()
        self.next_l_out = self.build_network()
        self.reset_target_network()

        # 2. initialize theano symbolic variables used for compiling functions
        states = T.tensor3('states')
        action_idxs = T.icol('action_idxs')
        rewards = T.col('rewards')
        next_states = T.tensor3('next_states')
        # terminals are used to indicate a terminal state in the episode 
        # and hence a mask over the future q values i.e., Q(s',a')
        terminals = T.icol('terminals')

        # 3. initialize the theano numeric variables used as input to functions
        self.states_shared = theano.shared(
                                np.zeros((batch_size, sequence_length, state_dim), 
                                dtype=theano.config.floatX))
        self.next_states_shared = theano.shared(
                                np.zeros((batch_size, sequence_length, state_dim), 
                                dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
                                np.zeros((batch_size, 1), 
                                dtype=theano.config.floatX), 
                                broadcastable=(False, True))
        self.action_idxs_shared = theano.shared(
                                np.zeros((batch_size, 1), 
                                dtype='int32'), 
                                broadcastable=(False, True))
        self.terminals_shared = theano.shared(
                                np.zeros((batch_size, 1), 
                                dtype='int32'), 
                                broadcastable=(False, True))

        # 4. formulate the symbolic loss 
        # get the qvalues for both
        q_vals = lasagne.layers.get_output(
                    self.l_out, states)
        next_q_vals = lasagne.layers.get_output(
                    self.next_l_out, next_states)

        # convert terminals to float so they can be used on gpu
        float_terminals = terminals.astype(theano.config.floatX)

        # use action mask and multiplication instead of indexing 
        # so that this can be used on gpu
        # reshape range as row and action_idxs as column and then 
        # use broadcasting to get the full matrix of true/false values
        actionmask = T.eq(T.arange(num_actions).reshape((1, -1)),
                        action_idxs.reshape(
                        (-1, 1))).astype(theano.config.floatX)

        best_next_q_values = T.max(next_q_vals, axis=1, keepdims=True)
        target = (rewards +
                  (T.ones_like(float_terminals) - float_terminals) *
                  self.opts.discount * best_next_q_values)
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        # a lot of the recent work clips the td error at 1 so we do that here
        # the problem is that gradient backpropagating through this minimum node
        # will be zero if diff is larger then 1.0 (because changing params before
        # the minimum does not impact the output of the minimum). To account for 
        # this we take the part of the td error (magnitude) greater than 1.0 and simply
        # add it to the loss, which allows gradient to backprop but just linearly
        # in the td error rather than quadratically
        quadratic_part = T.minimum(abs(diff), self.opts.max_quadratic_loss)
        linear_part = abs(diff) - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + linear_part
        loss = T.mean(loss) + self.opts.regularization * \
                regularize_network_params(self.l_out, l2)
        
        # 5. formulate the symbolic updates 
        params = lasagne.layers.helper.get_all_params(
                self.l_out, trainable=True)  
        self.learning_rate_shared = theano.shared(
                np.array(self.opts.learning_rate, dtype=np.float32))
        updates = lasagne.updates.adamax(
                loss, params, self.learning_rate_shared)

        # 6. compile theano functions for training and for getting q_values
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            action_idxs: self.action_idxs_shared,
            terminals: self.terminals_shared
        }
        self._train = theano.function([], [loss], 
                        updates=updates, givens=givens)

        # 7. build q_values function
        qvalues_states = T.tensor3('states')
        
        deterministic_q_vals = lasagne.layers.get_output(
                                self.l_out, qvalues_states, 
                                deterministic=True)

        self.state_shared = theano.shared(
                                np.zeros((sequence_length, state_dim), 
                                dtype=theano.config.floatX))

        givens = {qvalues_states: self.state_shared.reshape(
                    (1, sequence_length, state_dim))}
        self._get_q_values = theano.function(
                                [], deterministic_q_vals[0], 
                                givens=givens)

    def get_build_network(self):
        nt = self.opts.subnetwork_type

        if nt == 'single_layer_lstm':
            return self.build_single_layer_lstm
        elif nt == 'stacked_lstm':
            return self.build_stacked_lstm
        elif nt == 'clockwork_lstm':
            return self.build_clockwork_lstm
        elif nt == 'linear_rnn':
            return self.build_linear_rnn
        else:
            raise ValueError("Unrecognized network_type: {}".format(self.opts.subnetwork_type))

    def build_single_layer_lstm(self):
        batch_size = self.opts.batch_size
        sequence_length = self.opts.sequence_length
        state_dim = self.opts.state_dim
        num_hidden = self.opts.num_hidden
        num_actions = self.opts.num_actions

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, state_dim)
        )

        default_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.))
        forget_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(2.))
        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.opts.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=True
        )
        
        l_out = lasagne.layers.DenseLayer(
            l_lstm1,
            num_units=num_actions,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out

    def build_stacked_lstm(self):
        """
        https://arxiv.org/pdf/1312.6026v5.pdf
        """

        batch_size = self.opts.batch_size
        sequence_length = self.opts.sequence_length
        state_dim = self.opts.state_dim
        num_hidden = self.opts.num_hidden
        num_actions = self.opts.num_actions

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, state_dim)
        )

        default_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.))
        forget_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(1.))
        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.opts.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=False
        )

        l_lstm2 = lasagne.layers.LSTMLayer(
            l_lstm1, 
            num_units=self.opts.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=True
        )

        l_slice1 = lasagne.layers.SliceLayer(l_lstm1, -1, 1)
        l_merge = lasagne.layers.ConcatLayer([l_slice1, l_lstm2])

        l_out = lasagne.layers.DenseLayer(
            l_merge,
            num_units=num_actions,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out

    def build_clockwork_lstm(self):
        """
        https://arxiv.org/pdf/1402.3511.pdf
        """

        batch_size = self.opts.batch_size
        sequence_length = self.opts.sequence_length
        state_dim = self.opts.state_dim
        num_hidden = self.opts.num_hidden
        num_actions = self.opts.num_actions

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, state_dim)
        )
        assert sequence_length % 3 == 1, """when using the hierarchical_stacked_lstm_with_merge, 
                the sequence length must be such that sequence_length % 3 == 1 because 
                this allows for taking the slice of a length 1 sequence while still 
                keeping at least one element and simultaneously allowing for any 
                slice made to incorporate the last element of the original sequence. 
                If you dont like this, you can change it easily by using a mask but im lazy."""

        default_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.))
        forget_gate = lasagne.layers.recurrent.Gate(
            W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(5.))
        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.opts.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=False,
            name='l_lstm1'
        )
        l_slice1_up = lasagne.layers.SliceLayer(l_lstm1, slice(0, sequence_length, 3), 1, name='l_slice1_up')

        l_slice1_in = lasagne.layers.SliceLayer(l_in, slice(0, sequence_length, 3), 1, name='l_slice1_in')
        l_rnn1 = lasagne.layers.RecurrentLayer(
            l_slice1_in,
            num_units=self.opts.num_hidden,
            W_in_to_hid=lasagne.init.HeNormal(),
            W_hid_to_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.),
            nonlinearity=None,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=False,
            name='rnn1'
        )

        l_merge_up = lasagne.layers.ConcatLayer([l_rnn1, l_slice1_up], name='l_merge_up')
        l_lstm2 = lasagne.layers.LSTMLayer(
            l_merge_up, 
            num_units=self.opts.num_hidden, 
            nonlinearity=lasagne.nonlinearities.tanh,
            cell=default_gate,
            ingate=default_gate,
            outgate=default_gate,
            forgetgate=forget_gate,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=True,
            name='l_lstm2'
        )

        l_slice1_out = lasagne.layers.SliceLayer(l_merge_up, -1, 1, name='l_slice1_out')
        l_merge_out = lasagne.layers.ConcatLayer([l_slice1_out, l_lstm2], name='l_merge_out')

        l_out = lasagne.layers.DenseLayer(
            l_merge_out,
            num_units=num_actions,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0),
            name='l_out'
        )

        return l_out

    def build_linear_rnn(self):
        batch_size = self.opts.batch_size
        sequence_length = self.opts.batch_size
        state_dim = self.opts.state_dim
        num_hidden = self.opts.num_hidden
        num_actions = self.opts.num_actions

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, state_dim)
        )

        l_rnn1 = lasagne.layers.RecurrentLayer(
            l_in,
            num_units=self.opts.num_hidden,
            W_in_to_hid=lasagne.init.HeNormal(),
            W_hid_to_hid=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0.),
            nonlinearity=None,
            grad_clipping=self.opts.rnn_grad_clip,
            only_return_final=True
        )

        l_out = lasagne.layers.DenseLayer(
            l_rnn1,
            num_units=num_actions,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out

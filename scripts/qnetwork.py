
import lasagne
from lasagne.regularization import regularize_network_params, l2
import numpy as np
import theano
import theano.tensor as T

class QNetwork(object):

    def __init__(self, opts):
        """
        args:
            - opts: variable options for the network
        """
        self.opts = opts
        self.update_counter = 0
        self.initialize_network()

    def train(self, states, action_idxs, rewards, next_states, terminals):
        """
        Perform a q-learning update using the (s,a,r,s') 
        tuples provided

        args:
            - states: batch of states, 
                shape (N,D) = (batch_size, state_dim)
            - action_idxs: the indices of actions 
                taken by the agent, shape = (N,)
            - rewards: rewards associated with being in 
                state s and taking action a, shape = (N,)
            - next_states: batch of next_states, 
                shape (N,D) = (batch_size, state_dim)
            - terminals: whether corresponding next_state 
                was a terminal state. If so, the max_a' Q(s',a')
                term will be zero in the q-learning update.
        """
        self.update_counter += 1

        # update target network
        if self.update_counter % self.opts.freeze_interval == 0:
            self.reset_target_network()

        # decrease learning rate
        if self.update_counter % self.opts.decay_lr_every == 0:
            self.opts.learning_rate *= self.opts.decay_lr_ratio
            self.opts.learning_rate = max(
                self.opts.learning_rate, self.opts.min_learning_rate)
            self.learning_rate_shared.set_value(
                np.float32(self.opts.learning_rate))

        self.states_shared.set_value(states)
        self.action_idxs_shared.set_value(action_idxs)
        self.rewards_shared.set_value(rewards)
        self.next_states_shared.set_value(next_states)
        self.terminals_shared.set_value(terminals)

        loss = self._train()
        return loss

    def get_q_values(self, state):
        """
        Returns the q_values associated with a single 
        state for the purposes of deciding which action to take.

        args:
            - state: state to compute q_values for, shape = (D,)
        """
        self.state_shared.set_value(
            state.astype(theano.config.floatX))
        q_values = self._get_q_values()
        return q_values

    def get_params(self):
        """
        Return a numpy array containing all of the parameters of the network. 
        Used for retrieving weights to save.
        """
        weights = lasagne.layers.helper.get_all_param_values(
                    self.l_out)
        return weights

    def set_params(self, params):
        """
        Set the parameters of the network to the 
        provided parameters. Used for loading saved weights.
        """
        lasagne.layers.set_all_param_values(
            self.l_out, params)
        self.reset_target_network()

    def reset_target_network(self):
        """
        Set the target weights to the current weights.
        """
        all_params = lasagne.layers.helper.get_all_param_values(
                        self.l_out)
        lasagne.layers.helper.set_all_param_values(
                        self.next_l_out, all_params)

    def initialize_network(self):
        """
        This method initializes the network, updates, 
        and theano functions for training and retrieving q values. 

        Outline: 
        1. build the q network and target q network
        2. initialize theano symbolic variables used 
            for compiling functions
        3. initialize the theano numeric variables 
            used as input to functions
        4. formulate the symbolic loss 
        5. formulate the symbolic updates 
        6. compile theano functions for training and 
            for getting q_values
        """
        batch_size = self.opts.batch_size
        state_dim = self.opts.state_dim
        num_actions = self.opts.num_actions

        # 1. build the q network and target q network
        self.l_out = self.build_network()
        self.next_l_out = self.build_network()
        self.reset_target_network()

        # 2. initialize theano symbolic variables used for compiling functions
        states = T.matrix('states')
        action_idxs = T.icol('action_idxs')
        rewards = T.col('rewards')
        next_states = T.matrix('next_states')
        # terminals are used to indicate a terminal state in the episode 
        # and hence a mask over the future q values i.e., Q(s',a')
        terminals = T.icol('terminals')

        # 3. initialize the theano numeric variables used as input to functions
        self.states_shared = theano.shared(
                                np.zeros((batch_size, state_dim), 
                                dtype=theano.config.floatX))
        self.next_states_shared = theano.shared(
                                np.zeros((batch_size, state_dim), 
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

        # updates = lasagne.updates.sgd(
        #         loss, params, self.learning_rate_shared)
        # updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=.8)

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

        # build q_values function
        deterministic_q_vals = lasagne.layers.get_output(
                                self.l_out, states, deterministic=True)

        self.state_shared = theano.shared(np.zeros((state_dim), 
                                dtype=theano.config.floatX))
        givens = {states: self.state_shared.reshape((1, state_dim))}
        self._get_q_values = theano.function(
                                [], deterministic_q_vals[0], givens=givens)

    def build_network(self):
        """
        Builds the computational graph in lasagne.
        """
        if self.opts.nonlinearity == 'relu':
            nonlinearity = lasagne.nonlinearities.rectify
        elif self.opts.nonlinearity == 'tanh':
            nonlinearity = lasagne.nonlinearities.tanh
        elif self.opts.nonlinearity == 'leaky_relu':
            nonlinearity = lasagne.nonlinearities.leaky_rectify
        elif self.opts.nonlinearity == 'linear':
            nonlinearity = None
            # if no nonlinearity used, then only use a single hidden layer
            self.opts.num_hidden_layers = 1
        else:
            msg = 'theano nonlinearity must be one of \{relu, tanh, leaky_relu, linear\}'
            raise ValueError(msg)

        in_shape = (self.opts.batch_size, self.opts.state_dim) 
        if self.opts.dropout_prob != 0.0:
            in_shape = (None, self.opts.state_dim) 

        l_in = lasagne.layers.InputLayer(
            shape=in_shape,
            name='in'
        )
        l_hid = l_in

        # if non-custom hidden layer sizes, then fill with all num_hidden
        if self.opts.hidden_layer_sizes == []:
            self.opts.hidden_layer_sizes = [self.opts.num_hidden] * self.opts.num_hidden_layers

        for hidden_idx, size in enumerate(self.opts.hidden_layer_sizes):
            l_hid = lasagne.layers.DenseLayer(
                l_hid,
                num_units=size,
                nonlinearity=nonlinearity, 
                W=lasagne.init.HeNormal(),
                b=lasagne.init.Constant(.1),
                name='hid_{}'.format(hidden_idx)
            )

            # apply dropout after each hidden layer
            if self.opts.dropout_prob != 0.0:
                l_hid = lasagne.layers.DropoutLayer(l_hid, p=self.opts.dropout_prob)

        l_out = lasagne.layers.DenseLayer(
            l_hid,
            num_units=self.opts.num_actions,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0),
            name='out'
        )

        return l_out

    def load_weights_from_file(self, filepath):
        # load the saved params
        weights_dict = np.load(filepath)
        param_keys = sorted([(int(k[4:]),k) for k in weights_dict.keys() if 'arr' in k])
        param_keys = [v for (k,v) in param_keys]
        loaded_params = [weights_dict[k] for k in param_keys]

        # check that they are the same shape as the current parameters
        current_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        if len(current_params) != len(loaded_params):
            msg = """there are differing numbers of parameters 
                     in the loaded and current versions. current: {}
                     loaded: {}""".format(len(current_params), len(loaded_params))

        current_shapes = [w.shape for w in current_params]
        loaded_shapes = [w.shape for w in loaded_params]
            
        for idx, (loaded, current) in enumerate(zip(loaded_params, current_params)):
            if np.shape(loaded) != np.shape(current):
                msg = """the shape of the loaded weights does not 
                        match the shape of current weights. The 
                        loaded weights in this position have shape {} whereas the 
                        current weights in this position have shape
                        {}. This difference occurs in layer {}. \nThe shapes overall
                        for loaded are {} \nand for current are {}. \nFilepath used: {}
                        """.format(np.shape(loaded), np.shape(current), idx, loaded_shapes, current_shapes, filepath)
                raise ValueError(msg)

        # set param values since they must be the same shape
        # and also reset the target network weights
        lasagne.layers.helper.set_all_param_values(self.l_out, loaded_params)
        self.reset_target_network()

class SequenceQNetwork(QNetwork):

    def __init__(self, opts):
        """
        args:
            - opts: variable options for the network
        """
        super(SequenceQNetwork, self).__init__(opts)

    def initialize_network(self):
        """
        This method initializes the network, updates, 
        and theano functions for training and retrieving q values. 

        Outline: 
        1. build the q network and target q network
        2. initialize theano symbolic variables used 
            for compiling functions
        3. initialize the theano numeric variables 
            used as input to functions
        4. formulate the symbolic loss 
        5. formulate the symbolic updates 
        6. compile theano functions for training and 
            for getting q_values
        """
        batch_size = self.opts.batch_size
        state_dim = self.opts.state_dim
        num_actions = self.opts.num_actions
        sequence_length = self.opts.sequence_length

        # 1. build the q network and target q network
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
        # first reshape the states  and next state to flatten 
        # across sequence dimension 
        flat_states = T.reshape(states,
                     (batch_size, sequence_length * state_dim))
        flat_next_states = T.reshape(next_states,
                     (batch_size, sequence_length * state_dim))

        # get the qvalues for both
        q_vals = lasagne.layers.get_output(
                    self.l_out, flat_states)
        next_q_vals = lasagne.layers.get_output(
                    self.next_l_out, flat_next_states)

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

        # updates = lasagne.updates.sgd(
        #         loss, params, self.learning_rate_shared)
        # updates = lasagne.updates.apply_nesterov_momentum(updates, momentum=.8)

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
        qvalues_states = T.matrix('states')
        flat_qvalues_states = T.reshape(qvalues_states,
                     (1, sequence_length * state_dim))
        
        deterministic_q_vals = lasagne.layers.get_output(
                                self.l_out, flat_qvalues_states, 
                                deterministic=True)

        self.state_shared = theano.shared(
                                np.zeros((sequence_length, state_dim), 
                                dtype=theano.config.floatX))

        givens = {qvalues_states: self.state_shared.reshape(
                    (1, sequence_length * state_dim))}
        self._get_q_values = theano.function(
                                [], deterministic_q_vals[0], 
                                givens=givens)

    def build_network(self):
        """
        Builds the computational graph in lasagne.
        """
        if self.opts.nonlinearity == 'relu':
            nonlinearity = lasagne.nonlinearities.rectify
        elif self.opts.nonlinearity == 'tanh':
            nonlinearity = lasagne.nonlinearities.tanh
        elif self.opts.nonlinearity == 'leaky_relu':
            nonlinearity = lasagne.nonlinearities.leaky_rectify
        elif self.opts.nonlinearity == 'linear':
            nonlinearity = None
            # if no nonlinearity used, then only use a single hidden layer
            self.opts.num_hidden_layers = 1
        else:
            msg = 'theano nonlinearity must be one of \{relu, tanh, leaky_relu, linear\}'
            raise ValueError(msg)

        in_shape = (self.opts.batch_size,
            self.opts.sequence_length * self.opts.state_dim) 
        if self.opts.dropout_prob != 0.0:
            in_shape = (None, self.opts.sequence_length * self.opts.state_dim) 

        l_in = lasagne.layers.InputLayer(
            shape=in_shape,
            name='in'
        )
        l_hid = l_in

        # if non-custom hidden layer sizes, then fill with all num_hidden
        if self.opts.hidden_layer_sizes == []:
            self.opts.hidden_layer_sizes = [self.opts.num_hidden] * self.opts.num_hidden_layers

        for hidden_idx, size in enumerate(self.opts.hidden_layer_sizes):
            l_hid = lasagne.layers.DenseLayer(
                l_hid,
                num_units=size,
                nonlinearity=nonlinearity, 
                W=lasagne.init.HeNormal(),
                b=lasagne.init.Constant(.1),
                name='hid_{}'.format(hidden_idx)
            )

            # apply dropout after each hidden layer
            if self.opts.dropout_prob != 0.0:
                l_hid = lasagne.layers.DropoutLayer(l_hid, p=self.opts.dropout_prob)

        l_out = lasagne.layers.DenseLayer(
            l_hid,
            num_units=self.opts.num_actions,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0),
            name='out'
        )

        return l_out


import numpy as np

import logger

class Agent(object):
    """
    A class that wraps a network so it may more easily interact 
    with an experiment. 
    """
    
    def __init__(self, network, policy, replay_memory, 
            state_adapter, log, opts):
        """
        args:
            - network: the network the agent uses to evaluate states
            - policy: a class that decides which action to 
                take given the values of those actions
            - replay_memory: memory used to store dataset as 
                it is gathered
            - state_adapter: adapts the state to a representation 
                internal to agent
            - log: logger for saving weights and training information
            - opts: options classed used for paramerter values
        """
        self.network = network
        self.policy = policy
        self.replay_memory = replay_memory
        self.state_adapter = state_adapter
        self.logger = log
        self.opts = opts
        self.step_counter = 0
        self.train_counter = 0
        
    def step(self, next_state, reward, terminal):
        """
        The primary method of this class, which steps 
        the agent and network forward one time step. 
        This includes selecting an action index and storing 
        the new state and reward.

        args:
            - next_state: the next state observed 
                (i.e., s') not yet formatted for the agent
            - reward: the reward associated with having 
                moved from the current state to the next state
            - terminal: whether next_state is terminal

        return values:
            - action index for next action to be taken within
                the environment
        """
        # increment number of steps
        self.step_counter += 1

        # need to transform an external state format 
        # to an internal one
        next_state = self.state_adapter.normalize_state(next_state)

        # if in validation then just return action
        if self.opts.in_validation:
            self.logger.log_value('reward', reward)
            return self.get_action_idx(next_state)

        # store current (s,a,r,s',t) tuple 
        # where t = {False, True} for terminal not terminal
        sample_terminal = 1 if terminal else 0
        sample = (self.state, self.action_idx, reward, 
                    next_state, sample_terminal)
        self.replay_memory.store(sample)

        # retrieve an action
        action_idx = self.get_action_idx(next_state)

        # set current values
        self.state = next_state
        self.action_idx = action_idx

        # log information
        self.logger.log_value('reward', reward)
        self.logger.log_value('action', self.action_idx)

        return action_idx

    def train(self):
        """
        Collects a minibatch of experiences and passes 
        them to the network for use in updating weights.
        """

        # wait until replay memory is full
        if not self.replay_memory.is_full():
            return

        # increment training counter
        self.train_counter += 1

        # collect minibatch
        (states, action_idxs, rewards, next_states, 
            terminals) = self.replay_memory.sample_batch()

        # temp
        states = np.squeeze(states, axis=1)
        next_states = np.squeeze(next_states, axis=1)

        # pass to network to perform training
        loss = self.network.train(
                states, action_idxs, rewards, next_states, terminals)

        # log information
        self.logger.log_value('loss', loss)

    def get_action_idx(self, state):
        """
        Gets an action index given the current state by 
        retrieving qvalues and deferring to a policy.

        args:
            - state: the state used to determine the 
                action index, in agent format

        return values:
            - action_idx: index of action to take in state
        """
        # wait until agent starts learning befor using network 
        # to decide action
        if not self.replay_memory.is_full():
            return self.policy.random_action_index()

        # action choice depends on qvalues so use separate class
        # for actual action selection

        q_values = self.network.get_q_values(state)
        # print state
        # print q_values
        action_idx = self.policy.choose_action_index(q_values)
        return action_idx

    def start_episode(self, state):
        """
        Determines the first action to take and 
        initializes internal variables.

        args:
            - state: the state used to determine the action,
                not yet in agent format
        """
        self.state = self.state_adapter.normalize_state(state)
        self.action_idx = self.get_action_idx(self.state)
        self.logger.log_value('action', self.action_idx)
        return self.action_idx

    def finish_episode(self):
        """
        Perform tasks at the end of episode. 
        """
        # log info
        self.logger.log_value('learning_rate', 
            self.opts.learning_rate)
        self.logger.log_value('exploration_prob', 
            self.opts.exploration_prob)
        self.logger.finish_episode(
            self.network, self.state_adapter, self.replay_memory)

class SequenceAgent(Agent):
    """
    An agent that orchastrates the interaction between 
    a sequential replay memory and a (possibly recurrent)
    network.
    """
    def __init__(self, network, policy, replay_memory, 
            state_adapter, log, opts):
        super(SequenceAgent, self).__init__(network,
            policy, replay_memory, state_adapter, log, opts)

    def step(self, next_state, reward, terminal):
        """
        The primary method of this class, which 'steps' 
        the agent and network forward one time step. 
        This includes selecting an action, making use of 
        the new state and reward, and performing training.

        args:
            - next_state: the next state observed (i.e., s')
            - reward: the reward associated with having moved
                from the previous state to the current state
        
        returns: 
            - the action to next be taken within the environment
        """
        # convert to agent format
        next_state = self.state_adapter.normalize_state(next_state)

        # store current (s,a,r,s') tuple
        self.replay_memory.store(
            self.state, self.action_idx, reward, terminal)

        # retrieve an action
        action_idx = self.get_action_idx(next_state)

        # set previous values
        self.state = next_state
        self.action_idx = action_idx

        # log information
        self.logger.log_value('reward', reward)
        self.logger.log_value('action', action_idx)

        return action_idx

    def get_action_idx(self, state):
        """
        Gets an action given the current state. 
        
        args:
            - state: the state used to determine the action
        """
        # wait until agent starts learning 
        # to use network to decide action
        if not self.replay_memory.is_full():
            return self.policy.random_action_index()

        sequence = self.replay_memory.make_last_sequence(state)

        # temp
        sequence = sequence.flatten()

        q_values = self.network.get_q_values(sequence)
        return self.policy.choose_action_index(q_values)

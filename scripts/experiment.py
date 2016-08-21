
class Experiment(object):
    """
    This class orchestrates the interaction 
    between an agent and an env.
    """

    def __init__(self, env, agent, opts):
        """
        args:
            - env: the environment in which the agent acts
            - agent: the agent that acts within the experiment
            - opts: experiment options
        """
        self.env = env
        self.agent = agent
        self.opts = opts

    def run(self):
        """
        Main method loop that runs the entire experiment.
        """
        for episode in xrange(self.opts.num_episodes):
            # run an episode, collecting experience
            self.run_episode()

            # run a number of training batches for each episode
            for batch in xrange(self.opts.train_updates_per_episode):
                self.agent.train()

            # occasionally run validation episodes
            if episode % self.opts.validation_every == 0:
                self.opts.in_validation = True
                self.run_episode()
                self.opts.in_validation = False

        self.env.monitor.close()

    def run_episode(self):
        """
        Runs a single episode.
        """
        obs = self.env.reset()
        action_idx = self.agent.start_episode(obs)
        action_idx = 1
        for step in xrange(self.opts.max_steps):
            # get the next state and reward
            next_obs, reward, terminal, _ = self.env.step(action_idx)

            # print 'obs: {}'.format(obs)
            # print 'action_idx: {}'.format(action_idx)
            # print 'next_obs {}'.format(next_obs)
            # print 'reward: {}'.format(reward)
            # print 'terminal: {}'.format(terminal)
            # raw_input()

            # inform the agent and get a new action_idx
            action_idx = self.agent.step(next_obs, reward, terminal)
            action_idx = 1

            # if episode has ended, then break
            if terminal:
                break

        # get feedback from agent
        self.agent.finish_episode()

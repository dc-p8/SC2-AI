import numpy as np
import threading
# from DQN import DQNAgent
from pysc2.env import sc2_env, run_loop
from DQN import DQNModel

import sys
from absl import flags
import Agent

FLAGS = flags.FLAGS
FLAGS(sys.argv)

n_worker = 1

class Worker(threading.Thread):
	def __init__(self, shared_model):
		super(Worker, self).__init__()
		self.model = shared_model

	def run(self):
		super(Worker, self).run()
		env = sc2_env.SC2Env(
			map_name='Simple64',
			players=[
				sc2_env.Agent(race=sc2_env.Race.terran),
				sc2_env.Bot(difficulty=sc2_env.Difficulty.very_easy, race=sc2_env.Race.random)],
			step_mul=16,
			agent_interface_format=sc2_env.AgentInterfaceFormat(
				use_feature_units=True,
				hide_specific_actions=False,
				feature_dimensions=sc2_env.Dimensions(
					screen=64,
					minimap=64
				)),
			game_steps_per_episode=0,
			visualize=True
		)
		agent = Agent.Agent(model=self.model)
		observation_spec = env.observation_spec()
		action_spec = env.action_spec()
		agent.setup(observation_spec[0], action_spec[0])
		run_loop.run_loop([agent], env)


# model = DQNModel(state_size=Agent.state_space,
# 				action_size=Agent.action_space,
# 				 replay_buffer_size=Agent.replay_buffer_size)

# workers = []
# for i in range(n_worker):
# 	worker = Worker(model)
# 	workers.append(worker)
# 	worker.start()
# for worker in workers:
# 	worker.join()


# Setup player with env

# try:
# 	while True:  # play n episodes
# 		episode_memory = []
# 		timeSteps = env.reset()
# 		agent.reset()
# 		final_score = 0
# 		game_won = False
# 		while True:  # play episode
# 			actions = [agent.step(timeSteps[0])]
# 			print('actions : ')
# 			print(actions)
# 			timeSteps = env.step(actions)
# 			done = timeSteps[0].last()
# 			if done:
# 				break
# 		# After game (episode) finished
# 		print('episode finished, saving model')
# except KeyboardInterrupt:
# 	pass

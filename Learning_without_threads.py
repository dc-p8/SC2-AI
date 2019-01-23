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
agent = Agent.Agent()
observation_spec = env.observation_spec()
action_spec = env.action_spec()
agent.setup(observation_spec[0], action_spec[0])
run_loop.run_loop([agent], env)


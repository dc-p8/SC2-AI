from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import numpy as np
import math
import random

from pysc2.agents import base_agent
from pysc2.lib import actions, features, units

from DQN import DQNModel

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5

_TERRAN_COMMANDCENTER = 18
_TERRAN_SCV = 45
_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_BARRACKS = 21
_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

batch_size = 1000
replay_buffer_size = 100000

PlayerIndices = [
	# don't pick player id, warp and larva
	1, 2, 3, 4, 5, 6, 7, 8
]
ScoreCumulativeIndices = [
	# don't pick spent_vespene
	0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12
]

s_hot_squares = np.zeros(4, dtype=np.int)
s_green_squares = np.zeros(4, dtype=np.int)
s_command_centers_count = 0
s_supply_depots_count = 0
s_barracks_count = 0
s_supply_free = 0

smart_actions = [
	ACTION_DO_NOTHING,
	ACTION_BUILD_SUPPLY_DEPOT,
	ACTION_BUILD_BARRACKS,
	ACTION_BUILD_MARINE,
]

for mm_x in range(0, 64):
	for mm_y in range(0, 64):
		if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
			smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))

state_space = len(PlayerIndices) +\
			  len(ScoreCumulativeIndices) +\
			  len(s_hot_squares) +\
			  len(s_green_squares) +\
			  len([s_command_centers_count,
				   s_supply_depots_count,
				   s_barracks_count,
				   s_supply_free])

action_space = len(smart_actions)

model_name = 'models/SC2Agent.hdf5'


class Agent(base_agent.BaseAgent):
	top_left_pos = (12, 16)
	bot_right_pos = (49, 49)

	def transformDistance(self, x, x_distance, y, y_distance):
		if not self.base_top_left:
			return [x - x_distance, y - y_distance]

		return [x + x_distance, y + y_distance]

	def transformLocation(self, x, y):
		if not self.base_top_left:
			return [64 - x, 64 - y]

		return [x, y]

	def splitAction(self, action_id):
		smart_action = smart_actions[action_id]

		x = 0
		y = 0
		if '_' in smart_action:
			smart_action, x, y = smart_action.split('_')

		return smart_action, x, y

	def modify_memory(self, memory, score, won):
		if won:
			memory[2] = int(memory[2] * 5)
		else:
			memory[2] = int(memory[2] / 5)
		return memory

	def fix_unit(self, unit):
		if unit.x < 0:
			unit.x = 0
		if unit.y < 0:
			unit.y = 0
		if unit.x >= self.obs_spec.feature_minimap[2]:
			unit.x = self.obs_spec.feature_minimap[2] - 1
		if unit.y >= self.obs_spec.feature_minimap[1]:
			unit.y = self.obs_spec.feature_minimap[1] - 1
		return unit

	def get_units_by_type(self, obs, unit_type):
		return [self.fix_unit(unit) for unit in obs.observation.feature_units if unit.unit_type == unit_type]

	def get_squares(self, obs):
		hot_squares = np.zeros(4)
		enemy_y, enemy_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.ENEMY).nonzero()
		for i in range(0, len(enemy_y)):
			y = int(math.ceil((enemy_y[i] + 1) / 32))
			x = int(math.ceil((enemy_x[i] + 1) / 32))
			hot_squares[((y - 1) * 2) + (x - 1)] = 1

		if not self.base_top_left:
			hot_squares = hot_squares[::-1]

		green_squares = np.zeros(4)
		friendly_y, friendly_x = (
			obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
		for i in range(0, len(friendly_y)):
			y = int(math.ceil((friendly_y[i] + 1) / 32))
			x = int(math.ceil((friendly_x[i] + 1) / 32))
			green_squares[((y - 1) * 2) + (x - 1)] = 1

		if not self.base_top_left:
			green_squares = green_squares[::-1]

		return hot_squares.tolist(), green_squares.tolist()

	def init_starting_pos(self, obs):
		player_y, player_x = (obs.observation.feature_minimap.player_relative == features.PlayerRelative.SELF).nonzero()
		xmean = player_x.mean()
		ymean = player_y.mean()

		if xmean < self.center_x and ymean < self.center_y:
			self.attack_coordinates = self.bot_right_pos
			self.base_top_left = True
		else:
			self.attack_coordinates = self.top_left_pos
			self.base_top_left = False

		self.first_command_center = self.get_units_by_type(obs, units.Terran.CommandCenter)[0]

	def setup(self, obs_spec, action_spec):
		super(Agent, self).setup(obs_spec, action_spec)
		self.size_minimap_y = self.obs_spec.feature_minimap[1]
		self.size_minimap_x = self.obs_spec.feature_minimap[2]
		self.center_y = int(self.size_minimap_y / 2)
		self.center_x = int(self.size_minimap_x / 2)

	def reset(self):
		super(Agent, self).reset()
		self.previous_action = None
		self.previous_state = None
		self.move_number = 0
		self.game_moves = []

	def step(self, obs):

		if obs.last():
			game_won = obs.reward > 0
			final_score = obs.observation.score_cumulative[0]
			episode_memory = [self.modify_memory(single_move, final_score, game_won) for single_move in self.game_moves]
			self.model.remember(episode_memory)  # append game memory to model
			self.model.replay(batch_size)  # apply model replay buffer
			self.model.save(model_name + '_' + str(self.episodes) + ('W' if game_won else 'L'))
			return actions.FunctionCall(_NO_OP, [])

		if obs.first():
			self.init_starting_pos(obs)

		command_centers = self.get_units_by_type(obs, units.Terran.CommandCenter)
		s_command_centers_count = len(command_centers)

		supply_depots = self.get_units_by_type(obs, units.Terran.SupplyDepot)
		s_supply_depots_count = len(supply_depots)

		barracks = self.get_units_by_type(obs, units.Terran.Barracks)
		s_barracks_count = len(barracks)

		supply_used = obs.observation['player'][3]
		supply_limit = obs.observation['player'][4]
		army_supply = obs.observation['player'][5]
		worker_supply = obs.observation['player'][6]

		s_supply_free = supply_limit - supply_used

		target = []

		if self.move_number == 0:
			self.move_number += 1

			s_hot_squares, s_green_squares = self.get_squares(obs)

			current_state = []

			current_state += [s_command_centers_count, s_supply_depots_count, s_barracks_count, s_supply_free]
			current_state += s_hot_squares
			current_state += s_green_squares
			current_state += [value for index, value in enumerate(obs.observation.player) if
							  index in PlayerIndices]
			current_state += [value for index, value in enumerate(obs.observation.score_cumulative) if
							  index in ScoreCumulativeIndices]

			current_state = [int(x) for x in current_state]

			if self.previous_action is not None:
				self.game_moves.append([self.previous_state, self.previous_action, obs.observation.score_cumulative[0], current_state])

			excluded_actions = []
			if worker_supply == 0:
				excluded_actions.append(1)

			if s_supply_depots_count == 0 or worker_supply == 0:
				excluded_actions.append(2)

			if s_supply_free == 0 or s_barracks_count == 0:
				excluded_actions.append(3)

			if army_supply == 0:
				excluded_actions += [4, 5, 6, 7]

			# chose the action to do for the next 2 game steps

			rl_action = self.model.act(current_state, excluded_actions)
			# print(rl_action)

			self.previous_state = current_state
			self.previous_action = rl_action

			smart_action, x, y = self.splitAction(self.previous_action)

			if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
				SCVs = self.get_units_by_type(obs, units.Terran.SCV)
				if SCVs:
					SCV = random.choice(SCVs)
					target = [SCV.x, SCV.y]
					return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

			elif smart_action == ACTION_BUILD_MARINE:
				if barracks:
					barrack = random.choice(barracks)
					target = [barrack.x, barrack.y]
					return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

			elif smart_action == ACTION_ATTACK:
				if _SELECT_ARMY in obs.observation.available_actions:
					return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

		elif self.move_number == 1:
			self.move_number += 1

			smart_action, x, y = self.splitAction(self.previous_action)

			if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
				if s_supply_depots_count < 2 and _BUILD_SUPPLY_DEPOT in obs.observation.available_actions:
					if command_centers:
						command_center = random.choice(command_centers)
						if s_supply_depots_count == 0:
							target = self.transformDistance(round(command_center.x), 10, round(command_center.y), 10)
						elif s_supply_depots_count == 1:
							target = self.transformDistance(round(command_center.x), -25, round(command_center.y), -25)
						return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

			elif smart_action == ACTION_BUILD_BARRACKS:
				if s_barracks_count < 2 and _BUILD_BARRACKS in obs.observation.available_actions:
					if command_centers:
						command_center = random.choice(command_centers)
						if s_barracks_count == 0:
							target = self.transformDistance(round(command_center.x), 15, round(command_center.y), -9)
						elif s_barracks_count == 1:
							target = self.transformDistance(round(command_center.x), 15, round(command_center.y), 12)
						return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

			elif smart_action == ACTION_BUILD_MARINE:
				if _TRAIN_MARINE in obs.observation.available_actions:
					return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

			elif smart_action == ACTION_ATTACK:
				do_it = True
				if len(obs.observation.single_select) > 0 and obs.observation.single_select[0][0] == _TERRAN_SCV:
					do_it = False
				if len(obs.observation.multi_select) > 0 and obs.observation.multi_select[0][0] == _TERRAN_SCV:
					do_it = False
				if do_it and _ATTACK_MINIMAP in obs.observation.available_actions:
					x_offset = random.randint(-1, 1)
					y_offset = random.randint(-1, 1)
					return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED,
																  self.transformLocation(int(x) + (x_offset * 8),
																						 int(y) + (y_offset * 8))])

		elif self.move_number == 2:
			self.move_number = 0
			smart_action, x, y = self.splitAction(self.previous_action)

			if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
				if _HARVEST_GATHER in obs.observation.available_actions:
					mineral_fields = self.get_units_by_type(obs, units.Neutral.MineralField)
					if mineral_fields:
						mineral_field = random.choice(mineral_fields)
						target = [int(mineral_field.x), int(mineral_field.y)]
						return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

		return actions.FunctionCall(_NO_OP, [])

	def __init__(self, model=None):
		super(Agent, self).__init__()
		#self.lock = threading.Lock()
		if not model:
			model = DQNModel(
				state_size=state_space,
				action_size=action_space,
				replay_buffer_size=replay_buffer_size)
		self.model = model
		self.model.load(model_name)

import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.optimizers import SGD
import random
import os

class DQNModel:
	def __init__(self, state_size, action_size, replay_buffer_size):
		self.state_size = state_size
		self.action_size = action_size
		self.memory = deque(maxlen=replay_buffer_size)
		self.gamma = 0.95  # Discount factor
		self.epsilon = 0.05  # Exploration / Exploitation rate
		self.learning_rate = 0.001
		self.model = self.build_model()

	def build_model(self):
		model = Sequential()
		model.add(Dense(128, input_dim=self.state_size, activation='relu'))  # 1st hidden layer; states as input
		model.add(Dense(128, activation='relu'))  # 2nd hidden layer
		model.add(Dense(128, activation='relu'))  # 2nd hidden layer
		model.add(Dense(self.action_size, activation='linear'))  # 2 actions, so 2 output neurons: 0 and 1 (L/R)
		model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
		return model

	def remember(self, array_sars_d):
		self.memory.extend(array_sars_d)

	def act(self, state, excluded_actions=None):
		if np.random.rand() <= self.epsilon:
			return random.randrange(self.action_size)
		act_values = self.model.predict(np.reshape(state, [1, len(state)]))[0]
		if excluded_actions:
			act_values = np.delete(act_values, excluded_actions)  # remove excluded_action indexes from predicted actions
		return np.argmax(act_values)

	def replay(self, batch_size=128):
		if len(self.memory) < batch_size :
			return
		minibatch = random.sample(self.memory, batch_size)
		for state, action, reward, next_state in minibatch:
			target = (reward + self.gamma * np.amax(
				self.model.predict(np.reshape(next_state, [1, len(next_state)]))[0]))  # (maximum target Q based on future action a')
			if np.isnan(target) or target != target:
				raise Exception()
			target_f = self.model.predict(np.reshape(state, [1, len(state)]))
			target_f[0][action] = target
			self.model.fit(np.reshape(state, [1, len(state)]), target_f, epochs=1, verbose=0)

	def load(self, name):
		if os.path.exists(name):
			self.model.load_weights(name)
			print(name + ' loaded')

	def save(self, name):
		self.model.save_weights(name)

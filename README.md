# SC2-AI
Attempt to replace the Q learning algorithm used in this [tutorial](https://github.com/skjb/pysc2-tutorial/blob/master/Refining%20the%20Sparse%20Reward%20Agent/refined_agent.py) by DQN.
My main source of information for DQN is [this guy](https://www.youtube.com/watch?v=OYhFoMySoVs) on youtube who make a network that play the cartpole game : https://github.com/the-deep-learners/TensorFlow-LiveLessons/blob/master/notebooks/live_training/cartpole_dqn_LT.ipynb

The main difference between the two DQN implementations is that I need stock the *state, action, reward, next_state* tuple in order to modify it, depending on the last observation, in which I know if the game is won or lost.

-	The inputs of the neural network are the different variables representing the state of the game.
-	The outputs are the differents Q values of each smart actions given a state.

**The project is currently a failure :** I still haven't found the hyperparameters to make this network functional. It is possible that there is a problem elsewhere in the code (changes had to be made on the PYSC2 side to be able to change the alhorithm used by the agent)

The main problem is that, since AI never wins and its goal is to maximize rewards, it cannot begin its learning process.
I tried to fix the problem by replacing the initial rewards 1 (win) / 0 (lose) by : final score * 5 (win) and final score / 5 (lose) but I don't feel like the AI plays better.

## Dependencies
-	pysc2
-	keras
-	tensorflow
-	hdf5 (to load and save a model)

## Areas for improvement
-	Obviously, increase the action space. With 4 possible "smart" actions, you'll never get far
-	Use the camera layers with a convolutional network to represent the game state
